"""RAGAS 0.4 generation metrics, cached on disk, with a meter for judge calls and tokens.

Needs the optional extra: uv sync --extra eval (RAGAS pulls in LangChain; the app never does).

- faithfulness:      share of the answer's claims supported by the contexts the LLM saw.
- answer_relevancy:  does the answer address the question? RAGAS writes questions the answer would
                     answer and compares them with the real one by embedding (our local bge-small,
                     so this part is free).
- context_precision: are the chunks that help the reference answer ranked near the top?
- context_recall:    does the context contain what the reference answer needs?

Which apply: context_* need a reference answer, so answerable questions only; faithfulness and
answer_relevancy need an answer, so refused answers are skipped (refusal accuracy covers them).
Citation markers "[file.pdf, p.3]" are stripped first: they're not claims.

The judge is any OpenAI-compatible model (Gemini, OpenAI, Ollama) at temperature ~0. Keep it
fixed across experiments: a different judge gives different numbers. Scores are cached by a hash
of (judge, metric, inputs) in eval/results/ragas_cache.jsonl, so re-running is free and an
interrupted run resumes.
"""

import asyncio
import hashlib
import json
import re
from pathlib import Path

import httpx
import openai
from ragas.embeddings import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

from hybrid_rag.config import ROOT, settings
from hybrid_rag.generation.llm import GEMINI_OPENAI_URL
from hybrid_rag.indexing import embed

CACHE = ROOT / "eval" / "results" / "ragas_cache.jsonl"
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
CITATION = re.compile(r"\s*\[[^\[\]]+\.pdf, p\.[\d-]+\]")
# Gemini free tier: requests per day per model (500 for flash-lite). Per-minute 429s are retried
# by the SDK; a per-day one lasts until the quota resets, so stop asking.
DAILY_QUOTA = "PerDay"


def is_daily_quota(e: BaseException | None) -> bool:
    """True if this error, or one it wraps (instructor wraps the SDK's), is a per-day 429."""
    while e is not None:
        if DAILY_QUOTA in str(e):
            return True
        e = e.__cause__ or e.__context__
    return False


class LocalEmbeddings(BaseRagasEmbedding):
    """The index's own bge-small (ONNX via fastembed): no API calls for answer_relevancy."""

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return embed.embed_passages([text])[0]

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return self.embed_text(text)


class Meter:
    """Counts judge HTTP calls and the tokens they report (for the budget decision)."""

    def __init__(self):
        self.calls = self.input_tokens = self.output_tokens = 0

    async def __call__(self, response: httpx.Response) -> None:
        if response.request.url.path.endswith("/chat/completions") and response.is_success:
            await response.aread()
            usage = response.json().get("usage") or {}
            self.calls += 1
            self.input_tokens += usage.get("prompt_tokens") or 0
            self.output_tokens += usage.get("completion_tokens") or 0


def judge_client(provider: str, meter: Meter) -> openai.AsyncOpenAI:
    base_url, key = {
        "gemini": (GEMINI_OPENAI_URL, settings.gemini_api_key),
        "openai": (None, settings.openai_api_key),
        "ollama": (f"{settings.ollama_base_url}/v1", "ollama"),
    }[provider]
    # max_retries: the SDK backs off on 429s, which the Gemini free tier (15/min) sends often
    return openai.AsyncOpenAI(
        base_url=base_url,
        api_key=key,
        max_retries=8,
        http_client=httpx.AsyncClient(timeout=120, event_hooks={"response": [meter]}),
    )


class RagasScorer:
    def __init__(self, provider: str, model: str, cache_path: Path = CACHE):
        self.judge = f"{provider}:{model}"
        self.meter = Meter()
        llm = llm_factory(
            model,
            provider="openai",  # all three speak the OpenAI API
            client=judge_client(provider, self.meter),
            adapter="instructor",
            temperature=0.0,
            max_tokens=4096,  # claim lists for long answers overflow the 1024 default
        )
        self.metrics = {
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=LocalEmbeddings()),
            "context_precision": ContextPrecisionWithReference(llm=llm),
            "context_recall": ContextRecall(llm=llm),
        }
        self.cache_path = cache_path
        self.cache: dict[str, float | None] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row["score"]
        self.errors: list[str] = []
        self.quota_exhausted = False  # set on the first per-day 429; later calls are skipped
        self.loop = asyncio.new_event_loop()  # one loop: the async HTTP client is bound to it

    def _key(self, metric: str, inputs: dict) -> str:
        blob = json.dumps([self.judge, metric, inputs], sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode()).hexdigest()

    def _score(self, metric: str, inputs: dict) -> float | None:
        key = self._key(metric, inputs)
        if key in self.cache:
            return self.cache[key]
        if self.quota_exhausted:
            return None  # every further call would fail too; a re-run tomorrow fills it in
        try:
            coro = self.metrics[metric].ascore(**inputs)
            value = self.loop.run_until_complete(coro).value
            score = None if value is None or value != value else float(value)  # NaN -> None
        except Exception as e:  # one bad judge reply shouldn't kill a 200-question run
            if is_daily_quota(e):
                self.quota_exhausted = True
            else:
                self.errors.append(f"{metric}: {type(e).__name__}: {str(e)[:200]}")
            return None  # not cached, so a re-run retries it
        self.cache[key] = score
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "metric": metric, "score": score}) + "\n")
        return score

    def score(
        self, question: str, answer: str, contexts: list[str], reference: str, refused: bool
    ) -> dict[str, float | None]:
        """contexts: the texts the LLM saw, in rank order. reference: '' for unanswerable."""
        out: dict[str, float | None] = dict.fromkeys(METRICS)
        response = CITATION.sub("", answer).strip()
        if response and not refused:
            out["faithfulness"] = self._score(
                "faithfulness",
                {"user_input": question, "response": response, "retrieved_contexts": contexts},
            )
            out["answer_relevancy"] = self._score(
                "answer_relevancy", {"user_input": question, "response": response}
            )
        if reference:
            ctx = {"user_input": question, "reference": reference, "retrieved_contexts": contexts}
            out["context_precision"] = self._score("context_precision", ctx)
            out["context_recall"] = self._score("context_recall", ctx)
        return out
