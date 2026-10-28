"""Faithfulness of generated answers, judged by an LLM against the context each answer was given.

uv run python eval/judge.py eval/results/<run>-generation.csv

Faithfulness = share of an answer's factual claims that the sources support (the same idea as
RAGAS faithfulness, without its LangChain dependency tree). The judge splits the answer into
claims and marks each supported or not; an answer's score is supported / total, and a mode's
score is the mean over its answered (non-refused) questions. This is the check that catches an
unsupported claim sitting behind a valid citation, which citation accuracy cannot.

Judge outputs are cached in eval/results/judge_cache.jsonl keyed by judge, answer and context,
so re-running costs nothing and API judges are paid once. Set JUDGE_PROVIDER / JUDGE_MODEL to use a
stronger model than the one that wrote the answers.
"""

import csv
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import openai

from hybrid_rag.config import ROOT, settings
from hybrid_rag.generation import llm

CACHE = ROOT / "eval" / "results" / "judge_cache.jsonl"
CITATION = re.compile(r"\[[^\[\]]+\.pdf, p\.[\d-]+\]")

SYSTEM = """You check whether an answer is supported by the given sources. Split the answer into \
its individual factual claims (figures, dates, names, rules, conditions). For each claim decide \
whether the sources state or directly imply it. Ignore citations and filler like "According to \
the sources". Reply with a single JSON object only: \
{"claims": [{"claim": "...", "supported": true}, ...]}"""


def judge_name() -> str:
    provider = settings.judge_provider or settings.llm_provider
    return f"{provider}:{settings.judge_model or settings.llm_model}"


def parse_claims(text: str) -> list[dict] | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        claims = json.loads(match.group(0))["claims"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(claims, list):
        return None
    return [c for c in claims if isinstance(c, dict) and isinstance(c.get("supported"), bool)]


def score(claims: list[dict] | None) -> float | None:
    """Share of claims supported; None when the judge found nothing to check or failed."""
    if not claims:
        return None
    return sum(c["supported"] for c in claims) / len(claims)


def complete_with_backoff(user: str) -> str:
    """Free-tier APIs (Gemini) cap requests per minute; wait it out instead of crashing."""
    provider, model = settings.judge_provider, settings.judge_model
    for wait in (30, 60, 120):
        try:
            return llm.complete(SYSTEM, user, provider, model)
        except openai.RateLimitError:
            print(f"\n  rate limited, waiting {wait}s")
            time.sleep(wait)
    return llm.complete(SYSTEM, user, provider, model)


def judge_answer(answer: str, contexts: list[str], cache: dict) -> list[dict] | None:
    answer = CITATION.sub("", answer).strip()
    key = hashlib.sha1(json.dumps([judge_name(), answer, contexts]).encode()).hexdigest()
    if key in cache:
        return cache[key]
    sources = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
    user = f"Sources:\n\n{sources}\n\nAnswer to check:\n{answer}"
    claims = None
    for _ in range(2):  # one retry on unparseable output
        claims = parse_claims(complete_with_backoff(user))
        if claims is not None:
            break
    if claims is not None:  # don't cache failures, so a later run retries them
        cache[key] = claims
        with CACHE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "judge": judge_name(), "claims": claims}) + "\n")
    return claims


def load_cache() -> dict:
    if not CACHE.exists():
        return {}
    with CACHE.open(encoding="utf-8") as f:
        return {r["key"]: r["claims"] for r in map(json.loads, f)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    gen_csv = Path(sys.argv[1])
    with gen_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows and "context_ids" not in rows[0]:
        sys.exit("this generation run predates context_ids; re-run run_eval.py --generate")
    with (settings.processed_dir / "chunks.jsonl").open(encoding="utf-8") as f:
        texts = {c["chunk_id"]: c["text"] for c in map(json.loads, f)}
    if judge_name() == f"{settings.llm_provider}:{settings.llm_model}":
        print(f"WARNING: judge {judge_name()} is the answering model; scores are self-graded.")

    cache = load_cache()
    by_mode: dict[str, list[float]] = defaultdict(list)
    judged = [r for r in rows if not r.get("error") and r["refused"] == "False"]
    for n, r in enumerate(judged, start=1):
        print(f"\r  judging {n}/{len(judged)}", end="", flush=True)
        contexts = [texts.get(cid, "") for cid in r["context_ids"].split("|") if cid]
        s = score(judge_answer(r["answer"], contexts, cache))
        if s is not None:
            by_mode[r["mode"]].append(s)
    print()

    lines = [
        f"# Faithfulness of `{gen_csv.name}`",
        "",
        f"Judge: `{judge_name()}`. Share of claims supported by the answer's own context, "
        "averaged over answered (non-refused) questions.",
        "",
        "| Mode | Faithfulness | Answers judged |",
        "|---|---|---|",
    ]
    for mode, scores in by_mode.items():
        lines.append(f"| {mode} | {sum(scores) / len(scores):.3f} | {len(scores)} |")
    text = "\n".join(lines) + "\n"
    print(text)
    suffix = "-faithfulness.md"
    if judge_name() != f"{settings.llm_provider}:{settings.llm_model}":  # keep one report per judge
        suffix = "-faithfulness-" + re.sub(r"[^\w.-]", "_", judge_name()) + ".md"
    out = gen_csv.with_name(gen_csv.name.replace("-generation.csv", suffix))
    out.write_text(text, encoding="utf-8")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
