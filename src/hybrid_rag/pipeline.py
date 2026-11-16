"""The one door into the RAG system: run_rag(question, config) -> RagResult.

The API, the UI, the CLI and the eval all call this, so what gets evaluated is exactly what
ships. RagConfig holds every query-time knob (an experiment is just a different RagConfig);
anything fixed when the index is built, like chunk size, is not here because it can't change
per question. RagResult keeps everything an evaluator needs: the context chunks with their
text and per-retriever ranks (retrieval metrics, faithfulness), validated citations (citation
accuracy), the refusal flag, latency per stage and token counts (cost).
"""

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hybrid_rag.config import settings
from hybrid_rag.generation import llm, prompts
from hybrid_rag.generation.answer import resolve_citations
from hybrid_rag.retrieval.retriever import Mode, retrieve


def _setting(name: str):
    # Read config.py when a RagConfig is created, not at import, so .env and test overrides apply.
    return Field(default_factory=lambda: getattr(settings, name))


class RagConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo in an experiment file fails loudly

    mode: Mode = "hybrid"
    context_top_n: int = _setting("context_top_n")
    dense_top_k: int = _setting("dense_top_k")
    bm25_top_k: int = _setting("bm25_top_k")
    rrf_k: int = _setting("rrf_k")
    rrf_bm25_weight: float = _setting("rrf_bm25_weight")
    rerank_candidates: int = _setting("rerank_candidates")
    reranker_provider: Literal["local", "cohere"] = _setting("reranker_provider")
    reranker_model: str = _setting("reranker_model")
    llm_provider: Literal["openai", "anthropic", "gemini", "ollama"] = _setting("llm_provider")
    llm_model: str = _setting("llm_model")
    prompt_version: prompts.PromptVersion = "v1"


class Context(BaseModel):
    """One chunk the LLM was given, in the order it was given."""

    rank: int
    chunk_id: str
    file_name: str
    title: str
    page_start: int
    page_end: int
    section_heading: str
    text: str
    score: float  # the number that decided its position (cosine, BM25, RRF or rerank score)
    vector_rank: int | None  # position in each retriever's list; None = not found by it
    bm25_rank: int | None
    rrf_rank: int | None  # only in hybrid_rerank: position before reranking
    rerank_score: float | None


class Citation(BaseModel):
    doc: str
    page: str = Field(examples=["p.12-13"])
    chunk_id: str
    snippet: str


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class RagResult(BaseModel):
    question: str
    config: RagConfig
    answer: str
    refused: bool
    citations: list[Citation]
    invalid_citations: list[int]  # source numbers cited but never given (dropped from answer)
    uncited: bool  # answered without citing anything
    contexts: list[Context]
    # dense / bm25 / fusion / rerank (those that ran), then retrieve, generate, total
    latency_ms: dict[str, int]
    tokens: Usage

    def to_response(self) -> dict:
        """The /query response shape (api/schemas.py QueryResponse)."""
        return {
            "answer": self.answer,
            "refused": self.refused,
            "citations": [c.model_dump() for c in self.citations],
            "invalid_citations": self.invalid_citations,
            "uncited": self.uncited,
            "retrieval_debug": {
                "mode": self.config.mode,
                "chunks": [
                    {
                        "rank": c.rank,
                        "chunk_id": c.chunk_id,
                        "vector_rank": c.vector_rank,
                        "bm25_rank": c.bm25_rank,
                        "rrf_rank": c.rrf_rank,
                        "rerank_score": c.rerank_score,
                        "score": round(c.score, 4),
                    }
                    for c in self.contexts
                ],
            },
            "latency_ms": {k: self.latency_ms[k] for k in ("retrieve", "generate", "total")},
            "tokens": self.tokens.model_dump(),
        }


def _context(rank: int, c: dict) -> Context:
    return Context(
        rank=rank,
        chunk_id=c["chunk_id"],
        file_name=c["file_name"],
        title=c["title"],
        page_start=c["page_start"],
        page_end=c["page_end"],
        section_heading=c.get("section_heading") or "",
        text=c["text"],
        score=c["score"],
        vector_rank=c["ranks"].get("vector"),
        bm25_rank=c["ranks"].get("bm25"),
        rrf_rank=c["ranks"].get("rrf"),
        rerank_score=c.get("rerank_score"),
    )


def run_rag(
    question: str,
    config: RagConfig | None = None,
    *,
    generate: bool = True,
    retrieve_n: int | None = None,
) -> RagResult:
    """Retrieve -> prompt -> LLM -> validated, cited answer. generate=False stops after
    retrieval (answer empty, no LLM call): what retrieval-only evals and CI need.

    retrieve_n (eval only) ranks deeper than the LLM sees, so one call yields both Hit@10 and an
    answer: contexts then holds retrieve_n chunks and the LLM got contexts[:context_top_n]."""
    config = config or RagConfig()
    timings: dict[str, int] = {}
    t0 = time.perf_counter()
    ranked = retrieve(
        question,
        config.mode,
        max(retrieve_n or 0, config.context_top_n),
        dense_top_k=config.dense_top_k,
        bm25_top_k=config.bm25_top_k,
        rrf_k=config.rrf_k,
        rrf_bm25_weight=config.rrf_bm25_weight,
        rerank_candidates=config.rerank_candidates,
        reranker_provider=config.reranker_provider,
        reranker_model=config.reranker_model,
        timings=timings,
    )
    chunks = ranked[: config.context_top_n]  # what the LLM sees
    t1 = time.perf_counter()

    text, citations, invalid, refused, usage = "", [], [], False, {}
    if generate:
        raw, usage = llm.complete_with_usage(
            prompts.SYSTEMS[config.prompt_version],
            prompts.user_prompt(question, chunks),
            config.llm_provider,
            config.llm_model,
        )
        text, citations, invalid = resolve_citations(raw, chunks)
        refused = prompts.REFUSAL.lower().rstrip(".") in raw.lower()
    t2 = time.perf_counter()

    return RagResult(
        question=question,
        config=config,
        answer=text,
        refused=refused,
        citations=citations,
        invalid_citations=invalid,
        uncited=generate and not refused and not citations,  # answered without grounding
        contexts=[_context(i, c) for i, c in enumerate(ranked, start=1)],
        latency_ms={
            **timings,
            "retrieve": round((t1 - t0) * 1000),
            "generate": round((t2 - t1) * 1000),
            "total": round((t2 - t0) * 1000),
        },
        tokens=Usage(**usage),
    )
