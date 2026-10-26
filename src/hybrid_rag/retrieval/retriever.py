"""One entry point for every retrieval mode, so the CLI, API and eval compare like with like.

    vector         dense top-k                                   (semantic)
    bm25           BM25 top-k                                    (keyword)
    hybrid         RRF(dense top-k, BM25 top-k)                  (both)
    hybrid_rerank  top RRF candidates re-scored by a cross-encoder (precise, slower)

Every returned chunk carries "ranks" ({"vector": 3, "bm25": 1, "rrf": 2}) and "score", the
number that decided its final position (cosine, BM25, RRF or cross-encoder score).
"""

from typing import Literal

from hybrid_rag.config import settings
from hybrid_rag.retrieval.dense import dense_search
from hybrid_rag.retrieval.fusion import rrf
from hybrid_rag.retrieval.rerank import rerank
from hybrid_rag.retrieval.sparse import bm25_search

Mode = Literal["vector", "bm25", "hybrid", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("vector", "bm25", "hybrid", "hybrid_rerank")


def _with_ranks(hits: list[dict], name: str) -> list[dict]:
    return [{**h, "ranks": {name: i}} for i, h in enumerate(hits, start=1)]


def retrieve(query: str, mode: Mode = "hybrid", top_n: int | None = None) -> list[dict]:
    """The top_n chunks to put in front of the LLM."""
    top_n = top_n or settings.context_top_n
    if mode == "vector":
        return _with_ranks(dense_search(query, settings.dense_top_k), "vector")[:top_n]
    if mode == "bm25":
        return _with_ranks(bm25_search(query, settings.bm25_top_k), "bm25")[:top_n]
    if mode not in ("hybrid", "hybrid_rerank"):
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    lists = {
        "vector": dense_search(query, settings.dense_top_k),
        "bm25": bm25_search(query, settings.bm25_top_k),
    }
    fused = rrf(lists, settings.rrf_k)
    if mode == "hybrid":
        return fused[:top_n]

    candidates = [
        {**c, "ranks": {**c["ranks"], "rrf": i}}
        for i, c in enumerate(fused[: settings.rerank_candidates], start=1)
    ]
    return [{**c, "score": c["rerank_score"]} for c in rerank(query, candidates, top_n)]
