"""One entry point for every retrieval mode, so the CLI, API and eval compare like with like.

    vector         dense top-k                                   (semantic)
    bm25           BM25 top-k                                    (keyword)
    hybrid         RRF(dense top-k, BM25 top-k)                  (both)
    hybrid_rerank  top RRF candidates re-scored by a cross-encoder (precise, slower)

Every returned chunk carries "ranks" ({"vector": 3, "bm25": 1, "rrf": 2}) and "score", the
number that decided its final position (cosine, BM25, RRF or cross-encoder score).

Every knob defaults to config.py; passing it explicitly lets one process evaluate several
configurations (hybrid_rag.pipeline.RagConfig) without touching the global settings.
"""

import time
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


def retrieve(
    query: str,
    mode: Mode = "hybrid",
    top_n: int | None = None,
    *,
    dense_top_k: int | None = None,
    bm25_top_k: int | None = None,
    rrf_k: int | None = None,
    rrf_bm25_weight: float | None = None,
    rerank_candidates: int | None = None,
    reranker_provider: str | None = None,
    reranker_model: str | None = None,
    timings: dict[str, int] | None = None,
) -> list[dict]:
    """The top_n chunks to put in front of the LLM. If `timings` is given, it is filled with
    milliseconds per stage that ran ("dense", "bm25", "fusion", "rerank")."""
    top_n = top_n or settings.context_top_n
    timings = {} if timings is None else timings

    def timed(stage: str, fn, *args):
        t0 = time.perf_counter()
        out = fn(*args)
        timings[stage] = round((time.perf_counter() - t0) * 1000)
        return out

    dense_k = dense_top_k or settings.dense_top_k
    bm25_k = bm25_top_k or settings.bm25_top_k
    if mode == "vector":
        return _with_ranks(timed("dense", dense_search, query, dense_k), "vector")[:top_n]
    if mode == "bm25":
        return _with_ranks(timed("bm25", bm25_search, query, bm25_k), "bm25")[:top_n]
    if mode not in ("hybrid", "hybrid_rerank"):
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    lists = {
        "vector": timed("dense", dense_search, query, dense_k),
        "bm25": timed("bm25", bm25_search, query, bm25_k),
    }
    k = rrf_k or settings.rrf_k
    weight = settings.rrf_bm25_weight if rrf_bm25_weight is None else rrf_bm25_weight
    fused = timed("fusion", rrf, lists, k, {"bm25": weight})
    if mode == "hybrid":
        return fused[:top_n]

    candidates = [
        {**c, "ranks": {**c["ranks"], "rrf": i}}
        for i, c in enumerate(fused[: rerank_candidates or settings.rerank_candidates], start=1)
    ]
    t0 = time.perf_counter()
    reranked = rerank(query, candidates, top_n, reranker_provider, reranker_model)
    timings["rerank"] = round((time.perf_counter() - t0) * 1000)
    return [{**c, "score": c["rerank_score"]} for c in reranked]
