"""Keyword retrieval: BM25 over the finance-aware tokenizer (see indexing/bm25_index.py)."""

from hybrid_rag.config import settings
from hybrid_rag.indexing import bm25_index


def bm25_search(query: str, top_k: int | None = None) -> list[dict]:
    return bm25_index.search(query, top_k or settings.bm25_top_k)
