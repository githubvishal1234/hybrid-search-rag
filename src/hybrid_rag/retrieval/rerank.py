"""Cross-encoder reranking: re-score the fused candidates by reading query and chunk together.

Bi-encoder (embeddings) vs cross-encoder: the embedder turns query and chunk into vectors
separately, so it can pre-compute every chunk and search millions in milliseconds, but it never
sees the two texts side by side. A cross-encoder feeds "query [SEP] chunk" through the model as
one input, so attention can match "borrow relative to the size of the economy" against
"debt-to-GDP ratio" word by word. Much more accurate, but it costs one model pass per pair
(~0.5 s per chunk here on CPU), which is why it only re-scores the top ~30 fused candidates.
"""

import json
import urllib.request
from functools import cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from hybrid_rag.config import settings
from hybrid_rag.indexing.embed import passage_text

COHERE_URL = "https://api.cohere.com/v2/rerank"


@cache
def _local_model() -> TextCrossEncoder:
    return TextCrossEncoder(settings.reranker_model)


def _local_scores(query: str, docs: list[str]) -> list[float]:
    return [float(s) for s in _local_model().rerank(query, docs)]


def _cohere_scores(query: str, docs: list[str]) -> list[float]:
    body = {"model": settings.cohere_rerank_model, "query": query, "documents": docs}
    req = urllib.request.Request(
        COHERE_URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {settings.cohere_api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        results = json.load(resp)["results"]
    scores = [0.0] * len(docs)  # Cohere returns results sorted by score, each with its index
    for r in results:
        scores[r["index"]] = r["relevance_score"]
    return scores


def rerank(query: str, chunks: list[dict], top_n: int) -> list[dict]:
    """Best top_n chunks by cross-encoder score, each with "rerank_score" added."""
    if not chunks:
        return []
    docs = [passage_text(c) for c in chunks]  # same "title / section + text" the embedder saw
    score = _cohere_scores if settings.reranker_provider == "cohere" else _local_scores
    scored = [{**c, "rerank_score": s} for c, s in zip(chunks, score(query, docs), strict=True)]
    return sorted(scored, key=lambda c: c["rerank_score"], reverse=True)[:top_n]
