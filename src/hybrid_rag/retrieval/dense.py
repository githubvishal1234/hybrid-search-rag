"""Vector (semantic) retrieval: embed the query, nearest neighbours by cosine similarity."""

from hybrid_rag.config import settings
from hybrid_rag.indexing import embed, vector_store


def dense_search(query: str, top_k: int | None = None) -> list[dict]:
    return vector_store.search(embed.embed_query(query), top_k or settings.dense_top_k)
