"""Dense embeddings with fastembed (ONNX runtime, no PyTorch): small image, fast on CPU.

bge is a bi-encoder: query and passage are embedded separately and compared with cosine
similarity, which is what makes searching thousands of chunks in milliseconds possible.
"""

from functools import cache

from fastembed import TextEmbedding

from hybrid_rag.config import settings


@cache
def model() -> TextEmbedding:
    return TextEmbedding(settings.embedding_model)


def passage_text(chunk: dict) -> str:
    """What gets embedded for a chunk. Title and section give the vector context the chunk text
    alone often lacks ("the said section" -> which Act?). chunk_size leaves room for this."""
    header = " / ".join(x for x in (chunk["title"], chunk["section_heading"]) if x)
    return f"{header}\n{chunk['text']}"


def embed_passages(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in model().embed(texts, batch_size=32)]


def embed_query(query: str) -> list[float]:
    return next(iter(model().embed([settings.query_prefix + query]))).tolist()
