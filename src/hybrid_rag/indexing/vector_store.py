"""Thin wrapper over Qdrant: one collection, one cosine-distance vector per chunk, and the whole
chunk record (text + metadata) stored as the point's payload so a search returns everything a
citation needs."""

import atexit
import uuid
from functools import cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from hybrid_rag.config import settings


@cache
def client() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    c = QdrantClient(path=str(settings.qdrant_path))  # embedded, single process at a time
    atexit.register(c.close)  # else its __del__ runs mid-shutdown and prints an ImportError
    return c


def point_id(chunk_id: str) -> str:
    # Qdrant ids must be ints or UUIDs; uuid5 maps our readable chunk_id to a stable UUID.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def recreate(chunks: list[dict], vectors: list[list[float]], batch: int = 256) -> None:
    name = settings.qdrant_collection
    if client().collection_exists(name):
        client().delete_collection(name)
    client().create_collection(
        name, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
    )
    points = [
        PointStruct(id=point_id(c["chunk_id"]), vector=v, payload=c)
        for c, v in zip(chunks, vectors, strict=True)
    ]
    for i in range(0, len(points), batch):
        client().upsert(name, points=points[i : i + batch])


def search(vector: list[float], top_k: int) -> list[dict]:
    """Returns chunk payloads, best first, each with its cosine similarity as "score"."""
    res = client().query_points(
        settings.qdrant_collection, query=vector, limit=top_k, with_payload=True
    )
    return [{**p.payload, "score": p.score} for p in res.points]
