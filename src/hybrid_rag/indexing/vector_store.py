"""Thin wrapper over Qdrant: one collection, one cosine-distance vector per chunk, and the whole
chunk record (text + metadata) stored as the point's payload so a search returns everything a
citation needs."""

import atexit
import uuid
from functools import cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

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


def sync(chunks: list[dict], vectors: list[list[float]], batch: int = 256) -> None:
    """Make the collection hold exactly these chunks: upsert all, then delete any other point.

    We don't rely on delete_collection to clear old data: on Windows the embedded store's
    SQLite file can stay open, the folder delete silently fails, and the "new" collection
    reopens the old file (seen: 251 stale points from a previous chunking survived). Deleting
    by id and checking the count works the same on embedded, Docker and Cloud Qdrant.
    """
    name, size = settings.qdrant_collection, len(vectors[0])
    if client().collection_exists(name):
        if client().get_collection(name).config.params.vectors.size != size:
            client().delete_collection(name)  # different embedding model: start over
    if not client().collection_exists(name):
        client().create_collection(
            name, vectors_config=VectorParams(size=size, distance=Distance.COSINE)
        )
    points = [
        PointStruct(id=point_id(c["chunk_id"]), vector=v, payload=c)
        for c, v in zip(chunks, vectors, strict=True)
    ]
    for i in range(0, len(points), batch):
        client().upsert(name, points=points[i : i + batch])
    delete_stale({p.id for p in points})


def delete_stale(keep: set[str]) -> int:
    """Delete points whose id isn't in `keep`; fail if the count still doesn't match."""
    name = settings.qdrant_collection
    stale, offset = [], None
    while True:
        page, offset = client().scroll(name, limit=1000, offset=offset)
        stale += [p.id for p in page if str(p.id) not in keep]
        if offset is None:
            break
    if stale:
        client().delete(name, points_selector=PointIdsList(points=stale))
    count = client().count(name, exact=True).count
    if count != len(keep):
        raise RuntimeError(f"Qdrant holds {count} points, expected {len(keep)}")
    return len(stale)


def search(vector: list[float], top_k: int) -> list[dict]:
    """Returns chunk payloads, best first, each with its cosine similarity as "score"."""
    res = client().query_points(
        settings.qdrant_collection, query=vector, limit=top_k, with_payload=True
    )
    return [{**p.payload, "score": p.score} for p in res.points]
