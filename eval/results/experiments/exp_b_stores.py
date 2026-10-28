"""Experiment B: copy the live vectors into new embedded stores, leaving out number-only chunks.

python exp_b_stores.py <out_dir>   -> <out_dir>/skip030, <out_dir>/skip040
"""

import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from hybrid_rag.config import settings


def letter_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace() and c not in "|-"]
    return sum(c.isalpha() for c in chars) / max(1, len(chars))


src = QdrantClient(path=str(settings.qdrant_path))
points, offset = [], None
while True:
    page, offset = src.scroll(
        settings.qdrant_collection, limit=1000, offset=offset, with_vectors=True
    )
    points += page
    if offset is None:
        break
src.close()
print("live points:", len(points))

for name, threshold in (("skip030", 0.30), ("skip040", 0.40)):
    keep = [p for p in points if letter_ratio(p.payload["text"]) >= threshold]
    dst = QdrantClient(path=str(Path(sys.argv[1]) / name))
    if dst.collection_exists(settings.qdrant_collection):
        dst.delete_collection(settings.qdrant_collection)
    dst.create_collection(
        settings.qdrant_collection,
        vectors_config=VectorParams(size=len(keep[0].vector), distance=Distance.COSINE),
    )
    structs = [PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in keep]
    for i in range(0, len(structs), 512):
        dst.upsert(settings.qdrant_collection, points=structs[i : i + 512])
    print(name, "kept", dst.count(settings.qdrant_collection, exact=True).count)
    dst.close()
