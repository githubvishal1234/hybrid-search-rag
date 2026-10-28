"""Experiment A: re-embed every chunk as "section heading + text" (no document title) into a
separate embedded store: <out_dir>/notitle. The production index is not touched."""

import json
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from hybrid_rag.config import settings
from hybrid_rag.indexing import embed
from hybrid_rag.indexing.vector_store import point_id

with (settings.processed_dir / "chunks.jsonl").open(encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

texts = [
    f"{c['section_heading']}\n{c['text']}" if c["section_heading"] else c["text"] for c in chunks
]
t0 = time.perf_counter()
vectors = embed.embed_passages(texts)
print(f"embedded {len(chunks)} chunks in {time.perf_counter() - t0:.0f}s", flush=True)

dst = QdrantClient(path=str(Path(sys.argv[1]) / "notitle"))
if dst.collection_exists(settings.qdrant_collection):
    dst.delete_collection(settings.qdrant_collection)
dst.create_collection(
    settings.qdrant_collection,
    vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
)
points = [
    PointStruct(id=point_id(c["chunk_id"]), vector=v, payload=c)
    for c, v in zip(chunks, vectors, strict=True)
]
for i in range(0, len(points), 512):
    dst.upsert(settings.qdrant_collection, points=points[i : i + 512])
print("notitle stored", dst.count(settings.qdrant_collection, exact=True).count, flush=True)
dst.close()
