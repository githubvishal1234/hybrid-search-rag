"""Build the search indexes from data/processed/chunks.jsonl.

Run: uv run python -m hybrid_rag.indexing.build            # BM25 (seconds) + vectors (~7 min)
     uv run python -m hybrid_rag.indexing.build --only bm25

ponytail: rebuilds each index from scratch every time (~2.4k chunks).
Switch to upserting only changed docs if the corpus grows past ~50k chunks.
"""

import argparse
import json
import logging
import time

from hybrid_rag.config import settings
from hybrid_rag.indexing import bm25_index, embed, vector_store

log = logging.getLogger(__name__)


def build(only: str | None = None) -> None:
    with (settings.processed_dir / "chunks.jsonl").open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    if only in (None, "bm25"):
        t0 = time.perf_counter()
        bm25_index.build(chunks)
        log.info("BM25: indexed %d chunks in %.1fs", len(chunks), time.perf_counter() - t0)

    if only in (None, "vector"):
        t0 = time.perf_counter()
        vectors = embed.embed_passages([embed.passage_text(c) for c in chunks])
        log.info("vectors: embedded %d chunks in %.0fs", len(chunks), time.perf_counter() - t0)
        vector_store.recreate(chunks, vectors)
        log.info("vectors: indexed into Qdrant collection %r", settings.qdrant_collection)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["bm25", "vector"])
    build(p.parse_args().only)
