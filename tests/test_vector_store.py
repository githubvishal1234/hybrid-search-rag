import pytest
from qdrant_client import QdrantClient

from hybrid_rag.indexing import vector_store


def chunk(i: int) -> dict:
    return {"chunk_id": f"doc:{i:04d}", "text": f"text {i}"}


@pytest.fixture
def memory_client(monkeypatch):
    c = QdrantClient(":memory:")
    monkeypatch.setattr(vector_store, "client", lambda: c)
    return c


def test_sync_removes_points_from_an_older_chunking(memory_client):
    vector_store.sync([chunk(0), chunk(1), chunk(2)], [[1.0, 0.0]] * 3)
    vector_store.sync([chunk(0), chunk(1)], [[0.0, 1.0]] * 2)  # re-chunked: doc:0002 is gone

    name = vector_store.settings.qdrant_collection
    assert memory_client.count(name, exact=True).count == 2
    hit = vector_store.search([0.0, 1.0], top_k=5)
    assert {h["chunk_id"] for h in hit} == {"doc:0000", "doc:0001"}


def test_sync_starts_over_when_the_embedding_size_changes(memory_client):
    vector_store.sync([chunk(0)], [[1.0, 0.0]])
    vector_store.sync([chunk(0)], [[1.0, 0.0, 0.0]])  # e.g. switched to a bigger model
    assert vector_store.search([1.0, 0.0, 0.0], top_k=1)[0]["chunk_id"] == "doc:0000"
