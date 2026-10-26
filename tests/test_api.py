import openai
import pytest
from fastapi.testclient import TestClient

from hybrid_rag.api import main
from hybrid_rag.config import settings

FAKE = {
    "answer": "LTV is 75% [gold.pdf, p.3].",
    "refused": False,
    "citations": [{"doc": "gold.pdf", "page": "p.3", "chunk_id": "gold:0001", "snippet": "..."}],
    "invalid_citations": [],
    "uncited": False,
    "retrieval_debug": {
        "mode": "hybrid",
        "chunks": [
            {
                "rank": 1,
                "chunk_id": "gold:0001",
                "vector_rank": 2,
                "bm25_rank": 1,
                "rrf_rank": None,
                "rerank_score": None,
                "score": 0.0325,
            }
        ],
    },
    "latency_ms": {"retrieve": 20, "generate": 900, "total": 920},
}


@pytest.fixture
def client(monkeypatch):
    main._hits.clear()
    monkeypatch.setattr(main, "answer", lambda q, mode: FAKE)
    return TestClient(main.app)  # no `with`: skips the lifespan warm-up (no models in tests)


def test_query_returns_the_contract(client):
    r = client.post("/query", json={"question": "What is the LTV cap?", "mode": "hybrid"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"][0]["page"] == "p.3"
    assert body["retrieval_debug"]["chunks"][0]["bm25_rank"] == 1


@pytest.mark.parametrize(
    "payload",
    [{"question": "hi"}, {"question": "x" * 1001}, {"question": "valid?", "mode": "keyword"}, {}],
)
def test_query_rejects_bad_input(client, payload):
    assert client.post("/query", json=payload).status_code == 422


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
    codes = [client.post("/query", json={"question": "LTV cap?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_llm_down_is_a_503_not_a_500(client, monkeypatch):
    def down(q, mode):
        raise openai.APIConnectionError(request=None)

    monkeypatch.setattr(main, "answer", down)
    r = client.post("/query", json={"question": "LTV cap?"})
    assert r.status_code == 503 and "unreachable" in r.json()["detail"]


def test_ingest_needs_admin_token(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_token", None)
    assert client.post("/ingest").status_code == 403  # disabled unless configured
    monkeypatch.setattr(settings, "admin_token", "s3cret")
    assert client.post("/ingest", headers={"X-Admin-Token": "nope"}).status_code == 401
    assert client.get("/ingest").json() == {"running": False, "last_result": None}
