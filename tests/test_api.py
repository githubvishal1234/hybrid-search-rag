import openai
import pytest
from fastapi.testclient import TestClient

from hybrid_rag import pipeline
from hybrid_rag.api import main
from hybrid_rag.config import settings

GOLD = {
    "chunk_id": "gold:0001",
    "file_name": "gold.pdf",
    "title": "Gold loans",
    "page_start": 3,
    "page_end": 3,
    "section_heading": "",
    "text": "LTV shall not exceed 75%.",
    "score": 0.0325,
    "ranks": {"vector": 2, "bm25": 1},
}


@pytest.fixture
def client(monkeypatch):
    # The real run_rag -> to_response path, with retrieval and the LLM faked.
    main._hits.clear()
    monkeypatch.setattr(pipeline, "retrieve", lambda *a, **kw: [GOLD])
    monkeypatch.setattr(
        pipeline.llm,
        "complete_with_usage",
        lambda *a: ("LTV is 75% [1].", {"input_tokens": 120, "output_tokens": 8}),
    )
    return TestClient(main.app)  # no `with`: skips the lifespan warm-up (no models in tests)


def test_query_returns_the_contract(client):
    r = client.post("/query", json={"question": "What is the LTV cap?", "mode": "hybrid"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "LTV is 75% [gold.pdf, p.3]."
    assert body["citations"][0]["page"] == "p.3"
    assert body["retrieval_debug"]["chunks"][0]["bm25_rank"] == 1
    assert body["tokens"] == {"input_tokens": 120, "output_tokens": 8}
    assert set(body["latency_ms"]) == {"retrieve", "generate", "total"}


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
    def down(q, config):
        raise openai.APIConnectionError(request=None)

    monkeypatch.setattr(main, "run_rag", down)
    r = client.post("/query", json={"question": "LTV cap?"})
    assert r.status_code == 503 and "unreachable" in r.json()["detail"]


def test_ingest_needs_admin_token(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_token", None)
    assert client.post("/ingest").status_code == 403  # disabled unless configured
    monkeypatch.setattr(settings, "admin_token", "s3cret")
    assert client.post("/ingest", headers={"X-Admin-Token": "nope"}).status_code == 401
    assert client.get("/ingest").json() == {"running": False, "last_result": None}
