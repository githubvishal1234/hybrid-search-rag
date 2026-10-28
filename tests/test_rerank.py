import io
import json

from hybrid_rag.config import settings
from hybrid_rag.retrieval import rerank as rerank_mod
from hybrid_rag.retrieval import retriever


def chunk(i: str) -> dict:
    return {"chunk_id": i, "title": "Doc", "section_heading": "", "text": f"text {i}"}


def test_rerank_orders_by_cross_encoder_score(monkeypatch):
    monkeypatch.setattr(settings, "reranker_provider", "local")
    monkeypatch.setattr(rerank_mod, "_local_scores", lambda q, docs: [0.1, 2.5, -1.0])
    out = rerank_mod.rerank("q", [chunk("a"), chunk("b"), chunk("c")], top_n=2)
    assert [c["chunk_id"] for c in out] == ["b", "a"]
    assert out[0]["rerank_score"] == 2.5
    assert rerank_mod.rerank("q", [], top_n=5) == []


def test_cohere_scores_are_mapped_back_by_index(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout):
        sent["auth"] = req.headers["Authorization"]
        sent["body"] = json.loads(req.data)
        results = [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.4}]
        return io.BytesIO(json.dumps({"results": results}).encode())

    monkeypatch.setattr(settings, "reranker_provider", "cohere")
    monkeypatch.setattr(settings, "cohere_api_key", "test-key")
    monkeypatch.setattr(rerank_mod.urllib.request, "urlopen", fake_urlopen)
    out = rerank_mod.rerank("q", [chunk("a"), chunk("b"), chunk("c")], top_n=3)

    assert [c["chunk_id"] for c in out] == ["c", "a", "b"]  # b missing from results -> 0.0
    assert sent["auth"] == "Bearer test-key"
    assert sent["body"]["query"] == "q" and len(sent["body"]["documents"]) == 3


def test_hybrid_rerank_reranks_top_fused_candidates(monkeypatch):
    seen = {}

    def fake_rerank(query, candidates, top_n):
        seen["candidates"] = candidates
        best_last = [{**c, "rerank_score": float(i)} for i, c in enumerate(candidates)]
        return best_last[::-1][:top_n]

    monkeypatch.setattr(retriever, "dense_search", lambda q, k: [chunk("a"), chunk("b")])
    monkeypatch.setattr(retriever, "bm25_search", lambda q, k: [chunk("b"), chunk("c")])
    monkeypatch.setattr(retriever, "rerank", fake_rerank)
    monkeypatch.setattr(settings, "rerank_candidates", 2)
    monkeypatch.setattr(settings, "rrf_k", 60)  # pin fusion: this test is about the hand-off
    monkeypatch.setattr(settings, "rrf_bm25_weight", 1.0)

    out = retriever.retrieve("q", "hybrid_rerank", top_n=1)
    assert [c["chunk_id"] for c in seen["candidates"]] == ["b", "a"]  # only top-2 fused
    assert seen["candidates"][0]["ranks"] == {"vector": 2, "bm25": 1, "rrf": 1}
    assert out[0]["chunk_id"] == "a" and out[0]["score"] == out[0]["rerank_score"]
