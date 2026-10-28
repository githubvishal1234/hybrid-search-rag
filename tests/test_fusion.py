import pytest

from hybrid_rag.retrieval import retriever
from hybrid_rag.retrieval.fusion import rrf


def hits(*ids: str) -> list[dict]:
    return [{"chunk_id": i, "text": f"text {i}"} for i in ids]


def test_rrf_scores_are_sum_of_reciprocal_ranks():
    fused = rrf({"vector": hits("a", "b"), "bm25": hits("b", "c")}, k=60)
    scores = {h["chunk_id"]: h["score"] for h in fused}
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)
    assert [h["chunk_id"] for h in fused] == ["b", "a", "c"]


def test_agreement_beats_a_single_first_place():
    # "x" is first in one list only; "y" is 3rd and 4th. With k=60, agreement wins.
    fused = rrf({"vector": hits("x", "p", "y"), "bm25": hits("q", "r", "s", "y")}, k=60)
    assert fused[0]["chunk_id"] == "y"


def test_small_k_rewards_top_ranks_more():
    lists = {"vector": hits("x", "p", "y"), "bm25": hits("q", "r", "s", "y")}
    assert rrf(lists, k=1)[0]["chunk_id"] != "y"  # 1/2 for a 1st place outweighs 1/4 + 1/5


def test_rrf_records_ranks_and_keeps_chunk_fields():
    fused = rrf({"vector": hits("a", "b"), "bm25": hits("b")}, k=60)
    b = next(h for h in fused if h["chunk_id"] == "b")
    assert b["ranks"] == {"vector": 2, "bm25": 1}
    assert b["text"] == "text b"
    assert rrf({"vector": [], "bm25": []}, k=60) == []


def test_retrieve_modes(monkeypatch):
    monkeypatch.setattr(retriever, "dense_search", lambda q, k: hits("a", "b", "c"))
    monkeypatch.setattr(retriever, "bm25_search", lambda q, k: hits("c", "d"))

    assert [h["chunk_id"] for h in retriever.retrieve("q", "vector", 2)] == ["a", "b"]
    assert retriever.retrieve("q", "bm25", 5)[0]["ranks"] == {"bm25": 1}
    hybrid = retriever.retrieve("q", "hybrid", 2)
    assert hybrid[0]["chunk_id"] == "c"  # only chunk in both lists
    assert hybrid[0]["ranks"] == {"vector": 3, "bm25": 1}
    with pytest.raises(ValueError):
        retriever.retrieve("q", "keyword")


def test_weighted_rrf_lets_the_stronger_list_win_over_mediocre_agreement():
    # "meh" is 30th in both lists; "b1" is BM25's #1 and absent from the vector list.
    vector = hits(*[f"v{i}" for i in range(1, 30)], "meh")
    bm25 = hits("b1", *[f"x{i}" for i in range(2, 30)], "meh")
    lists = {"vector": vector, "bm25": bm25}
    equal = {h["chunk_id"]: h["score"] for h in rrf(lists, k=60)}
    assert equal["meh"] > equal["b1"]  # 2/90 beats 1/61: the failure seen in the v1 eval
    heavy = {h["chunk_id"]: h["score"] for h in rrf(lists, k=60, weights={"bm25": 3.0})}
    assert heavy["b1"] == pytest.approx(3 / 61)
    assert heavy["b1"] > heavy["meh"]  # 3/61 beats 1/90 + 3/90
