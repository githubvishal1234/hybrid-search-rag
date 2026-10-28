import math

import pytest
from metrics import (
    citation_correct,
    first_hit_rank,
    ndcg_at,
    percentile,
    recall_at,
    reciprocal_rank,
)


def c(doc: str, start: int, end: int | None = None) -> dict:
    return {"file_name": doc, "page_start": start, "page_end": end or start}


SINGLE = [{"doc": "a.pdf", "pages": [5]}]
TWO_HOP = [{"doc": "a.pdf", "pages": [5]}, {"doc": "b.pdf", "pages": [9, 10]}]


def test_chunk_spanning_the_page_counts_and_other_docs_dont():
    assert first_hit_rank([c("b.pdf", 5), c("a.pdf", 4, 6)], SINGLE) == 2
    assert first_hit_rank([c("a.pdf", 6), c("b.pdf", 5)], SINGLE) is None


def test_recall_counts_each_requirement():
    ranking = [c("x.pdf", 1), c("a.pdf", 5), c("y.pdf", 1), c("y.pdf", 2), c("y.pdf", 3)]
    assert recall_at(ranking, TWO_HOP, 5) == 0.5  # found 1 of the 2 passages
    assert recall_at(ranking + [c("b.pdf", 10)], TWO_HOP, 10) == 1.0
    assert recall_at(ranking, SINGLE, 1) == 0.0


def test_mrr():
    assert reciprocal_rank([c("a.pdf", 5)], SINGLE) == 1.0
    assert reciprocal_rank([c("x.pdf", 1), c("x.pdf", 2), c("a.pdf", 5)], SINGLE) == pytest.approx(
        1 / 3
    )
    assert reciprocal_rank([c("x.pdf", 1)], SINGLE) == 0.0


def test_ndcg_rewards_order_and_credits_each_requirement_once():
    assert ndcg_at([c("a.pdf", 5), c("b.pdf", 9)], TWO_HOP, 10) == pytest.approx(1.0)
    late = ndcg_at([c("x.pdf", 1), c("a.pdf", 5), c("b.pdf", 9)], TWO_HOP, 10)
    ideal = 1 + 1 / math.log2(3)
    assert late == pytest.approx((1 / math.log2(3) + 1 / math.log2(4)) / ideal)
    dup = ndcg_at([c("a.pdf", 5), c("a.pdf", 5)], TWO_HOP, 10)  # same passage twice: no credit
    assert dup == pytest.approx(1 / ideal)


def test_citation_correct_parses_page_labels():
    assert citation_correct({"doc": "b.pdf", "page": "p.8-9"}, TWO_HOP)
    assert citation_correct({"doc": "a.pdf", "page": "p.5"}, TWO_HOP)
    assert not citation_correct({"doc": "a.pdf", "page": "p.6-7"}, TWO_HOP)


def test_percentile_nearest_rank():
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile(list(range(1, 101)), 95) == 95
    assert math.isnan(percentile([], 50))


def test_split_is_stratified_disjoint_and_stable():
    from run_eval import split

    qs = [{"id": f"q{i}", "type": t} for i in range(10) for t in ("numeric", "paraphrase")]
    dev, test = split(qs, "dev"), split(qs, "test")
    assert {q["id"] for q in dev}.isdisjoint({q["id"] for q in test})
    assert len(dev) + len(test) == len(qs)
    assert sum(q["type"] == "numeric" for q in dev) == 5  # each type split evenly
    assert split(qs, "dev") == dev and split(qs, "all") == qs


def test_generation_resumes_after_a_crash(tmp_path, monkeypatch):
    from run_eval import load_progress, run_generation

    import hybrid_rag.generation.answer as answer_mod

    asked = []

    def fake_answer(question, mode):
        asked.append((mode, question))
        if question == "boom?":
            raise RuntimeError("LLM down")
        chunk = {"chunk_id": "a:0001"}
        return {
            "refused": False,
            "uncited": False,
            "citations": [],
            "invalid_citations": [],
            "latency_ms": {"total": 5},
            "answer": f"answer to {question}",
            "retrieval_debug": {"chunks": [chunk]},
        }

    monkeypatch.setattr(answer_mod, "answer", fake_answer)
    qs = [
        {"id": i, "question": f"{i}?", "type": "numeric", "expected_answer": "", "relevant": []}
        for i in ("q1", "q2")
    ] + [
        {"id": "q3", "question": "boom?", "type": "numeric", "expected_answer": "", "relevant": []}
    ]
    progress = tmp_path / "run-generation.progress.jsonl"
    # simulate a crash after q1 was answered in "bm25"
    progress.write_text(
        '{"mode": "bm25", "id": "q1", "type": "numeric", "answer": "saved earlier"}\n',
        encoding="utf-8",
    )

    rows = run_generation(qs, ["bm25"], progress)

    assert ("bm25", "q1?") not in asked  # already done: skipped
    assert [r["id"] for r in rows] == ["q1", "q2", "q3"]
    assert rows[0]["answer"] == "saved earlier"
    assert "error" in rows[2]  # the failed call is reported...
    saved = load_progress(progress)
    assert set(saved) == {("bm25", "q1"), ("bm25", "q2")}  # ...but not saved, so it retries
    assert saved[("bm25", "q2")]["context_ids"] == "a:0001"
