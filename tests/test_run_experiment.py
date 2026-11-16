import json
from pathlib import Path

import pytest
import run_experiment as rx

from hybrid_rag.pipeline import Citation, Context, RagConfig, RagResult, Usage


def ctx(rank: int, doc: str, page: int) -> Context:
    return Context(
        rank=rank, chunk_id=f"{doc}:{rank}", file_name=doc, title="T", page_start=page,
        page_end=page, section_heading="", text="...", score=0.1, bm25_rank=rank, vector_rank=None,
        rrf_rank=None, rerank_score=None,
    )  # fmt: skip


def q(qid: str, qtype: str, relevant: list[dict]) -> dict:
    return {
        "id": qid, "question": "q?", "type": qtype, "difficulty": "easy", "split": "dev",
        "answerable": qtype != "unanswerable", "relevant": relevant,
        "expected_answer": "" if qtype == "unanswerable" else "5%",
    }  # fmt: skip


def test_experiment_file_typos_fail_before_running(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nhypothesis: y\nconfig:\n  rrf_kk: 10\n", encoding="utf-8")
    with pytest.raises(ValueError):  # pydantic ValidationError: unknown RagConfig field
        rx.load_experiment(bad)
    bad.write_text("name: x\nconfig: {}\n", encoding="utf-8")
    with pytest.raises(SystemExit):  # no hypothesis: say what you expect before you run it
        rx.load_experiment(bad)


EXP = {"name": "t", "hypothesis": "h", "config": {"mode": "bm25"}}
QUESTIONS = [
    q("q1", "numeric", [{"doc": "a.pdf", "pages": [5]}]),
    q("q2", "unanswerable", []),
]


@pytest.fixture
def fake_rag(tmp_path, monkeypatch):
    """q1 is answered citing a.pdf p.5 (correct); q2 (unanswerable) is refused."""
    asked = []

    def fake_run_rag(question, config, generate, retrieve_n):
        asked.append((question, retrieve_n, config.context_top_n, generate))
        contexts = [ctx(1, "x.pdf", 1), ctx(2, "a.pdf", 5)]
        answering = generate and question == "question 1?"
        cites = [Citation(doc="a.pdf", page="p.5", chunk_id="a.pdf:2", snippet="...")]
        return RagResult(
            question=question, config=config, answer="It is 5% [a.pdf, p.5]." if answering else "",
            refused=generate and not answering, citations=cites if answering else [],
            invalid_citations=[], uncited=False, contexts=contexts,
            latency_ms={"bm25": 4, "retrieve": 5, "generate": 7, "total": 12},
            tokens=Usage(input_tokens=100, output_tokens=10),
        )  # fmt: skip

    monkeypatch.setattr(rx, "run_rag", fake_run_rag)
    monkeypatch.setattr(rx, "RESULTS", tmp_path)
    for i, query in enumerate(QUESTIONS):
        query["question"] = f"question {i + 1}?"
    return asked


def test_run_writes_config_rows_summary_and_report(fake_rag):
    out = rx.run_one(EXP, RagConfig(mode="bm25"), QUESTIONS, "dev")

    assert [a[1:] for a in fake_rag] == [(10, 5, False)] * 2  # rank 10 deep; no LLM
    rows = [json.loads(line) for line in (out / "per_question.jsonl").open(encoding="utf-8")]
    assert rows[0]["metrics"]["hit@1"] == 0.0 and rows[0]["metrics"]["mrr"] == 0.5
    assert rows[1]["metrics"] is None  # unanswerable: nothing to retrieve
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall"]["n"] == 1 and summary["by_type"]["numeric"]["hit@5"] == 1.0
    assert summary["latency_ms"]["retrieve"] == {"p50": 5, "p95": 5}
    assert "generate" not in summary["latency_ms"] and summary["generation"] == {}
    meta = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert meta["config"]["context_top_n"] == 5
    assert "| 0.000 | 1.000 |" in (out / "report.md").read_text(encoding="utf-8")
    ladder = rx.ladder([(meta, summary), (meta, summary)], "dev")
    assert ladder.count("| t |") == 6  # one row per run in each of the 3 tables


class FakeScorer:
    judge = "fake:judge"

    def __init__(self, quota_after: int | None = None):
        self.meter = type("M", (), {"calls": 3, "input_tokens": 900, "output_tokens": 30})()
        self.seen, self.errors, self.quota_exhausted = [], [], False
        self.quota_after = quota_after  # simulate the daily quota running out after N questions

    def score(self, question, answer, contexts, reference, refused):
        self.seen.append((question, contexts, refused))
        if self.quota_after is not None and len(self.seen) >= self.quota_after:
            self.quota_exhausted = True
        return {"faithfulness": None if refused else 0.5, "answer_relevancy": None,
                "context_precision": None, "context_recall": None}  # fmt: skip


def test_generation_scores_citations_refusals_and_judge(fake_rag):
    scorer = FakeScorer()
    cfg = RagConfig(mode="bm25", context_top_n=1)
    out = rx.run_one(EXP, cfg, QUESTIONS, "dev", generate=True, scorer=scorer)

    assert scorer.seen[0][1] == ["..."]  # the judge sees only what the LLM saw (top 1)
    g = json.loads((out / "summary.json").read_text(encoding="utf-8"))["generation"]
    assert g["refusal_accuracy"] == {"value": 1.0, "n": 1}  # unanswerable q2 refused
    assert g["false_refusal_rate"] == {"value": 0.0, "n": 1}
    assert g["citation_strict"]["value"] == 1.0  # a.pdf p.5 is the golden page
    assert g["faithfulness"] == {"value": 0.5, "n": 1}  # refused q2 isn't judged
    assert g["tokens_per_answer"]["input_tokens"] == 100
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "judge fake:judge" in report and "3 calls" in report


def test_resume_skips_done_questions_and_refuses_other_settings(fake_rag):
    cfg = RagConfig(mode="bm25")
    out = rx.run_one(EXP, cfg, QUESTIONS[:1], "dev", generate=True)
    fake_rag.clear()
    rx.run_one(EXP, cfg, QUESTIONS, "dev", generate=True, resume=out)
    assert [a[0] for a in fake_rag] == ["question 2?"]  # q1 was already done
    rows = (out / "per_question.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    with pytest.raises(SystemExit):
        rx.run_one(EXP, RagConfig(mode="vector"), QUESTIONS, "dev", generate=True, resume=out)


def test_committed_experiments_are_valid():
    paths = sorted(Path(rx.ROOT, "experiments").glob("*.yaml"))
    assert len(paths) >= 5
    names = [rx.load_experiment(p)[0]["name"] for p in paths]
    assert len(set(names)) == len(names)


def test_compare_marks_noise_and_lists_flips(fake_rag):
    import compare

    a = rx.run_one(EXP, RagConfig(mode="bm25"), QUESTIONS, "dev")
    b = rx.run_one({**EXP, "name": "u"}, RagConfig(mode="vector"), QUESTIONS, "dev")
    text = compare.compare(a, b)
    assert "| hit@5 | 1.000 | 1.000 | +0.000 ~ |" in text  # same result: within noise
    assert "found by B only (0): none" in text
    meta = json.loads((b / "config.json").read_text(encoding="utf-8"))
    (b / "config.json").write_text(json.dumps({**meta, "split": "test"}), encoding="utf-8")
    with pytest.raises(SystemExit):  # dev vs test numbers aren't comparable
        compare.compare(a, b)


@pytest.fixture
def saved_answers(fake_rag, tmp_path, monkeypatch):
    """A --generate run with no judge, plus the golden set and chunk texts rejudge reads."""
    golden = tmp_path / "golden.jsonl"
    extra = {"origin": "hand", "reviewed": True, "edited": False}
    golden.write_text("".join(json.dumps(qq | extra) + "\n" for qq in QUESTIONS), "utf-8")
    monkeypatch.setattr(rx, "GOLDEN", golden)
    chunks = [{"chunk_id": "x.pdf:1", "text": "chunk one"}, {"chunk_id": "a.pdf:2", "text": "two"}]
    (tmp_path / "chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunks), "utf-8")
    monkeypatch.setattr(rx.settings, "processed_dir", tmp_path)
    return rx.run_one(EXP, RagConfig(mode="bm25", context_top_n=1), QUESTIONS, "dev", generate=True)


def test_rejudge_scores_saved_answers_with_the_texts_the_llm_saw(saved_answers):
    scorer = FakeScorer()
    assert rx.rejudge(saved_answers, scorer) is True
    assert [s[1] for s in scorer.seen] == [["chunk one"], ["chunk one"]]  # top 1, by chunk id
    summary = json.loads((saved_answers / "summary.json").read_text(encoding="utf-8"))
    assert summary["generation"]["faithfulness"] == {"value": 0.5, "n": 1}
    meta = json.loads((saved_answers / "config.json").read_text(encoding="utf-8"))
    assert meta["judge"] == "fake:judge"
    other = FakeScorer()
    other.judge = "another:judge"
    with pytest.raises(SystemExit):  # mixing judges in one ladder makes numbers incomparable
        rx.rejudge(saved_answers, other)


def test_rejudge_stops_at_the_daily_quota_and_keeps_the_file_whole(saved_answers):
    scorer = FakeScorer(quota_after=1)
    assert rx.rejudge(saved_answers, scorer) is False  # not finished: run again tomorrow
    assert len(scorer.seen) == 1
    rows = (saved_answers / "per_question.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2 and "ragas" not in json.loads(rows[1])["gen"]
