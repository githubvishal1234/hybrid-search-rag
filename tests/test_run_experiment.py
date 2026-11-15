import json
from pathlib import Path

import pytest
import run_experiment as rx

from hybrid_rag.pipeline import Context, RagConfig, RagResult, Usage


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
    }  # fmt: skip


def test_experiment_file_typos_fail_before_running(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nhypothesis: y\nconfig:\n  rrf_kk: 10\n", encoding="utf-8")
    with pytest.raises(ValueError):  # pydantic ValidationError: unknown RagConfig field
        rx.load_experiment(bad)
    bad.write_text("name: x\nconfig: {}\n", encoding="utf-8")
    with pytest.raises(SystemExit):  # no hypothesis: say what you expect before you run it
        rx.load_experiment(bad)


def test_run_writes_config_rows_summary_and_report(tmp_path, monkeypatch):
    asked = []

    def fake_run_rag(question, config, generate):
        asked.append((config.context_top_n, generate))
        contexts = [ctx(1, "x.pdf", 1), ctx(2, "a.pdf", 5)]
        return RagResult(
            question=question, config=config, answer="", refused=False, citations=[],
            invalid_citations=[], uncited=False, contexts=contexts,
            latency_ms={"bm25": 4, "retrieve": 5, "total": 5}, tokens=Usage(),
        )  # fmt: skip

    monkeypatch.setattr(rx, "run_rag", fake_run_rag)
    monkeypatch.setattr(rx, "RESULTS", tmp_path)
    exp = {"name": "t", "hypothesis": "h", "config": {"mode": "bm25"}}
    questions = [
        q("q1", "numeric", [{"doc": "a.pdf", "pages": [5]}]),
        q("q2", "unanswerable", []),
    ]
    out = rx.run_one(exp, RagConfig(mode="bm25"), questions, "dev")

    assert asked == [(10, False), (10, False)]  # top 10 for the metrics, never the LLM
    rows = [json.loads(line) for line in (out / "per_question.jsonl").open(encoding="utf-8")]
    assert rows[0]["metrics"]["hit@1"] == 0.0 and rows[0]["metrics"]["mrr"] == 0.5
    assert rows[1]["metrics"] is None  # unanswerable: nothing to retrieve
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall"]["n"] == 1 and summary["by_type"]["numeric"]["hit@5"] == 1.0
    assert summary["latency_ms"]["retrieve"] == {"p50": 5, "p95": 5}
    meta = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert meta["config"]["context_top_n"] == 5  # the config as written, not the eval override
    assert "| 0.000 | 1.000 |" in (out / "report.md").read_text(encoding="utf-8")
    ladder = rx.ladder([(meta, summary), (meta, summary)], "dev")
    assert ladder.count("| t |") == 6  # one row per run in each of the 3 tables


def test_committed_experiments_are_valid():
    paths = sorted(Path(rx.ROOT, "experiments").glob("*.yaml"))
    assert len(paths) >= 5
    names = [rx.load_experiment(p)[0]["name"] for p in paths]
    assert len(set(names)) == len(names)
