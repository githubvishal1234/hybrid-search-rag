import pytest

pytest.importorskip("ragas")  # the optional eval extra; CI's plain install skips this file
import generation_metrics as gm  # noqa: E402


class FakeMetric:
    def __init__(self, value):
        self.value, self.calls = value, []

    async def ascore(self, **inputs):
        self.calls.append(inputs)
        return type("R", (), {"value": self.value})()


@pytest.fixture
def scorer(tmp_path):
    s = gm.RagasScorer("ollama", "fake-judge", tmp_path / "cache.jsonl")  # no network at init
    s.metrics = {m: FakeMetric(0.8) for m in gm.METRICS}
    return s


def test_metrics_apply_only_where_they_make_sense(scorer):
    out = scorer.score("q?", "LTV is 75% [gold.pdf, p.3].", ["ctx"], "75%", refused=False)
    assert set(out.values()) == {0.8}
    assert scorer.metrics["faithfulness"].calls[0]["response"] == "LTV is 75%."  # no citation

    refused = scorer.score("q2?", "I couldn't find this.", ["ctx"], "75%", refused=True)
    assert refused["faithfulness"] is None and refused["context_recall"] == 0.8
    unanswerable = scorer.score("q3?", "Something.", ["ctx"], "", refused=False)
    assert unanswerable["context_precision"] is None and unanswerable["faithfulness"] == 0.8


def test_scores_are_cached_on_disk_and_failures_are_not(scorer, tmp_path):
    scorer.score("q?", "A.", ["ctx"], "A", refused=False)
    again = gm.RagasScorer("ollama", "fake-judge", tmp_path / "cache.jsonl")
    again.metrics = {m: FakeMetric(0.1) for m in gm.METRICS}
    assert again.score("q?", "A.", ["ctx"], "A", refused=False)["faithfulness"] == 0.8  # cached
    assert again.metrics["faithfulness"].calls == []

    class Broken(FakeMetric):
        async def ascore(self, **inputs):
            raise ValueError("judge returned junk")

    again.metrics["faithfulness"] = Broken(None)
    assert again.score("new?", "B.", ["ctx"], "B", refused=False)["faithfulness"] is None
    assert "judge returned junk" in again.errors[0]
    assert len((tmp_path / "cache.jsonl").read_text().splitlines()) == 7  # not the failure


def test_daily_quota_stops_further_judge_calls(scorer):
    class OutOfQuota(FakeMetric):
        async def ascore(self, **inputs):
            try:
                raise RuntimeError("429 GenerateRequestsPerDayPerProjectPerModel-FreeTier")
            except RuntimeError as inner:
                raise ValueError("instructor gave up") from inner  # wrapped, like instructor

    scorer.metrics["faithfulness"] = OutOfQuota(None)
    out = scorer.score("q?", "A.", ["ctx"], "A", refused=False)
    assert scorer.quota_exhausted and out["faithfulness"] is None
    assert out["answer_relevancy"] is None  # skipped: it would fail the same way
    assert scorer.metrics["answer_relevancy"].calls == [] and scorer.errors == []
