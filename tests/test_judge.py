import json

import judge
import pytest


@pytest.fixture
def fake_llm(tmp_path, monkeypatch):
    calls = []

    def complete(system, user, provider=None, model=None):
        calls.append(user)
        return calls_reply.pop(0)

    calls_reply = []
    monkeypatch.setattr(judge, "CACHE", tmp_path / "cache.jsonl")
    monkeypatch.setattr(judge.llm, "complete", complete)
    return calls, calls_reply


def reply(*supported: bool) -> str:
    claims = [{"claim": f"c{i}", "supported": s} for i, s in enumerate(supported)]
    return "Here you go:\n" + json.dumps({"claims": claims})


def test_score_is_share_of_supported_claims():
    assert judge.score([{"supported": True}, {"supported": False}]) == 0.5
    assert judge.score([]) is None and judge.score(None) is None


def test_parse_claims_rejects_malformed_output():
    assert judge.parse_claims("no json") is None
    assert judge.parse_claims('{"claims": "nope"}') is None
    assert judge.parse_claims('{"claims": [{"claim": "x", "supported": "yes"}]}') == []


def test_judge_strips_citations_and_caches(fake_llm):
    calls, replies = fake_llm
    replies.append(reply(True, False, True))
    answer = "The LTV cap is 75% [rbi_gold.pdf, p.3]. It applies to NBFCs [rbi_gold.pdf, p.4-5]."
    cache: dict = {}
    claims = judge.judge_answer(answer, ["LTV shall not exceed 75%."], cache)
    assert judge.score(claims) == pytest.approx(2 / 3)
    assert "rbi_gold.pdf" not in calls[0]  # citations removed before judging
    assert "[1] LTV shall not exceed 75%." in calls[0]

    assert judge.judge_answer(answer, ["LTV shall not exceed 75%."], cache) == claims
    assert len(calls) == 1  # second time served from cache
    assert len(judge.load_cache()) == 1  # and persisted to disk


def test_unparseable_judge_output_is_retried_once_and_not_cached(fake_llm):
    calls, replies = fake_llm
    replies.extend(["garbage", "still garbage"])
    assert judge.judge_answer("Some answer.", ["ctx"], {}) is None
    assert len(calls) == 2
    assert judge.load_cache() == {}
