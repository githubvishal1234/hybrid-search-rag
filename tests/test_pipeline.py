import pytest
from pydantic import ValidationError

from hybrid_rag import pipeline
from hybrid_rag.api.schemas import QueryResponse
from hybrid_rag.config import settings
from hybrid_rag.generation import prompts


def chunk(i: int, page: int, ranks: dict, rerank_score: float | None = None) -> dict:
    return {
        "chunk_id": f"doc{i}:000{i}",
        "file_name": f"doc{i}.pdf",
        "title": f"Doc {i}",
        "page_start": page,
        "page_end": page,
        "section_heading": "",
        "text": f"text of chunk {i}",
        "score": 0.5,
        "ranks": ranks,
        **({"rerank_score": rerank_score} if rerank_score is not None else {}),
    }


@pytest.fixture
def fakes(monkeypatch):
    seen = {}

    def fake_retrieve(question, mode, top_n, *, timings, **knobs):
        seen.update(mode=mode, top_n=top_n, **knobs)
        timings.update(dense=3, bm25=2, fusion=1)
        return [chunk(1, 3, {"vector": 2, "bm25": 1}), chunk(2, 9, {"bm25": 4}, rerank_score=1.5)]

    def fake_llm(system, user, provider, model):
        seen.update(system=system, user=user, provider=provider, model=model)
        return seen.get("reply", "LTV is 75% [1]. Tenor 12 months [7]."), {
            "input_tokens": 300,
            "output_tokens": 12,
        }

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(pipeline.llm, "complete_with_usage", fake_llm)
    return seen


def test_config_defaults_come_from_settings_and_typos_fail(monkeypatch):
    monkeypatch.setattr(settings, "rrf_k", 42)
    assert pipeline.RagConfig().rrf_k == 42  # read at creation, so .env/test overrides apply
    with pytest.raises(ValidationError):
        pipeline.RagConfig(rrf_kk=10)  # misspelt knob in an experiment file
    with pytest.raises(ValidationError):
        pipeline.RagConfig(mode="keyword")


def test_config_reaches_retrieval_and_llm(fakes):
    cfg = pipeline.RagConfig(
        mode="hybrid_rerank", context_top_n=2, rrf_k=7, llm_provider="gemini", llm_model="m"
    )
    pipeline.run_rag("q", cfg)
    assert fakes["mode"] == "hybrid_rerank" and fakes["top_n"] == 2 and fakes["rrf_k"] == 7
    assert fakes["provider"] == "gemini" and fakes["model"] == "m"
    assert fakes["system"] == prompts.SYSTEMS["v1"]


def test_result_has_contexts_citations_latency_and_tokens(fakes):
    r = pipeline.run_rag("What is the LTV cap?")
    assert [c.chunk_id for c in r.contexts] == ["doc1:0001", "doc2:0002"]
    first, second = r.contexts
    assert (first.rank, first.vector_rank, first.bm25_rank, first.rerank_score) == (1, 2, 1, None)
    assert (second.vector_rank, second.bm25_rank, second.rerank_score) == (None, 4, 1.5)
    assert second.text == "text of chunk 2"  # contexts carry text, for faithfulness judges
    assert r.answer == "LTV is 75% [doc1.pdf, p.3]. Tenor 12 months."
    assert [c.chunk_id for c in r.citations] == ["doc1:0001"]
    assert r.invalid_citations == [7]  # cited a source it was never given
    assert not r.refused and not r.uncited
    assert {"dense", "bm25", "fusion", "retrieve", "generate", "total"} <= set(r.latency_ms)
    assert (r.tokens.input_tokens, r.tokens.output_tokens) == (300, 12)


def test_refusal_is_flagged_and_not_counted_as_uncited(fakes):
    fakes["reply"] = prompts.REFUSAL
    r = pipeline.run_rag("What is the GST rate on crypto?")
    assert r.refused and not r.uncited and r.citations == []


def test_retrieval_only_skips_the_llm(fakes):
    r = pipeline.run_rag("q", generate=False)
    assert "system" not in fakes  # LLM never called
    assert r.answer == "" and not r.uncited and r.tokens.input_tokens is None
    assert len(r.contexts) == 2


def test_to_response_matches_the_api_contract(fakes):
    QueryResponse.model_validate(pipeline.run_rag("q").to_response())


def test_eval_can_rank_deeper_than_the_llm_sees(fakes):
    r = pipeline.run_rag("q", pipeline.RagConfig(context_top_n=1), retrieve_n=2)
    assert fakes["top_n"] == 2 and len(r.contexts) == 2  # both ranked, for Hit@k
    assert "text of chunk 1" in fakes["user"] and "text of chunk 2" not in fakes["user"]
