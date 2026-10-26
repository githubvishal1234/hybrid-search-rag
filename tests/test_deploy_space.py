import pytest
from deploy_space import keep, parse_vars


def test_keep_ships_runtime_files_only():
    assert keep("uv.lock")  # the Docker build needs it (uv sync --frozen)
    assert keep("src/hybrid_rag/api/main.py") and keep("data/processed/bm25.pkl")
    assert not keep("data/processed/qdrant/.lock")
    assert not keep("tests/test_api.py") and not keep("eval/run_eval.py")


def test_parse_vars_refuses_secrets():
    assert parse_vars(["LLM_PROVIDER=gemini"]) == [{"key": "LLM_PROVIDER", "value": "gemini"}]
    with pytest.raises(SystemExit):
        parse_vars(["GEMINI_API_KEY=abc"])
    with pytest.raises(SystemExit):
        parse_vars(["no-equals-sign"])
