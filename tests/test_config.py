from hybrid_rag.config import ROOT, Settings


def test_env_example_is_a_valid_config(monkeypatch):
    """Copying .env.example to .env must not break startup (e.g. an empty Literal setting)."""
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            monkeypatch.setenv(key.strip(), value.strip())
    Settings(_env_file=None)  # raises if any value doesn't fit its field
