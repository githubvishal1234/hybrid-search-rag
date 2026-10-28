"""All tunable settings live here. Values come from env vars / .env, falling back to these defaults.

Usage: `from hybrid_rag.config import settings` then `settings.rrf_k`.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Paths
    data_dir: Path = ROOT / "data"
    sources_csv: Path = ROOT / "data" / "sources.csv"
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"

    # Qdrant. Empty URL = embedded mode: same API, stored in qdrant_path, no server needed.
    # Docker: http://localhost:6333. Qdrant Cloud: cluster URL + API key.
    qdrant_url: str = ""
    qdrant_path: Path = ROOT / "data" / "processed" / "qdrant"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "finance_docs"

    # Ingestion
    # Tokens, counted with the embedding model's tokenizer. bge models see at most 512 tokens;
    # 450 leaves room to prefix "title / section" to each chunk at embedding time.
    chunk_size: int = 450
    chunk_overlap: int = 64
    max_non_english_ratio: float = 0.3  # lines with more non-ASCII letters (Hindi) are dropped
    ocr_min_chars: int = 50  # pages with fewer alphanumeric chars (and an image) get OCR'd
    ocr_dpi: int = 300
    # Edge blocks repeated on >= this share of pages are headers/footers. Low because long docs
    # switch headers per section (SEBI MF circular: main body / Annexures / Formats).
    header_footer_min_share: float = 0.1
    min_letter_ratio: float = 0.25  # blocks with fewer letters are chart-label soup, dropped

    # Retrieval
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # BGE's recommended instruction for short query -> long passage retrieval (queries only).
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    dense_top_k: int = 30
    bm25_top_k: int = 30
    # Chosen on the golden set's dev half (eval/results/*-v1-dev-sweep.md): k=60 with equal
    # weights scored below BM25 alone; k=10 with BM25 weighted 2x matched BM25's Recall@5 with a
    # better MRR. Reported on the untouched test half.
    rrf_k: int = 10
    rrf_bm25_weight: float = 2.0  # weighted RRF: >1 trusts BM25's ranks more than the vector list
    context_top_n: int = 5  # chunks given to the LLM (after reranking, when enabled)

    # Reranker. bge-reranker-base: best quality in Day 10 probes but ~15 s / 30 chunks on this
    # laptop CPU; "jinaai/jina-reranker-v1-turbo-en" is ~4x faster. Let the eval decide.
    reranker_provider: Literal["local", "cohere"] = "local"
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_candidates: int = 30  # fused chunks the cross-encoder re-scores
    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-v3.5"

    # LLM
    # Default runs locally with no key (ollama pull llama3.1). Switch provider + model in .env.
    llm_provider: Literal["openai", "anthropic", "gemini", "ollama"] = "ollama"
    llm_model: str = "llama3.1"
    llm_max_tokens: int = 800
    # Faithfulness judge for the eval (eval/judge.py). Empty = same as the answering LLM, which
    # works but is biased (a model grading itself); prefer a stronger API model here.
    judge_provider: Literal["openai", "anthropic", "gemini", "ollama"] | None = None
    judge_model: str | None = None

    # API
    api_url: str = "http://localhost:8000"  # where the Streamlit UI finds the API
    rate_limit_per_minute: int = 10  # per client IP, on /query
    admin_token: str | None = None  # required for POST /ingest; unset = ingest disabled
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"


settings = Settings()
