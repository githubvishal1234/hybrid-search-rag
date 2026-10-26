# One image, two processes: FastAPI on 8000 (internal) + Streamlit on 7860 (the port HF Spaces
# exposes). Indexes are built outside the image (`make ingest index`) and copied in, or Qdrant
# Cloud is used via QDRANT_URL; the container never parses PDFs.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    FASTEMBED_CACHE_PATH=/app/models \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HOME=/app

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
WORKDIR /app

# Dependencies first: this layer is cached until pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# Bake the embedding, reranker and tokenizer models into the image so startup is fast.
RUN uv run python -c "\
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
from tokenizers import Tokenizer; \
from hybrid_rag.config import settings; \
TextEmbedding(settings.embedding_model); \
TextCrossEncoder(settings.reranker_model); \
Tokenizer.from_pretrained(settings.embedding_model)"

COPY ui ./ui
COPY scripts/start.sh ./scripts/start.sh
# Prebuilt indexes: chunks.jsonl, bm25.pkl and the embedded Qdrant store (see .dockerignore).
COPY data/processed ./data/processed

# HF Spaces runs containers as uid 1000; make everything it writes to owned by that user.
RUN useradd -m -u 1000 app && chown -R app /app
USER app

EXPOSE 7860 8000
CMD ["sh", "scripts/start.sh"]
