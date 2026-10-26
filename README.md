# Hybrid Search RAG over Indian finance PDFs

BM25 + dense retrieval fused with RRF, cross-encoder reranking, and cited answers over RBI / SEBI / Union Budget documents, with an eval that measures each step.

Work in progress. See [CLAUDE.md](CLAUDE.md) for the full plan.

## Quickstart (so far)

```bash
uv sync                                          # install deps into .venv
cp .env.example .env                             # defaults: embedded Qdrant + local Ollama
uv run python scripts/download_data.py           # fetch PDFs listed in data/sources.csv
uv run python -m hybrid_rag.ingestion.pipeline   # PDFs -> pages.jsonl -> chunks.jsonl
uv run python -m hybrid_rag.indexing.build       # embed chunks into Qdrant (~7 min on CPU)
ollama pull llama3.1                             # or set LLM_PROVIDER / LLM_MODEL in .env
uv run python -m hybrid_rag.cli "What is the debt-to-GDP ratio estimated for 2026-27?"   # --mode vector|bm25|hybrid|hybrid_rerank
uv run pytest -q
```

Qdrant runs embedded by default. To use a server instead: `docker compose up -d qdrant` and set
`QDRANT_URL=http://localhost:6333` in `.env`, then rebuild the index.

## Golden set (evaluation questions)

```bash
uv run python scripts/build_golden_set.py            # LLM drafts ~70 questions -> eval/golden_set.draft.jsonl
uv run python scripts/review_golden_set.py           # keep / edit / drop each draft (resumable)
uv run python scripts/review_golden_set.py --add     # write your own questions from the PDFs (~30%)
uv run python scripts/review_golden_set.py --check   # validate eval/golden_set.jsonl, show the type mix
```

Only reviewed questions land in `eval/golden_set.jsonl`. Labels are `(doc, pages)`, not chunk ids.

## Evaluation

```bash
uv run python eval/run_eval.py --sweep      # retrieval metrics for all 4 modes + RRF k/depth sweep
uv run python eval/run_eval.py --generate   # + LLM answers: citation acc., refusal acc., latency (slow)
```

Reports land in `eval/results/` as Markdown plus per-question CSVs (use those to study failures).
Metrics are page-level: Recall@5/10, MRR, nDCG@10 (see `eval/metrics.py`).

## API and UI

```bash
uv run uvicorn hybrid_rag.api.main:app --port 8000   # API, docs at http://localhost:8000/docs
uv run streamlit run ui/app.py                       # UI at http://localhost:8501 (needs the API)
```

```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json"   -d '{"question": "What is the TCS rate on LRS remittances for education?", "mode": "hybrid_rerank"}'
```

Response: `answer`, `refused`, `citations[{doc, page, chunk_id, snippet}]`, `invalid_citations`,
`uncited`, `retrieval_debug{mode, chunks[{vector_rank, bm25_rank, rrf_rank, rerank_score}]}`,
`latency_ms{retrieve, generate, total}`. `/health` reports the index; `POST /ingest` (header
`X-Admin-Token`, enabled only when `ADMIN_TOKEN` is set) rebuilds everything from `data/raw`.
`/query` is rate-limited per IP (`RATE_LIMIT_PER_MINUTE`, default 10).

Note: embedded Qdrant allows one process at a time, so stop the API before running the CLI or
eval (or point everything at a Qdrant server with `QDRANT_URL`).

## Docker

```bash
make ingest index                 # build indexes on the host first; the image copies data/processed
docker compose up --build app     # UI http://localhost:7860, API http://localhost:8000/docs
```

One image runs both processes (`scripts/start.sh`); models are downloaded at build time.
Inside Docker the default Ollama provider reaches the host via `host.docker.internal`; for a
public deployment set an API provider (`LLM_PROVIDER`, key) as a secret instead.
