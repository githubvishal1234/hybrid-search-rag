# Functional Requirements Document (FRD)

**Product:** Hybrid Search RAG over Indian Finance Regulations
**Version:** 1.0 (as built)
**Date:** 2026-10-28
**Related:** [BRD](BRD.md) (business requirements), [Runbook](RUNBOOK.md) (operations),
[README](../README.md) (results and design decisions)

Each requirement states what the system must do, how it is implemented, and how it is verified.
"Verified by" names the automated test (`tests/`), evaluation report (`eval/results/`) or manual
check that demonstrates it.

---

## 1. Purpose and system overview

The system answers natural-language questions about Indian financial regulations using only a
fixed set of PDF documents, and returns the answer with page-level citations, or a refusal.

```
PDFs -> Parse -> Clean -> Chunk -> chunks.jsonl -> BM25 index (pickle)
                                               -> Embeddings -> Qdrant
Question -> BM25 top 30 + Vector top 30 -> weighted RRF -> (cross-encoder rerank) -> top 5
         -> LLM with numbered sources -> citation mapping + validation -> JSON response / UI
```

Components: ingestion (`src/hybrid_rag/ingestion/`), indexing (`indexing/`), retrieval
(`retrieval/`), generation (`generation/`), REST API (`api/`), web UI (`ui/app.py`), evaluation
(`eval/`), operations scripts (`scripts/`).

## 2. Actors

| Actor | Description |
|---|---|
| End user | Asks questions in the web UI |
| API client | Another program calling the REST API |
| Administrator | Adds documents, rebuilds indexes, runs evaluations, deploys |
| LLM provider | External or local model: Ollama, Gemini, OpenAI or Anthropic |
| Reranker provider | Local cross-encoder or Cohere Rerank API |

---

## 3. Functional requirements

Priority: M = must, S = should, C = could.

### 3.1 Document ingestion (FR-ING)

| ID | Requirement | Pri | Implementation | Verified by |
|---|---|---|---|---|
| FR-ING-1 | Maintain a document registry with `file_name, title, source_url, publisher, date` | M | `data/sources.csv` | `test_sources.py::test_sources_csv_is_well_formed` |
| FR-ING-2 | Download registered PDFs, skipping files already present and rejecting non-PDF responses (checks `%PDF` header) | M | `scripts/download_data.py` | Manual run; 27/30 downloaded (RBI blocks scripts) |
| FR-ING-3 | Extract text per page with original page numbers preserved | M | PyMuPDF, `ingestion/parse.py` | Manual inspection of 5 difficult PDFs; `pages.jsonl` |
| FR-ING-4 | Convert tables to markdown so numbers stay with their column headers | M | PyMuPDF `find_tables()` | `test_clean.py::test_clean_table_strips_markdown_artifacts_and_hindi`, `::test_dedupe_cells_collapses_merged_cell_copies_but_keeps_numbers` |
| FR-ING-5 | Drop non-English lines (share of non-ASCII letters > `MAX_NON_ENGLISH_RATIO`, default 0.3); skip pages left empty | M | `ingestion/clean.py` | `test_clean.py::test_non_english_ratio`, `::test_clean_block_unwraps_lines_and_drops_hindi` |
| FR-ING-6 | Remove repeated headers/footers, page numbers, hyphenation breaks and chart-label noise | M | `ingestion/clean.py` | `test_clean.py::test_clean_document_removes_running_headers_and_page_numbers`, `::test_clean_block_keeps_hyphenated_compounds`, `::test_clean_block_drops_chart_label_soup` |
| FR-ING-7 | OCR pages with almost no text but an image, when Tesseract is installed; otherwise log a warning | C | `parse.py`, `OCR_MIN_CHARS`, `OCR_DPI` | Manual; Tesseract not installed in v1.0 (~20 image-only pages have no text) |
| FR-ING-8 | Ingestion is idempotent: unchanged files (same SHA-256) are not re-parsed | M | Parse cache keyed by file hash + OCR flag | Manual re-run |
| FR-ING-9 | Split text into chunks of at most `CHUNK_SIZE` (450) embedding-model tokens with `CHUNK_OVERLAP` (64); sentences are the unit; headings close a chunk; chunks may span pages | M | `ingestion/chunk.py` | `test_chunk.py` (7 tests: size, coverage, overlap, headings, page spans, oversized sentences) |
| FR-ING-10 | Split large tables into row groups with the header row repeated in each chunk | M | `chunk.py` | `test_chunk.py::test_big_table_is_split_with_header_repeated_once_per_chunk` |
| FR-ING-11 | Store metadata per chunk (section 5.1) | M | `chunks.jsonl` | Schema inspection |

### 3.2 Indexing (FR-IDX)

| ID | Requirement | Pri | Implementation | Verified by |
|---|---|---|---|---|
| FR-IDX-1 | Embed each chunk as "title / section heading + text" with `EMBEDDING_MODEL` (bge-small-en-v1.5, 384 dims); queries get the BGE instruction prefix | M | fastembed (ONNX), `indexing/embed.py` | Eval: removing the title prefix lowered Recall@5 (`eval/results/experiments/`) |
| FR-IDX-2 | Store vectors in Qdrant with cosine distance; embedded mode when `QDRANT_URL` is empty, server otherwise | M | `indexing/vector_store.py` | Fresh-clone test (runbook 1.2) |
| FR-IDX-3 | Index sync is exact: upsert current chunks, delete points not in the current chunk set; rebuild if the embedding size changes | M | `vector_store.sync()` | `test_vector_store.py::test_sync_removes_points_from_an_older_chunking`, `::test_sync_starts_over_when_the_embedding_size_changes` |
| FR-IDX-4 | Build a BM25 index whose tokenizer keeps identifiers whole and as parts (`10(23C)`, `114B`, `26AS`), normalises Indian digit grouping (`50,00,000`) and "per cent", and stems plain words only | M | `indexing/bm25_index.py` (rank_bm25 + PyStemmer) | `test_bm25.py` (3 tests) |
| FR-IDX-5 | Persist indexes so the application starts without re-processing PDFs | M | `data/processed/` (committed) | Fresh-clone test: retrieval works with no PDFs |

### 3.3 Retrieval (FR-RET)

| ID | Requirement | Pri | Implementation | Verified by |
|---|---|---|---|---|
| FR-RET-1 | Support four modes: `vector`, `bm25`, `hybrid`, `hybrid_rerank` | M | `retrieval/retriever.py` | `test_fusion.py::test_retrieve_modes` |
| FR-RET-2 | Dense retrieval returns top `DENSE_TOP_K` (30) chunks by cosine similarity | M | `retrieval/dense.py` | Eval reports |
| FR-RET-3 | Keyword retrieval returns top `BM25_TOP_K` (30) chunks, excluding chunks with no matching term | M | `retrieval/sparse.py` | `test_bm25.py::test_search_ranks_exact_term_first_and_skips_non_matches` |
| FR-RET-4 | Hybrid mode fuses the two lists with weighted Reciprocal Rank Fusion: `score = sum(w_i / (k + rank_i))`, `k = RRF_K` (10), BM25 weight `RRF_BM25_WEIGHT` (2.0); records each chunk's rank per list | M | `retrieval/fusion.py` (own implementation) | `test_fusion.py` (6 tests incl. weighted case) |
| FR-RET-5 | `hybrid_rerank` re-scores the top `RERANK_CANDIDATES` (30) fused chunks with a cross-encoder and orders by its score | M | `retrieval/rerank.py` | `test_rerank.py::test_rerank_orders_by_cross_encoder_score`, `::test_hybrid_rerank_reranks_top_fused_candidates` |
| FR-RET-6 | The reranker is switchable by configuration: local (`RERANKER_MODEL`) or Cohere (`RERANKER_PROVIDER=cohere`) | S | `rerank.py` | `test_rerank.py::test_cohere_scores_are_mapped_back_by_index` |
| FR-RET-7 | Pass the top `CONTEXT_TOP_N` (5) chunks to generation | M | `generation/answer.py` | `test_api.py::test_query_returns_the_contract` |

### 3.4 Answer generation (FR-GEN)

| ID | Requirement | Pri | Implementation | Verified by |
|---|---|---|---|---|
| FR-GEN-1 | Present context as numbered sources `[1]..[n]` with title, page span and section | M | `generation/prompts.py` | `test_answer.py::test_sources_are_numbered_with_page_spans` |
| FR-GEN-2 | Instruct the LLM to use only the sources, cite a source number after each sentence, quote figures exactly, and reply with the fixed refusal sentence when unsupported | M | `prompts.SYSTEM` | Generation eval (refusal accuracy) |
| FR-GEN-3 | Map cited numbers to `[file_name, p.X]` / `[file_name, p.X-Y]` in the answer | M | `answer.py` | `test_answer.py::test_valid_citations_become_file_and_page` |
| FR-GEN-4 | Drop cited numbers that were not in the context and report them in `invalid_citations` | M | `answer.py` | `test_answer.py::test_hallucinated_citation_is_dropped_and_reported` |
| FR-GEN-5 | Detect refusals (`refused: true`) and flag answers with no citation (`uncited: true`) | M | `answer.py` | API contract test; UI warnings |
| FR-GEN-6 | Support Ollama, OpenAI, Anthropic and Gemini through one interface, selected by `LLM_PROVIDER` / `LLM_MODEL` | M | `generation/llm.py` | Ollama and Gemini used in practice; `test_config.py` |
| FR-GEN-7 | Return per-stage latency (retrieve, generate, total) in milliseconds | M | `answer.py` | API contract test |

### 3.5 REST API (FR-API)

| ID | Requirement | Pri | Implementation | Verified by |
|---|---|---|---|---|
| FR-API-1 | `POST /query` accepts `{question, mode}` and returns the response in section 4.2 | M | `api/main.py`, `api/schemas.py` | `test_api.py::test_query_returns_the_contract` |
| FR-API-2 | Validate input: question 3-1000 characters, mode one of the four; invalid input returns 422 | M | pydantic `QueryRequest` | `test_api.py::test_query_rejects_bad_input` |
| FR-API-3 | Rate-limit `/query` per client IP to `RATE_LIMIT_PER_MINUTE` (10) over a sliding minute; excess returns 429 | M | `check_rate_limit()` | `test_api.py::test_rate_limit` |
| FR-API-4 | LLM unreachable returns 503; LLM error response returns 502 (never an unhandled 500) | M | exception mapping | `test_api.py::test_llm_down_is_a_503_not_a_500` |
| FR-API-5 | `GET /health` reports status, chunks indexed, configured LLM and reranker, ingest state | M | `health()` | Manual; UI sidebar |
| FR-API-6 | `POST /ingest` rebuilds pages, chunks and indexes as a background task; requires `X-Admin-Token` equal to `ADMIN_TOKEN` (constant-time compare); disabled (403) when unset; 401 on wrong token; 409 if already running; `/query` returns 503 while it runs | S | `ingest()` | `test_api.py::test_ingest_needs_admin_token` |
| FR-API-7 | `GET /ingest` returns the last ingest status | S | `ingest_status()` | Manual |
| FR-API-8 | Log one JSON line per query: id, mode, per-stage latency, citation counts, refusal | M | `log.info(json.dumps(...))` | Manual (API console) |
| FR-API-9 | Warm models at startup so the first user doesn't pay the load time; start even if no index exists | S | `lifespan()` | Manual |
| FR-API-10 | Serve interactive API documentation at `/docs` | S | FastAPI OpenAPI | Manual |

### 3.6 Web UI (FR-UI)

| ID | Requirement | Pri | Implementation | Verified by |
|---|---|---|---|---|
| FR-UI-1 | Question box, Ask button (disabled under 3 characters), and four example questions | M | `ui/app.py` | Live demo |
| FR-UI-2 | Retrieval mode selector with a one-line description of each mode | M | sidebar radio | Live demo |
| FR-UI-3 | Show the answer; show refusals distinctly (info box); warn on dropped and missing citations | M | `ui/app.py` | Live demo (cited answer and refusal both observed) |
| FR-UI-4 | Show retrieval, generation and total latency | M | metrics row | Live demo |
| FR-UI-5 | List sources as expandable document + page entries with a text snippet | M | expanders | Live demo |
| FR-UI-6 | Retrieval debug panel: per chunk vector rank, BM25 rank, RRF rank, rerank score | S | dataframe | Live demo |
| FR-UI-7 | Show index size and active LLM in the sidebar | S | `health()` | Live demo ("6,117 chunks indexed") |
| FR-UI-8 | Run either against the API (`API_URL` set) or in-process (`API_URL` empty) with the same behaviour; show errors as messages, not stack traces | M | `ask()`, `health()` | Local test of both modes; live deployment uses in-process |

### 3.7 Evaluation (FR-EVAL)

| ID | Requirement | Pri | Implementation | Verified by |
|---|---|---|---|---|
| FR-EVAL-1 | Golden set in JSONL with id, question, type, expected answer, relevant (doc, pages) list, origin, review flags (section 5.2) | M | `eval/golden_set.jsonl` (74 questions) | `review_golden_set.py --check` |
| FR-EVAL-2 | Five question types: exact_term, paraphrase, numeric, multi_hop, unanswerable | M | golden set | `--check` prints the mix |
| FR-EVAL-3 | Draft questions with an LLM, auto-reject obvious defects, then review keep/edit/drop with reasons; add hand-written questions | M | `scripts/build_golden_set.py`, `review_golden_set.py` | `test_golden_set.py` (8 tests) |
| FR-EVAL-4 | Relevance is (document, page) based so chunking changes don't invalidate labels | M | `eval/metrics.py` | `test_metrics.py::test_chunk_spanning_the_page_counts_and_other_docs_dont` |
| FR-EVAL-5 | Compute Recall@5, Recall@10, MRR, nDCG@10 and latency p50/p95 per mode, and Recall@5 per question type; exclude unanswerable questions from retrieval metrics | M | `metrics.py`, `run_eval.py` | `test_metrics.py` (recall, MRR, nDCG, percentile tests) |
| FR-EVAL-6 | Stratified, stable dev/test split (`--split`) so tuning uses dev only | M | `run_eval.split()` | `test_metrics.py::test_split_is_stratified_disjoint_and_stable` |
| FR-EVAL-7 | Sweep RRF k x fusion depth x BM25 weight from cached lists (`--sweep`) | S | `run_eval.py` | Dev sweep report |
| FR-EVAL-8 | Generation metrics: citation accuracy, refusal accuracy, false refusals, uncited rate, latency; resumable after interruption | M | `run_eval.py --generate` | `test_metrics.py::test_generation_resumes_after_a_crash`; report `20261027-1925-v1-generation.md` |
| FR-EVAL-9 | Claim-level faithfulness judge with a separately configurable judge model, cached results, retry on rate limits, one report per judge | S | `eval/judge.py` | `test_judge.py` (4 tests) |
| FR-EVAL-10 | Write each run as markdown + CSV under `eval/results/` with timestamp and label | M | `run_eval.py` | Reports in repo |

### 3.8 Configuration, packaging and deployment (FR-OPS)

| ID | Requirement | Pri | Implementation | Verified by |
|---|---|---|---|---|
| FR-OPS-1 | All tunable settings live in one settings class and can be overridden by environment variables / `.env` | M | `config.py` (pydantic-settings) | `test_config.py::test_env_example_is_a_valid_config` |
| FR-OPS-2 | Single Docker image runs API (:8000) and UI (:7860), with models and prebuilt indexes baked in; container exits if either process dies | M | `Dockerfile`, `scripts/start.sh` | CI Docker build |
| FR-OPS-3 | docker-compose for local use, with an optional Qdrant server profile | S | `docker-compose.yml` | Manual |
| FR-OPS-4 | CI runs lint, tests and a Docker build on every push | M | GitHub Actions | Actions tab |
| FR-OPS-5 | Public deployment on Streamlit Community Cloud from `main`, redeploying on push | M | in-process UI + committed indexes | Live URL |
| FR-OPS-6 | Hugging Face Space deploy script that never uploads secrets | C | `scripts/deploy_space.py` | `test_deploy_space.py` (2 tests) |

---

## 4. Interface specification

### 4.1 `POST /query` request

```json
{ "question": "What does section 148A provide?", "mode": "hybrid" }
```
`question`: string, 3-1000 characters. `mode`: `vector` | `bm25` | `hybrid` (default) | `hybrid_rerank`.

### 4.2 `POST /query` response (200)

Shape is exact (`api/schemas.py`); values are illustrative.

```json
{
  "answer": "Section 148A ... [budget_2026_27_finance_bill.pdf, p.202]",
  "refused": false,
  "citations": [
    {"doc": "budget_2026_27_finance_bill.pdf", "page": "p.202",
     "chunk_id": "budget_2026_27_finance_bill:0257", "snippet": "The Finance Act, 2021 had inserted..."}
  ],
  "invalid_citations": [],
  "uncited": false,
  "retrieval_debug": {
    "mode": "hybrid",
    "chunks": [{"rank": 1, "chunk_id": "...", "vector_rank": 3, "bm25_rank": 1,
                "rrf_rank": 1, "rerank_score": null, "score": 0.2587}]
  },
  "latency_ms": {"retrieve": 1104, "generate": 4800, "total": 5904}
}
```

### 4.3 Status codes

| Code | When |
|---|---|
| 200 | Answer or refusal returned |
| 401 | `/ingest` with missing or wrong admin token |
| 403 | `/ingest` while ingestion is disabled (`ADMIN_TOKEN` unset) |
| 409 | `/ingest` while one is already running |
| 422 | Invalid request body |
| 429 | Rate limit exceeded |
| 502 | LLM provider returned an error |
| 503 | LLM unreachable, or indexes being rebuilt |

### 4.4 Other endpoints
`GET /health` -> `{status, chunks_indexed, llm, reranker, ingest_running}`;
`POST /ingest` (202) and `GET /ingest` -> `{running, last_result}`; `GET /docs` (OpenAPI UI).

### 4.5 Command-line interface
`python -m hybrid_rag.cli "<question>" [--mode M] [--retrieve-only] [--json]`.

---

## 5. Data specification

### 5.1 Chunk record (`data/processed/chunks.jsonl`, one JSON object per line)

| Field | Type | Description |
|---|---|---|
| chunk_id | string | `<doc_id>:<4-digit index>`, stable across runs |
| doc_id, file_name | string | Document identifier and PDF file name |
| title, publisher, date, source_url | string | From `sources.csv` |
| text | string | Cleaned chunk text (tables as markdown) |
| page_start, page_end | int | Original PDF page numbers covered |
| section_heading | string | Nearest heading above the chunk |
| n_tokens | int | Length in embedding-model tokens |
| chunk_index | int | Position within the document |

### 5.2 Golden set record (`eval/golden_set.jsonl`)

`{"id", "question", "type", "expected_answer", "relevant": [{"doc", "pages": [int]}], "origin": "llm_draft"|"hand", "reviewed", "edited"}`.
`relevant` is empty for unanswerable questions; each entry is one requirement (multi-hop
questions have two).

### 5.3 Corpus (v1.0)

27 indexed PDFs, 6,117 chunks: 14 SEBI master circulars, 8 Union Budget 2026-27 documents,
4 IRDAI regulations, 1 PRS summary. 3 RBI documents are registered but not indexed.

---

## 6. Non-functional requirements

| ID | Category | Requirement | Current status |
|---|---|---|---|
| NFR-1 | Retrieval quality | Recall@5 above the vector-only baseline (0.625) on the golden set | 0.867 (hybrid_rerank), 0.836 (hybrid) |
| NFR-2 | Latency | Retrieval p95 under 100 ms for vector, BM25 and hybrid | 34 / 47 / 79 ms |
| NFR-3 | Latency | Interactive answer under 15 s on the public demo | ~6-11 s observed (Gemini, jina reranker, 10 candidates) |
| NFR-4 | Grounding | Refuse at least 80% of unanswerable questions | 0.90 (hybrid) |
| NFR-5 | Citation correctness | At least 75% of cited pages correct | 0.828 (hybrid) |
| NFR-6 | Reliability | API never returns an unhandled 500 for LLM failures | 502/503 mapping, tested |
| NFR-7 | Security | No secrets in the repository; keys only in `.env` or platform secrets; admin endpoint off by default; deploy script refuses secret-looking variables | `.gitignore`, `test_deploy_space.py`, constant-time token compare |
| NFR-8 | Abuse protection | Per-IP rate limit on `/query` | 10/minute (per process) |
| NFR-9 | Cost | Zero running cost for the demo | Free hosting + free LLM tier |
| NFR-10 | Portability | Runs on Windows/macOS/Linux with uv, or anywhere with Docker | Fresh-clone test on Windows |
| NFR-11 | Maintainability | Settings in one place; lint and tests pass before commit; CI on every push | 65 tests, ruff clean |
| NFR-12 | Reproducibility | Locked dependencies; evaluation reports stored with settings and timestamp | `uv.lock`, `eval/results/` |
| NFR-13 | Observability | Per-stage latency in every response and one JSON log line per query | Implemented |

Known limitations: the rate limit is per process (not shared across replicas); the free demo LLM
allows 20 requests/day; local CPU generation with llama3.1 takes ~1-2 minutes; image-only pages
have no text without Tesseract.

---

## 7. Configuration reference

| Setting | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` / `LLM_MODEL` | ollama / llama3.1 | Answering model |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` | empty | Provider keys |
| `JUDGE_PROVIDER` / `JUDGE_MODEL` | answering model | Faithfulness judge |
| `EMBEDDING_MODEL` | BAAI/bge-small-en-v1.5 | Dense embeddings |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 450 / 64 | Chunking (tokens) |
| `DENSE_TOP_K` / `BM25_TOP_K` | 30 / 30 | Candidates per retriever |
| `RRF_K` / `RRF_BM25_WEIGHT` | 10 / 2.0 | Fusion (chosen on dev split) |
| `RERANKER_PROVIDER` / `RERANKER_MODEL` | local / BAAI/bge-reranker-base | Reranking |
| `RERANK_CANDIDATES` / `CONTEXT_TOP_N` | 30 / 5 | Rerank depth / chunks to the LLM |
| `QDRANT_URL` / `QDRANT_API_KEY` | empty (embedded) | Vector store location |
| `API_URL` | http://localhost:8000 | UI -> API; empty = in-process |
| `RATE_LIMIT_PER_MINUTE` | 10 | Per-IP limit on `/query` |
| `ADMIN_TOKEN` | unset (ingest disabled) | Enables `POST /ingest` |
| `MAX_NON_ENGLISH_RATIO` | 0.3 | Hindi line filter |

Full list: `src/hybrid_rag/config.py`. Setup and examples: [Runbook](RUNBOOK.md) section 2.

---

## 8. Acceptance criteria (release 1.0)

| # | Criterion | Result |
|---|---|---|
| AC-1 | `ruff check .` and `pytest` pass (65 tests) | Pass |
| AC-2 | A fresh clone answers a question with no PDFs and no ingestion | Pass (verified 2026-10-28) |
| AC-3 | Evaluation report shows hybrid_rerank Recall@5 > vector-only | Pass (0.867 vs 0.625) |
| AC-4 | An unanswerable question on the live demo returns the refusal sentence | Pass ("GST rate on cryptocurrency trading") |
| AC-5 | A cited answer on the live demo shows document + page sources | Pass (TCS on LRS remittances) |
| AC-6 | Invalid citation numbers are removed from answers | Pass (unit test) |
| AC-7 | No secrets in git history for this release | Pass (`.env` ignored; keys in platform secrets) |

---

## 9. Traceability: business -> functional requirements

| BRD requirement | FRD requirements |
|---|---|
| BR-1 Answer from documents only | FR-ING-1..11, FR-IDX-1..5, FR-RET-1..7, FR-GEN-2 |
| BR-2 Document + page citations | FR-ING-3, FR-GEN-1, FR-GEN-3, FR-UI-5 |
| BR-3 Refuse when unsupported | FR-GEN-2, FR-GEN-5, FR-UI-3, NFR-4 |
| BR-4 Exact terms and paraphrases | FR-IDX-1, FR-IDX-4, FR-RET-4, FR-RET-5 |
| BR-5 Measured against a baseline | FR-EVAL-1..10, NFR-1 |
| BR-6 Public web access | FR-UI-1..8, FR-OPS-5 |
| BR-7 API integration | FR-API-1..10 |
| BR-8 Add documents without code changes | FR-ING-1, FR-ING-2, FR-ING-8, FR-IDX-3, FR-API-6 |
| BR-9 Zero running cost | FR-GEN-6, FR-OPS-5, NFR-9 |
| BR-10 Transparency | FR-UI-6, FR-API-8, FR-GEN-7 |
| BR-11 Acceptable response time | FR-RET-6, NFR-2, NFR-3 |

## 10. Open items

| Item | Owner | Reference |
|---|---|---|
| Faithfulness scored by an independent judge | Owner | Runbook 7.3 |
| Human spot-check of the golden set | Owner | Runbook 9 |
| Demo video and README screenshot | Owner | Runbook 10 |
| RBI documents and OCR for image-only pages | Future | BRD section 12 |
