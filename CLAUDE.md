# Hybrid Search RAG (BM25 + Vector) — Project Brief

> How to use this file: create an empty repo, save this file as `CLAUDE.md` in the root, open Claude Code there, and paste the **Starter prompt** from the bottom of this file. Work one milestone at a time. Ask Claude Code to explain anything you don't understand, because you will need to defend every choice in interviews.

---

## 1. Goal

Build a production-style question-answering system over real, messy PDFs (Indian finance and policy documents) that:

1. Combines **keyword search (BM25)** and **semantic search (dense vectors)**, merged with **Reciprocal Rank Fusion (RRF)**.
2. **Reranks** the fused candidates with a cross-encoder.
3. Generates **answers with citations** (document name + page number) and says "I don't know" when the context doesn't support an answer.
4. **Proves it works with numbers**: a golden evaluation set and before/after metrics for vector-only vs hybrid vs hybrid + reranker.
5. Ships as a **Dockerized FastAPI service** with a simple UI, deployed to a public URL.

The point for the job switch: most candidates can show "chat with PDF". Few can show *measured* retrieval improvements, hybrid search, reranking and evals. That is the story this project tells.

## 2. Dataset

Use 30 to 60 public PDFs, mixed in quality on purpose (tables, multi-column layouts, scanned pages, footnotes). Good sources:

- **RBI**: Master Directions and circulars (KYC, digital lending, NBFC regulations) — rbi.org.in
- **SEBI**: circulars and regulations (LODR, mutual fund regulations) — sebi.gov.in
- **Union Budget**: Budget speech, Finance Bill, key-to-budget documents — indiabudget.gov.in
- **Income Tax**: CBDT circulars, ITR filing instructions — incometaxindia.gov.in
- **Insurance**: IRDAI regulations, plus 5 to 10 sample health insurance policy wordings
- Optional: annual reports of 3 to 5 listed companies (Infosys, HDFC Bank, etc.) for table-heavy content

Corpus as built (30 listed, 27 auto-downloadable): 3 RBI (bot-challenged, save by hand), 15 SEBI master circulars, 8 Union Budget 2026-27 documents (speech, at a glance, memorandum, Finance Bill, key to budget, receipt budget, annual financial statement, FRBM statements), 4 IRDAI regulations (bilingual gazette PDFs), 1 PRS summary of the Income-tax Bill 2025 (incometaxindia.gov.in returns 403 to scripts). Known parsing limits: ~20 image-only pages (ICDR offer tables, Depositories forms) need Tesseract; a few Kruti Dev (ASCII-font Hindi) gazette lines survive because a detector for them wrongly flagged 1.2% of lines in English-only circulars (measured), and the repeated ones are already removed as headers.

Keep a `data/sources.csv` with `file_name, title, source_url, publisher, date`. Don't commit large PDFs; add a `scripts/download_data.py` that fetches them from the URLs.

Why this data works: legal and financial text is full of exact terms (section numbers, "Rule 114B", "LTV ratio", "Form 26AS") where BM25 beats embeddings, and paraphrased questions where embeddings beat BM25. That makes hybrid search measurably better, which is the whole point.

## 3. Architecture

```
PDFs ──► Parse (PyMuPDF / unstructured, OCR fallback) ──► Clean ──► Chunk (with metadata)
                                                                       │
                                     ┌─────────────────────────────────┤
                                     ▼                                 ▼
                              BM25 index                      Embeddings ─► Qdrant
                         (rank_bm25 / Qdrant sparse)          (bge-small/base-en-v1.5)
                                     │                                 │
Query ──► (optional rewrite) ──► BM25 top-k (30)              Vector top-k (30)
                                     └──────────► RRF fusion ◄─────────┘
                                                     │  top 30
                                                     ▼
                                     Cross-encoder reranker (bge-reranker-base or Cohere Rerank)
                                                     │  top 5
                                                     ▼
                                  LLM answer with inline citations [doc, p.X]
                                                     │
                                      Response: answer + sources + scores + latency
```

### 3.1 Ingestion and chunking
- Parse with **PyMuPDF** (fast) and keep page numbers. Fall back to OCR (`pytesseract` or `unstructured` hi_res) for pages with almost no extractable text.
- Skip non-English text (RBI/SEBI/Budget PDFs are often bilingual, sometimes on the same page): drop each line whose share of non-ASCII letters is above a config threshold (catches Unicode Devanagari and legacy-font Hindi); strip Devanagari from table cells. Pages left empty are skipped; original page numbers are kept.
- Clean: remove repeated headers/footers, page numbers, hyphenation breaks, collapse whitespace.
- Extract tables as markdown where possible (`pdfplumber`), so numbers survive.
- Chunk: start with **recursive splitting, ~512 tokens, 64 token overlap**, respecting headings. Store metadata per chunk: `doc_id, file_name, title, page_start, page_end, section_heading, chunk_index, source_url`.
- Make ingestion **idempotent** (hash each file; skip unchanged files).
- Experiment later: compare 256 / 350 / 450 token chunk sizes in your eval. That comparison is a great README table. (Not 1024: bge models truncate at 512 tokens, so most of a 1024-token chunk would never be embedded. Testing longer chunks needs a long-context embedder like bge-m3.)
- Decided: chunk_size defaults to 450 BGE tokens (room for a title/section prefix inside the 512 window). Units are sentences and table row-groups (header row repeated); headings close a chunk; chunks may span pages (`page_start`/`page_end`).

### 3.2 Retrieval
- **Dense**: `BAAI/bge-small-en-v1.5` locally (or `bge-base` if you have a GPU), cosine similarity in Qdrant.
- **Sparse/keyword**: BM25. Two options:
  - Simple: `rank_bm25` in memory, persisted with pickle.
  - Better: Qdrant **sparse vectors** (BM25 or SPLADE via `fastembed`) so both indexes live in one DB.
- **Fusion**: Reciprocal Rank Fusion, `score(d) = Σ 1 / (k + rank_i(d))` with `k = 60`. Implement it yourself in ~15 lines (interviewers love asking about this) rather than only calling a library.
- Optional: metadata filters (publisher, year) passed through the API.

### 3.3 Reranking
- Cross-encoder `BAAI/bge-reranker-base` via `sentence-transformers` (free, local), with a config switch to **Cohere Rerank** API.
- Rerank top 30 fused results, keep top 5 for the LLM.

### 3.4 Answer generation
- Any LLM via one interface (OpenAI, Claude, Gemini, or local via Ollama). Keep the provider in config.
- Prompt rules: answer only from context; cite every claim as `[file_name, p.X]`; if the context is insufficient, reply "I couldn't find this in the documents."
- Return structured JSON: `answer, citations[{doc, page, chunk_id, snippet}], retrieval_debug{bm25_ranks, vector_ranks, rerank_scores}, latency_ms`.
- Post-check: verify every cited chunk id was actually in the context (drop or flag hallucinated citations).

## 4. Stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | type hints, `pydantic` models |
| API | FastAPI + Uvicorn | Built: `/query`, `/health`, `/ingest` (admin token, background task), `/docs`. One JSON log line per query with per-stage latency. In-memory per-IP rate limit. LLM down -> 503, LLM error -> 502 |
| Orchestration | None: PyMuPDF + provider SDKs directly | Decided: no LlamaIndex/LangChain. Retrieval, RRF, prompts are our own code |
| Vector DB | **Qdrant** | Decided: embedded mode (`QDRANT_URL` empty) for local dev, Docker or Qdrant Cloud via `QDRANT_URL`. Same client code |
| Keyword | rank_bm25 (in memory, pickled) | Decided. Custom tokenizer keeps `26AS`, `114B`, `10(23C)` intact (whole form + parts), normalises `50,00,000` and "per cent", Snowball-stems plain words only (PyStemmer). Qdrant sparse = later upgrade |
| Packaging (deps) | uv + pyproject.toml | `uv sync`, lockfile committed |
| Embeddings | bge-small-en-v1.5 / bge-base-en-v1.5 | Decided: `fastembed` (ONNX, no PyTorch). BGE query instruction added in config; passages embedded as "title / section + text" |
| Reranker | bge-reranker-base or Cohere Rerank | config switch (`RERANKER_PROVIDER`); local via fastembed `TextCrossEncoder`, Cohere v2 `/rerank` via stdlib HTTP. Measured: ~15 s per 30 candidates on the dev laptop CPU (8 threads); jina-reranker-v1-turbo-en ~4 s but weaker on exact-term probes |
| LLM | Ollama llama3.1 (default), OpenAI / Anthropic / Gemini | Sources numbered [1]..[n] in the prompt; answer.py maps numbers to `[file, p.X]` and drops numbers not in context |
| Eval | **RAGAS** + your own retrieval metrics | |
| UI | Streamlit (or Gradio) | shows answer, citations, and retrieval debug panel |
| Packaging | Docker + docker-compose | Built: single image (API :8000 + Streamlit :7860 via scripts/start.sh), models baked in, prebuilt indexes copied from data/processed. Compose: `app` (+ optional `qdrant` under profile `server`) |
| Quality | pytest, ruff, GitHub Actions CI | |
| Observability (bonus) | Langfuse or simple structured JSON logs | latency per stage, token cost |

## 5. Folder structure

```
hybrid-rag/
├── CLAUDE.md
├── README.md
├── pyproject.toml            # or requirements.txt
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── Makefile                  # make ingest / serve / eval / test
├── data/
│   ├── sources.csv
│   ├── raw/                  # PDFs (gitignored)
│   └── processed/            # chunks.jsonl (gitignored)
├── scripts/
│   ├── download_data.py
│   └── build_golden_set.py   # LLM-assisted draft of eval questions
├── src/hybrid_rag/
│   ├── config.py             # pydantic-settings
│   ├── ingestion/
│   │   ├── parse.py          # PyMuPDF + OCR fallback + tables
│   │   ├── clean.py
│   │   ├── chunk.py
│   │   └── pipeline.py
│   ├── indexing/
│   │   ├── embed.py
│   │   ├── vector_store.py   # Qdrant client wrapper
│   │   └── bm25_index.py
│   ├── retrieval/
│   │   ├── dense.py
│   │   ├── sparse.py
│   │   ├── fusion.py         # RRF
│   │   ├── rerank.py
│   │   └── retriever.py      # mode = vector | bm25 | hybrid | hybrid_rerank
│   ├── generation/
│   │   ├── prompts.py
│   │   ├── llm.py
│   │   └── answer.py         # citations + validation
│   └── api/
│       ├── main.py
│       └── schemas.py
├── ui/app.py                 # Streamlit
├── eval/
│   ├── golden_set.jsonl
│   ├── run_eval.py
│   ├── metrics.py            # recall@k, MRR, nDCG
│   └── results/              # csv + markdown reports per run
└── tests/
    ├── test_chunk.py
    ├── test_fusion.py
    └── test_api.py
```

## 6. Milestones (3 weeks, ~2 hours on weekdays, more on weekends)

### Week 1 — Data, ingestion, baseline
- **Day 1–2**: Repo scaffold, config, docker-compose with Qdrant, `download_data.py`, collect 30+ PDFs.
- **Day 3–4**: Parsing + cleaning + OCR fallback + table extraction. Inspect output for 5 ugly PDFs by hand and fix issues.
- **Day 5**: Chunking with metadata, write `chunks.jsonl`, unit tests.
- **Day 6–7**: Embeddings into Qdrant, **vector-only baseline** retrieval + simple answer generation with citations via a CLI.
- ✅ Done when: you can ask a question in the terminal and get a cited answer.

### Week 2 — Hybrid, rerank, evaluation
- **Day 8**: BM25 index + sparse retriever.
- **Day 9**: RRF fusion (own implementation + tests), `retriever.py` with 4 modes.
- **Day 10**: Cross-encoder reranker + Cohere switch.
- **Day 11–12**: Build the golden set (see section 7). Review every question by hand.
- **Day 13–14**: `run_eval.py` comparing all 4 modes; save results table. Tune top-k, chunk size, RRF k.
- ✅ Done when: you have a results table showing hybrid + rerank beats vector-only.

### Week 3 — API, UI, deploy, polish
- **Day 15–16**: FastAPI endpoints, pydantic schemas, error handling, structured logs with per-stage latency.
- **Day 17**: Streamlit UI with citations and a "retrieval debug" expander (BM25 rank, vector rank, rerank score per chunk).
- **Day 18**: Dockerfile (single image: api + ui for HF Spaces) + compose for local dev; GitHub Actions running ruff + pytest.
- **Day 19**: Deploy (section 9).
- **Day 20–21**: README, architecture diagram, 2-minute demo video (Loom), LinkedIn post.
- ✅ Done when: a recruiter can open a public link, ask a question, and see cited answers.

## 7. Evaluation plan

### Golden set (50–100 questions)
Store as `eval/golden_set.jsonl`:
```json
{"id": "q001", "question": "What is the maximum LTV ratio for gold loans by NBFCs?", "type": "exact_term", "expected_answer": "...", "relevant_doc": "rbi_gold_loan_circular.pdf", "relevant_pages": [3]}
```
Mix of question types (tag each one):
- **exact_term** (~25%): section numbers, form names, acronyms — BM25 should shine.
- **paraphrase** (~25%): same meaning, different words — vectors should shine.
- **numeric/table** (~20%): figures from tables.
- **multi-hop** (~15%): needs 2 chunks or 2 documents.
- **unanswerable** (~15%): not in the corpus — system must refuse.

Draft questions with an LLM (`build_golden_set.py` samples chunks and asks for Q/A pairs), then **review and fix every one by hand**. Say this in interviews.

Decided rules:
- Schema as built: `{"id", "question", "type", "expected_answer", "relevant": [{"doc", "pages"}], "origin": "llm_draft"|"hand", "reviewed", "edited"}`. `relevant` is a list so multi_hop can span two documents; unanswerable has `relevant: []`.
- Workflow: `build_golden_set.py` drafts (sqrt-weighted doc sampling, per-type prompts, auto-rejects numeric answers without digits / exact_term questions without an identifier / TOC and blank-form chunks) -> `review_golden_set.py` keep/edit/drop (drop reasons logged) -> `--add` for hand-written -> `--check` validates pages against the parsed corpus.
- Relevance labels are `(doc, page)` pairs, never chunk ids, so chunk-size experiments stay comparable.
- ~30% of questions written by hand from the PDFs without looking at chunks (limits leakage toward our own chunker).
- Unanswerable questions are near-misses: on-topic for the corpus, but not answered in it.
- Tune with retrieval metrics only (free, seconds). Run RAGAS only on the 4 final modes, with judge outputs cached.
- Hypothesis from Day 9 probes (5 queries, not evidence yet): with k=60 and 30-deep lists, chunks ranked mediocre in BOTH lists (e.g. 26 + 15) outscore a chunk ranked #1 in only one, so hybrid lost to the better single retriever on 2/5 probes. k=10 or fusing only the top 10 of each list fixed both. Sweep `rrf_k` x fusion depth on the golden set before picking defaults.
- Day 10 probes: hybrid_rerank 15/25 hits vs bm25 13, hybrid 10, vector 9; it undid both RRF failures (circular query: BM25 ranks 1-5 -> RRF 10-17 -> reranked back to 1-5). Seen once: llama3.1 added an unsupported claim ("rescinded") with a valid citation; citation checks can't catch that, faithfulness eval must.

### Metrics
- Built (`eval/metrics.py`, `eval/run_eval.py`): each `relevant` entry is one requirement; a chunk satisfies it if same doc and page spans overlap. Recall@k = share of requirements met in top k; MRR on first hit; nDCG@10 credits each requirement once. Unanswerable questions are excluded from retrieval metrics. `--sweep` re-fuses cached top-30 lists for rrf_k {5,10,20,60} x depth {10,20,30}. `--generate` adds citation acc., refusal acc., false refusals, uncited rate, latency. Faithfulness: decided to implement our own claim-level LLM judge (eval/judge.py) instead of RAGAS (avoids the LangChain dependency tree); generation CSVs store `context_ids` so answers can be judged later; JUDGE_PROVIDER/JUDGE_MODEL pick a stronger judge than the answering model; outputs cached in eval/results/judge_cache.jsonl.
- Day 13 smoke run on 17 unreviewed drafts (debug only): BM25 scored 1.0 on "paraphrase" drafts, i.e. llama3.1's paraphrases reuse passage wording. Review must enforce that paraphrase questions avoid the passage's distinctive words; hand-written questions matter here.
- Golden set v1 (2026-10-26): 74 questions, 64 answerable. 46 llama3.1 drafts reviewed against source pages (43 edited, 24 dropped; every "multi_hop" draft was single-page, most "unanswerable" drafts were answerable) + 28 written from page text. Reviewer for all 74 is Claude (`reviewer: "claude"`); a human spot-check is still owed before claiming "hand-verified".
- v1 retrieval results (eval/results/20261026-1708-v1-fast.md): Recall@5 vector 0.625, BM25 0.859, hybrid RRF k=60 0.789; BM25 best on every type incl. paraphrase. Sweep best k=5 depth 20 -> 0.820, still < BM25. Diagnosis: dense top hits for Budget questions are number tables; title prefix makes same-doc chunks look alike. Full set: hybrid_rerank 0.867 (MRR 0.708, paraphrase 0.889, numeric 0.864).
- Decided (dev/test protocol): golden set split 50/50 stratified by type (`run_eval.py --split`). Dev sweep chose rrf_k=10, rrf_bm25_weight=2.0 (now defaults). Test half: BM25 0.875 / MRR 0.633; hybrid equal k=60 0.766 / 0.610; hybrid weighted 0.828 / 0.647; hybrid_rerank 0.875 / 0.677. Generation eval done (eval/results/20261027-1925-v1-generation.md, llama3.1, all 74): citation acc. vector 0.774 / BM25 0.793 / hybrid 0.828 / hybrid_rerank 0.754; refusal acc. 1.0 / 0.9 / 0.9 / 0.9; false refusals 0.172 / 0.094 / 0.094 / 0.047; p95 82 / 65 / 73 / 93 s. First run lost 136 calls to MemoryError: Ollama with OLLAMA_CONTEXT_LENGTH=8192 grew to 14.6 GB commit; 4096 is enough (prompts ~2.2k tokens). Generation is resumable (progress file per label).
- Final (2026-10-28), current defaults, full set (eval/results/20261028-0100-v1-final-retrieval.md): Recall@5 / MRR vector 0.625/0.499, BM25 0.859/0.665, hybrid weighted 0.836/0.683, hybrid_rerank 0.867/0.707; p95 retrieval 34 ms / 47 ms / 79 ms / 14.3 s. Faithfulness (self-graded llama3.1, eval/results/20261027-1925-v1-generation-faithfulness.md): 0.917 / 0.915 / 0.920 / 0.916, 219/233 scored, 102/836 claims unsupported; flat across modes, so needs a stronger judge to be meaningful.
- Rejected on dev (eval/results/experiments/): excluding number-only chunks (letter ratio <0.3/<0.4) from the vector index changed nothing; embedding without the document title hurt (vector 0.688->0.625, hybrid 0.844->0.828, MRR 0.720->0.689). Production embedding stays "title / section + text".
- **Retrieval** (own code): Recall@5, Recall@10, MRR, nDCG@10 — "is a relevant page in the results?"
- **Generation** (RAGAS): faithfulness, answer relevancy, context precision, context recall.
- **Citation accuracy**: % of answers whose citations point to a correct page.
- **Refusal accuracy**: % of unanswerable questions correctly refused.
- **Latency** p50/p95 per stage and **cost per query**.

### Before/after table (the money shot for your README)
| Mode | Recall@5 | MRR | Faithfulness | Citation acc. | p95 latency |
|---|---|---|---|---|---|
| Vector only | | | | | |
| BM25 only | | | | | |
| Hybrid (RRF) | | | | | |
| Hybrid + rerank | | | | | |

Also break results down by question type. Showing "BM25 wins on exact_term, vectors win on paraphrase, hybrid wins overall" is exactly the insight interviewers want to hear. Report real numbers, even if the gains are modest.

## 8. README checklist
1. One-line pitch + live demo link + 2-minute video/GIF.
2. The problem: why naive vector RAG fails on financial/legal documents.
3. Architecture diagram (draw.io or Mermaid).
4. **Results table** (section 7) with per-question-type breakdown.
5. Key design decisions and trade-offs (chunk size, why RRF over weighted scores, why cross-encoder, local vs API reranker cost/latency).
6. Quickstart: `git clone`, `cp .env.example .env`, `docker compose up`, `make ingest`, `make eval`.
7. API examples (curl for `/query`) and a sample response JSON.
8. Failure cases you found and what you'd do next (query rewriting, HyDE, ColBERT, fine-tuned embeddings, multilingual Hindi docs).
9. Tech stack badges, license.

## 9. Deployment
Decided (2026-10-28): **Streamlit Community Cloud**, live at https://hybrid-rag-finance.streamlit.app/. Hugging Face Docker Spaces were the first choice but now need a PRO plan (create_repo returned 402), and the user wants free hosting. Render/Railway free tiers (~512 MB RAM) can't hold the bge models + reranker.
- Built: with `API_URL=""` the Streamlit UI calls `answer()` in-process (no FastAPI). Prebuilt indexes (chunks.jsonl, bm25.pkl, embedded Qdrant store, ~66 MB) are committed because Community Cloud has no build step; it installs from `uv.lock`. Secrets: LLM_PROVIDER=gemini, LLM_MODEL=gemini-3.6-flash (free tier, no billing), RERANKER_MODEL=jinaai/jina-reranker-v1-turbo-en, RERANK_CANDIDATES=10. Measured live: hybrid_rerank ~11 s total (6 s retrieval, 5 s Gemini).
- Adding documents: PDF into data/raw + row in sources.csv -> ingest -> index -> run_eval (check no regression) -> commit data/processed + push (Cloud redeploys). Only *.pdf is parsed.
- Kept for a paid plan: `scripts/deploy_space.py` (HF Space). It stages tracked code + prebuilt indexes and uploads with `huggingface_hub` (`--dry-run` lists files; `--var` sets non-secret Space variables and refuses anything named *KEY*/*TOKEN*). The embedded Qdrant store ships inside the image, so Qdrant Cloud is optional (set `QDRANT_URL`), not required.
- Earlier HF plan detail: `scripts/deploy_space.py` stages tracked code + prebuilt indexes (~26 MB: chunks.jsonl, bm25.pkl, embedded Qdrant store) and uploads with `huggingface_hub` (`--dry-run` lists files; `--var` sets non-secret Space variables and refuses anything named *KEY*/*TOKEN*). The embedded Qdrant store ships inside the image, so Qdrant Cloud is optional (set `QDRANT_URL`), not required.
- Reranker latency is the deployment bottleneck: bge-reranker-base measured ~15 s for 30 × ~470-token pairs on CPU (not the ~0.3–1 s often quoted for short passages). For the public demo use Cohere Rerank, a faster local model, or fewer `RERANK_CANDIDATES`; pick using the eval's latency column.
- Pre-ingest the corpus locally and push vectors to Qdrant Cloud, so the deployed app doesn't need to parse PDFs at boot.
- Add a simple rate limit and keep API keys in environment secrets.
- Bonus: an AWS EC2 / GCP Cloud Run deployment is worth mentioning on your resume if you have credits.

## 10. Resume bullet (fill in your real numbers)
> Built a hybrid-search RAG system over 50+ Indian regulatory PDFs (RBI, SEBI, Budget) using BM25 + dense retrieval with RRF and a cross-encoder reranker; improved Recall@5 from X% to Y% and faithfulness from A to B on a hand-verified 80-question golden set. Dockerized FastAPI service with cited answers, deployed on Hugging Face Spaces + Qdrant Cloud.

Filled in (2026-10-28, real numbers; final wording in docs/launch_posts.md):
> Built a hybrid-search RAG system over 27 Indian regulatory PDFs (SEBI, Union Budget, IRDAI, PRS; 6,100 chunks) combining BM25 and dense retrieval with weighted RRF and a cross-encoder reranker; raised Recall@5 from 62.5% (vector-only) to 86.7% and MRR from 0.50 to 0.71 on a 74-question golden set with a held-out test split. Cited answers with hallucinated-citation filtering; Dockerized FastAPI + Streamlit, deployed free on Streamlit Community Cloud with Gemini.

## 11. Interview questions to prepare
- Why does BM25 beat embeddings on some queries? When does it lose?
- How does RRF work, and why not just add normalized scores?
- Bi-encoder vs cross-encoder: why rerank only the top 30?
- How did you pick chunk size? What happens with tables?
- How did you build and validate the golden set? How do you avoid eval leakage?
- How do you stop hallucinated citations?
- How would you scale to 1 million documents? Handle document updates?

## 12. Rules for Claude Code (working agreement)
- Work milestone by milestone; at the end of each, summarize what was built and how to run it.
- Explain non-obvious code in 2–3 sentences so I can learn it, especially retrieval, RRF, reranking and eval.
- Write tests for chunking, fusion, and API schemas. Run `ruff` and `pytest` before saying a step is done.
- Keep all settings (models, top-k, chunk size, RRF k, provider) in `config.py` / `.env`, never hardcoded.
- Never commit secrets or large PDFs.
- Don't fake eval numbers; always run the eval script and report what it prints.
- Prefer simple, readable code over clever abstractions.

---

## Starter prompt (paste into Claude Code)

```
I'm building the project described in CLAUDE.md in this repo: a Hybrid Search RAG system (BM25 + vector with RRF, cross-encoder reranker, cited answers) over Indian finance/policy PDFs, with a RAGAS-based eval. I have about 1 year of GenAI experience and I'm building this as a portfolio project for a job switch, so I want to understand every part.

Please read CLAUDE.md fully, then:
1. Propose any changes you'd make to the plan and wait for my OK.
2. Start Milestone Week 1, Day 1–2: scaffold the folder structure, pyproject.toml, config.py with pydantic-settings, .env.example, docker-compose.yml with Qdrant, a Makefile, and scripts/download_data.py driven by data/sources.csv (seed it with ~10 real RBI/SEBI/Budget PDF URLs I can verify).
3. Explain what you built, how to run it, and what we do next.

Work one milestone at a time and stop for my review at the end of each.
```
