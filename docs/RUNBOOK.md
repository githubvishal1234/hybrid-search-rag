# Runbook: how to run, change, evaluate and deploy this project

This is the complete operating manual. It assumes nothing from earlier chat sessions. Every
command here was run on this project; commands are written for **Windows PowerShell** (they work
the same in Git Bash, macOS and Linux).

Live demo: https://hybrid-rag-finance.streamlit.app/
Code: https://github.com/githubvishal1234/hybrid-search-rag

---

## 0. Status (2026-10-28)

| Area | State |
|---|---|
| Ingestion, chunking, BM25 + vector indexes | Done. Prebuilt indexes are **committed** in `data/processed/`, so a fresh clone answers questions with no PDFs and no ingestion |
| Retrieval: vector / bm25 / hybrid (weighted RRF) / hybrid_rerank | Done |
| Answers with citations, hallucinated-citation filter, refusals | Done |
| Eval: 74-question golden set, retrieval + generation metrics, dev/test split | Done |
| FastAPI service, Streamlit UI, Docker, GitHub Actions CI | Done |
| Live deployment on Streamlit Community Cloud (Gemini) | Done |
| Golden-set human spot-check | **Open** (section 9) |
| Demo video | **Open** (section 10) |
| Faithfulness from an independent judge | **Optional** (section 7.3) |

---

## 1. One-time setup (about 10 minutes)

### 1.1 Install the tools

1. **Python 3.11** (3.12 also works). Check: `python --version`.
2. **uv** (the package manager this project uses):
   ```powershell
   python -m pip install uv
   ```
   Everywhere below, `python -m uv` is used so it works even if `uv` isn't on PATH.
3. **Git** and (optional) **Docker Desktop** if you want the container route.
4. **An LLM**, one of:
   - **Ollama** (local, free, no key): install from ollama.com, then `ollama pull llama3.1`.
   - **An API key**: Gemini (free tier), OpenAI, or Anthropic. See section 2.

### 1.2 Get the code and install dependencies

```powershell
git clone https://github.com/githubvishal1234/hybrid-search-rag.git
cd hybrid-search-rag
python -m uv sync
```

**Windows: clone into a short path** such as `C:\code\hybrid-search-rag` or `D:\projects\...`.
Windows limits paths to 260 characters unless long paths are enabled, and some packages
(`anthropic`) have very long file names; in a deep folder the install looks fine but `import`
then fails with `ModuleNotFoundError`. (Verified: a clone under a deep temp folder failed, the
same clone at `D:\hr_fresh_test` passed all 65 tests and answered.)

`uv sync` creates `.venv/` with exact versions from `uv.lock` (including dev tools: pytest, ruff).
The first run of any command also downloads the embedding and reranker models (~1.3 GB, once,
cached afterwards).

### 1.3 Create your `.env`

```powershell
Copy-Item .env.example .env
```

Open `.env` in **VS Code or Notepad** (never Excel/Word) and set the LLM (section 2). `.env` is
gitignored: it is never committed or pushed. Never paste keys into chat, code or `.env.example`.

### 1.4 Check it works (no LLM needed)

```powershell
python -m uv run pytest -q
python -m uv run python -m hybrid_rag.cli "What does section 148A of the Income-tax Act provide?" --retrieve-only
```

Expected: all tests pass; the second command prints a ranking table (BM25 rank, vector rank,
RRF rank per chunk) with Finance Bill pages near the top.

---

## 2. Choosing the LLM (set in `.env`)

Only these lines matter. Set **one** provider.

**A. Ollama, local (default, free, slow on CPU: ~30-100 s per answer)**
```ini
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
```
Start Ollama with a 4k context first, or it can use 14+ GB RAM:
```powershell
$env:OLLAMA_CONTEXT_LENGTH = "4096"; ollama serve
```
(If the Ollama tray app is already running, quit it first so this setting applies.)

**B. Gemini (free tier, ~5 s per answer)**
Get a key at aistudio.google.com -> API Keys -> Create API key. Don't enable billing: then you can
never be charged; you just get "quota exceeded" errors at the limit.
```ini
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.6-flash
GEMINI_API_KEY=your-key
```
Free-tier limits are **per model, per project, per day** and reset at midnight Pacific
(~12:30 PM IST). `gemini-3.6-flash` allows only **20 requests/day**. `gemini-3.5-flash-lite`
and `gemini-3.1-flash-lite` also work on the free tier (tested) and usually allow far more; see
your exact limits at aistudio.google.com -> Rate Limit.

**C. OpenAI or Anthropic (paid)**
```ini
LLM_PROVIDER=openai          # or anthropic
LLM_MODEL=<model name from the provider's docs>   # e.g. claude-haiku-5-5 for anthropic
OPENAI_API_KEY=your-key      # or ANTHROPIC_API_KEY=your-key
```

**Optional: faster reranking**
```ini
RERANKER_MODEL=jinaai/jina-reranker-v1-turbo-en   # ~4x faster than bge-reranker-base, a bit weaker
RERANK_CANDIDATES=10                              # fewer chunks to re-score = faster
# or Cohere's hosted reranker (free trial key at cohere.com):
RERANKER_PROVIDER=cohere
COHERE_API_KEY=your-key
```

Every other knob (chunk size, top-k, RRF k and weight, rate limit) is in
`src/hybrid_rag/config.py` and can be overridden in `.env` by its upper-case name.

---

## 3. Running it

### 3.1 Ask from the terminal (quickest)

```powershell
python -m uv run python -m hybrid_rag.cli "What is the TCS rate on LRS remittances for education?"
python -m uv run python -m hybrid_rag.cli "your question" --mode hybrid_rerank
python -m uv run python -m hybrid_rag.cli "your question" --json          # full response JSON
python -m uv run python -m hybrid_rag.cli "your question" --retrieve-only # ranking only, no LLM
```
Modes: `vector`, `bm25`, `hybrid` (default), `hybrid_rerank`.

### 3.2 UI only (simplest: one process, no API server)

In `.env` set `API_URL=` (empty). Then:
```powershell
python -m uv run streamlit run ui/app.py
```
Open http://localhost:8501. This is exactly how the live demo runs.

### 3.3 API + UI (the "production" layout)

Leave `API_URL` unset (defaults to http://localhost:8000). Two terminals:
```powershell
python -m uv run uvicorn hybrid_rag.api.main:app --port 8000
```
```powershell
python -m uv run streamlit run ui/app.py
```
- API docs (try requests in the browser): http://localhost:8000/docs
- Health: http://localhost:8000/health
- Query from PowerShell:
  ```powershell
  Invoke-RestMethod -Method Post http://localhost:8000/query -ContentType "application/json" -Body '{"question": "What does section 148A provide?", "mode": "hybrid"}'
  ```
- Each query logs one JSON line with per-stage latency to the API terminal.
- Rate limit: 10 queries/minute per IP (`RATE_LIMIT_PER_MINUTE`).
- `POST /ingest` (rebuild indexes) is disabled unless you set `ADMIN_TOKEN=some-long-secret` in
  `.env`; then send it as the `X-Admin-Token` header.

### 3.4 Docker (one container: API on :8000, UI on :7860)

```powershell
docker compose up --build app
```
UI http://localhost:7860, API http://localhost:8000/docs. It reads your `.env`. For Ollama, run
Ollama on the host; the container reaches it via `host.docker.internal`.
Optional Qdrant server instead of the embedded store:
`docker compose --profile server up -d qdrant` and set `QDRANT_URL=http://qdrant:6333`
(then rebuild the index into it, section 4 step 4).

---

## 4. Adding or changing documents

The corpus is 27 PDFs (SEBI, Union Budget 2026-27, IRDAI, PRS income-tax summary). The 3 RBI
PDFs in `sources.csv` were never downloaded (RBI blocks scripts). Only `*.pdf` is parsed; convert
Word/HTML to PDF first. Hindi lines are dropped on purpose; scanned pages need Tesseract.

1. Add a row to `data/sources.csv` (edit in VS Code, not Excel, which rewrites the dates):
   `file_name,title,source_url,publisher,date` with date as `YYYY-MM-DD`.
2. Get the PDF into `data/raw/`:
   ```powershell
   python -m uv run python scripts/download_data.py   # skips files already present
   ```
   or save it there by hand with the exact `file_name`.
   On a fresh clone `data/raw/` is empty: run this to fetch the original 27 before re-ingesting,
   otherwise step 4 would index only your new file.
3. Parse and chunk (only new/changed files are parsed; cache keyed by file hash):
   ```powershell
   python -m uv run python -m hybrid_rag.ingestion.pipeline
   ```
4. Rebuild indexes (embeds new chunks into Qdrant, removes stale ones, rebuilds BM25):
   ```powershell
   python -m uv run python -m hybrid_rag.indexing.build
   ```
5. Try a question (section 3.1), then check nothing regressed (section 7.1). Recall@5 for
   hybrid_rerank should stay near 0.867.
6. To update the live demo: commit `data/processed/chunks.jsonl`, `data/processed/bm25.pkl`,
   `data/processed/qdrant/` and `data/sources.csv`, then `git push`. Streamlit Cloud redeploys
   by itself. Never commit PDFs (`data/raw/` is gitignored).

Optional OCR for image-only pages: install Tesseract (UB Mannheim build on Windows) and
`python -m uv add pytesseract`; re-running step 3 re-parses the affected documents.

If the topic changes (non-finance PDFs), also edit the system prompt in
`src/hybrid_rag/generation/prompts.py` ("Indian financial regulations") and the title, caption
and example questions in `ui/app.py`.

---

## 5. Project map (where things live)

```
src/hybrid_rag/
  config.py               all settings (read from .env)
  cli.py                  terminal Q&A
  ingestion/parse.py      PyMuPDF text + tables -> markdown, Hindi filter, OCR fallback
  ingestion/clean.py      headers/footers, hyphenation, chart-label noise
  ingestion/chunk.py      450-token chunks, 64 overlap, headings close chunks
  ingestion/pipeline.py   data/raw -> pages.jsonl -> chunks.jsonl
  indexing/embed.py       fastembed bge-small, "title / section + text"
  indexing/vector_store.py Qdrant (embedded or server), idempotent sync
  indexing/bm25_index.py  BM25 + finance-aware tokenizer
  indexing/build.py       builds both indexes
  retrieval/fusion.py     weighted Reciprocal Rank Fusion (own code)
  retrieval/rerank.py     cross-encoder (local or Cohere)
  retrieval/retriever.py  the 4 modes
  generation/prompts.py   numbered sources, cite-or-refuse rules
  generation/llm.py       one complete() for ollama/openai/anthropic/gemini
  generation/answer.py    citations -> [file, p.X], drops invented ones
  api/main.py, schemas.py FastAPI service
ui/app.py                 Streamlit (API mode or in-process mode)
eval/                     golden set, run_eval.py, metrics.py, judge.py, results/
scripts/                  download_data, build/review golden set, deploy_space, start.sh
tests/                    pytest suite (run before every commit)
data/processed/           committed prebuilt indexes (chunks.jsonl, bm25.pkl, qdrant/)
```

---

## 6. Development loop

```powershell
python -m uv run ruff check .            # lint
python -m uv run ruff format .           # auto-format
python -m uv run pytest -q               # tests (~15 s)
git add -A; git commit -m "message"; git push
```
GitHub Actions runs ruff, pytest and a Docker build on every push (`.github/workflows/`).
Check the result under the repo's Actions tab.

---

## 7. Evaluation

### 7.1 Retrieval metrics (free, ~1-2 minutes with the reranker, no LLM)

```powershell
python -m uv run python eval/run_eval.py --label my-run
```
Writes `eval/results/<timestamp>-my-run-retrieval.md/.csv`: Recall@5/10, MRR, nDCG@10 and p50/p95
latency per mode, plus Recall@5 by question type. Useful flags:
`--modes bm25 hybrid` (subset), `--split dev|test` (halves), `--sweep` (rrf_k x depth x weight
grid), `--limit 5` (smoke test).

Tuning rule (keeps numbers honest): tune on `--split dev`, report on `--split test` once.

### 7.2 Generation metrics (needs the LLM; slow on Ollama: ~3-4 hours for 4 modes x 74)

```powershell
python -m uv run python eval/run_eval.py --generate --no-retrieval --label gen
```
Adds citation accuracy, refusal accuracy, false refusals, uncited rate, latency. Resumable: if it
stops, re-run the **same command with the same label** and it skips answers already saved.
Gemini's free tier is too small for this (296 calls); use Ollama or a paid key.

### 7.3 Faithfulness judge (optional)

Scores what share of each answer's claims are supported by its sources.
In `.env`:
```ini
JUDGE_PROVIDER=gemini
JUDGE_MODEL=gemini-3.5-flash-lite    # use a model the live demo does NOT use (separate quota)
GEMINI_API_KEY=your-key
```
Then:
```powershell
python -m uv run python eval/judge.py eval/results/20261027-1925-v1-generation-generation.csv
```
About 220 answers to judge. Results are cached in `eval/results/judge_cache.jsonl`, so if the
daily quota runs out, re-run the same command the next day and it continues (4 are already
cached). It waits and retries on per-minute limits. The report is saved as
`...-faithfulness-gemini_<model>.md`. Put the numbers in the README's Faithfulness column and
replace the "self-graded" footnote.

### 7.4 Current results (what to quote)

Full set, current defaults (`eval/results/20261028-0100-v1-final-retrieval.md`):

| Mode | Recall@5 | MRR | Citation acc. | p95 retrieval |
|---|---|---|---|---|
| Vector only | 0.625 | 0.499 | 0.774 | 34 ms |
| BM25 only | 0.859 | 0.665 | 0.793 | 47 ms |
| Hybrid (weighted RRF) | 0.836 | 0.683 | 0.828 | 79 ms |
| Hybrid + rerank | 0.867 | 0.707 | 0.754 | 14.3 s |

Refusal accuracy (hybrid) 0.90. Faithfulness 0.92 is self-graded by llama3.1 and flat across
modes; don't quote it as evidence.

---

## 8. Deployment

### 8.1 Live demo: Streamlit Community Cloud (free)

Already deployed from `main`, entrypoint `ui/app.py`, Python 3.11. Every `git push` to `main`
redeploys automatically. Manage it at share.streamlit.io (sign in with GitHub) or via
"Manage app" at the bottom right of the live page.

Secrets (Manage app -> Settings -> Secrets), TOML format:
```toml
API_URL = ""
LLM_PROVIDER = "gemini"
LLM_MODEL = "gemini-3.6-flash"
GEMINI_API_KEY = "your-key"
RERANKER_MODEL = "jinaai/jina-reranker-v1-turbo-en"
RERANK_CANDIDATES = "10"
```
- **Quota errors on the live app**: `gemini-3.6-flash` = 20 questions/day for everyone combined.
  Change `LLM_MODEL` to `gemini-3.5-flash-lite` (tested on your key) and Save; it restarts in
  about a minute. Do this before sharing the link widely.
- **App asleep**: after a few days without visits it sleeps; anyone can click "wake up" (~1 min).
- **Rotate the key**: create a new one in AI Studio, update the secret, delete the old key.

### 8.2 Hugging Face Spaces (only with HF PRO)

Docker Spaces now require PRO (free accounts get "402 Payment Required"). With PRO:
```powershell
python -m uv run hf auth login
python -m uv run python scripts/deploy_space.py --space <user>/hybrid-rag --dry-run
python -m uv run python scripts/deploy_space.py --space <user>/hybrid-rag --var LLM_PROVIDER=gemini --var LLM_MODEL=gemini-3.5-flash-lite
```
Add `GEMINI_API_KEY` under the Space's Settings -> Variables and secrets (the script refuses to
upload keys).

### 8.3 Any server with Docker (Oracle free VM, EC2, Cloud Run)

Copy the repo and a `.env`, then `docker compose up -d --build app`. Expose port 7860 (UI)
and optionally 8000 (API).

---

## 9. Remaining task: golden-set spot-check (~1 hour, you)

Why: all 74 questions were checked by Claude, not a human. After this you can honestly write
"hand-verified" on your resume.

1. Open `eval/golden_set.jsonl` in VS Code. Each line has `question`, `expected_answer` and
   `relevant` (document + page numbers).
2. Pick ~15 questions across types (3 exact_term, 3 paraphrase, 3 numeric, 3 multi_hop,
   3 unanswerable). The PDFs are in `data/raw/` (or run `scripts/download_data.py` first).
3. For each: open the PDF at the listed page and check the answer is really there and correct.
   For unanswerable ones, search the PDFs (Ctrl+F) to confirm it isn't answered.
4. Fix any wrong line directly (keep it valid JSON on one line), then:
   ```powershell
   python -m uv run python scripts/review_golden_set.py --check
   python -m uv run python eval/run_eval.py --label after-spot-check
   ```
5. If you changed nothing material, the numbers stay the same. Record "15/74 spot-checked by
   hand, N fixes" in the README's golden set section.

To add your own questions interactively: `python -m uv run python scripts/review_golden_set.py --add`.

---

## 10. Remaining task: 2-minute demo video (you)

Use Loom or OBS. Wake the live app first (open it once a few minutes before).

1. 0:00-0:15 The problem: vector-only RAG misses exact terms like section numbers in legal text.
2. 0:15-0:45 Ask an exact-term question (e.g. section 148A) in **vector** mode, then in
   **hybrid + rerank**. Show citations; expand a source. (Try vector mode first off-camera to
   make sure it actually misses; if not, pick another exact-term question.)
3. 0:45-1:05 Open "Retrieval debug": BM25 rank vs vector rank vs rerank score per chunk.
4. 1:05-1:20 Ask "What is the GST rate on cryptocurrency trading?" -> it refuses (correct).
5. 1:20-2:00 README results table: Recall@5 62.5% -> 86.7%, dev/test split, honest caveats.

Then put the video link in README line 10 (replace "_coming soon_"), add a screenshot
(`docs/screenshot.png`, referenced as `![screenshot](docs/screenshot.png)`), commit, push.
Resume bullets and the LinkedIn post are ready in `docs/launch_posts.md`.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `LLM (ollama) is unreachable` / 503 | Ollama not running | `ollama serve` (with `OLLAMA_CONTEXT_LENGTH=4096`) |
| `MemoryError` or Ollama using 14 GB | Ollama context 8192 | Restart Ollama with `OLLAMA_CONTEXT_LENGTH=4096` |
| `429 ... quota exceeded` (Gemini) | Free-tier daily limit for that model | Wait for midnight Pacific, or switch `LLM_MODEL` to a Flash-Lite model |
| Qdrant "already accessed by another instance" | Embedded store opened by two processes (e.g. API + CLI) | Stop the other process, or run a Qdrant server (`QDRANT_URL`) |
| UI says "API not reachable" | UI in API mode without the API | Start uvicorn (3.3) or set `API_URL=` empty (3.2) |
| First query takes ~1 min | Models load/download on first use | Normal; later queries are fast |
| `hybrid_rerank` takes ~15 s | bge-reranker-base on CPU, 30 candidates | Jina turbo reranker, `RERANK_CANDIDATES=10`, or Cohere |
| Settings in `.env` ignored | Typo, or Excel/Word saved the file | Edit in VS Code; names are case-insensitive, `KEY=value`, no spaces around `=` |
| `ModuleNotFoundError` inside `anthropic` (or another package) right after a fresh install | Windows 260-character path limit | Clone into a short path (section 1.2), delete `.venv`, `python -m uv sync` again |
| `huggingface-cli` deprecated | Renamed | `python -m uv run hf auth login` |
| Streamlit Cloud build fails | Dependency or Python version | Check the log in Manage app; Python must be 3.11 |
| Dates in `sources.csv` changed | Excel re-saved it | `git restore data/sources.csv`; edit in VS Code |

---

## 12. Security checklist

- Keys only in `.env` (local) or platform secrets (Streamlit / HF). `.env` is gitignored.
- If a key is ever pasted into chat, a commit or a screenshot: delete it at the provider and make a
  new one. For a committed key, rotating is required; deleting the commit is not enough.
- Keep Gemini billing **off** to make charges impossible on the demo key.
- `POST /ingest` stays disabled unless `ADMIN_TOKEN` is set.
