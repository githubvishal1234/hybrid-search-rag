# Roadmap: from 7.5/10 to 10/10

A self-contained improvement plan. Each task says **why** it matters, **what to change** (files),
**how to measure** it, and **when it is done**. Work top to bottom; stop at any level. Commands are
PowerShell, as in [RUNBOOK.md](RUNBOOK.md).

Scoring is for a GenAI-engineer portfolio. Where it stands on 2026-10-28: **7.5/10**. Strong:
measured hybrid retrieval, own RRF, reranker, dev/test split, cited answers, deployed. Weak: small
eval set, self-graded faithfulness, no LLMOps, no real users.

| Level | Adds | Time |
|---|---|---|
| 8.5 | Phase 0: demo stability, video, spot-check | 1 day |
| 9 | Phase 1: rigorous evaluation | ~1 week |
| 9.5 | Phase 2 + 3: better retrieval, LLMOps | ~2 weeks |
| 10 | Phase 4: real users and a public write-up | ~2-3 weeks |
| (role-specific) | Phase 5: agentic, multilingual, conversation | optional |

Rules for every task (they keep the numbers honest):
- Tune on `--split dev`; report on `--split test` once, at the end of a phase.
- Never edit a number into the README by hand: copy it from an `eval/results/` report.
- Keep the old result next to the new one. Losses get written down too (README "Failure cases").
- `python -m uv run ruff check .` and `python -m uv run pytest -q` before every commit.

Progress checklist (tick as you go):

- [ ] 0.1 Demo on a high-limit model
- [ ] 0.2 Golden-set spot-check
- [ ] 0.3 Demo video + screenshot
- [ ] 1.1 Golden set to 200 questions
- [ ] 1.2 Confidence intervals and significance tests
- [ ] 1.3 Independent faithfulness judge, validated
- [ ] 1.4 Answer correctness metric
- [ ] 1.5 Chunk-size experiment
- [ ] 2.1 Stronger embedder
- [ ] 2.2 Query rewriting / HyDE
- [ ] 2.3 Fine-tuned embedder
- [ ] 2.4 Metadata filters
- [ ] 2.5 Cheaper reranking
- [ ] 3.1 Tracing and cost per query
- [ ] 3.2 Eval in CI (regression gate)
- [ ] 3.3 Streaming answers
- [ ] 3.4 Response cache
- [ ] 3.5 Feedback buttons
- [ ] 4.1 Real users
- [ ] 4.2 Learn from their questions
- [ ] 4.3 Public write-up

---

## Phase 0: get to 8.5 (1 day)

### 0.1 Put the live demo on a high-limit model (10 minutes)
**Why:** `gemini-3.6-flash` allows 20 requests/day; a recruiter can hit a quota error.
**Do:** Streamlit app -> Manage app -> Settings -> Secrets: `LLM_MODEL = "gemini-3.5-flash-lite"`
(tested on your key). Check your daily limits at aistudio.google.com -> Rate Limit.
**Done when:** ask 3 questions on the live app; all answer; README/RUNBOOK mention the new model.

### 0.2 Human spot-check of the golden set (1 hour)
**Why:** all 74 questions were reviewed with an AI assistant. You can't say "hand-verified" yet.
**Do:** follow RUNBOOK section 9 (15 questions, 3 per type, check each against the PDF page).
**Done when:** README states "N/74 spot-checked by hand, M corrected"; eval re-run if anything changed.

### 0.3 Demo video and screenshot (1-2 hours)
**Do:** script in RUNBOOK section 10. Add the link to README line 10 and
`![screenshot](docs/screenshot.png)` under it.
**Done when:** README shows the video link and an image of the app.

---

## Phase 1: rigorous evaluation (to 9/10, ~1 week)

The biggest weakness: with 64 answerable questions, hybrid + rerank (0.867) vs BM25 (0.859) is one
question's difference. These tasks turn "looks better" into "is better, with evidence".

### 1.1 Grow the golden set to 200 questions (2-3 days)
**Why:** bigger sample = smaller error bars = claims that survive a sharp interviewer.
**Do:**
1. Draft more: `python -m uv run python scripts/build_golden_set.py --n 150 --seed 11`
   (needs an LLM; uses `.env`'s provider).
2. Review: `python -m uv run python scripts/review_golden_set.py` (keep / edit / drop).
   Enforce: paraphrase questions must not reuse the passage's distinctive words.
3. Write at least 60 by hand from the PDFs: `python -m uv run python scripts/review_golden_set.py --add`.
4. Keep the type mix: ~25% exact_term, 25% paraphrase, 20% numeric, 15% multi_hop, 15% unanswerable.
5. Validate: `python -m uv run python scripts/review_golden_set.py --check`.
6. Save as `eval/golden_set.jsonl` (v2). Keep v1 as `eval/golden_set_v1.jsonl` for comparison.
**Measure:** rerun everything: `python -m uv run python eval/run_eval.py --label v2`.
**Done when:** >= 200 questions, >= 30% hand-written, report committed, README table updated with
v2 numbers (v1 kept in a "previous" row).

### 1.2 Confidence intervals and significance tests (half a day)
**Why:** "hybrid beats BM25" needs a p-value or an interval, not just a bigger number.
**Do:** in `eval/metrics.py` add:
- `bootstrap_ci(values, n=2000, seed=0) -> (low, high)`: resample per-question scores with
  replacement, take the 2.5th/97.5th percentile of the mean.
- `paired_bootstrap_p(a, b, n=2000)`: per-question differences between two modes; share of
  resamples where the mean difference <= 0.
In `eval/run_eval.py` report, for each mode, `Recall@5 (95% CI)` and a table "mode vs BM25:
difference, p". Use the per-question CSV rows that `run_retrieval` already produces.
Add tests in `tests/test_metrics.py` (CI contains the mean; identical inputs give p ~ 0.5).
**Done when:** README results table shows CIs, and states plainly which differences are significant.

### 1.3 Independent faithfulness judge, and validate the judge (1 day)
**Why:** self-graded faithfulness is ~0.92 for every mode, so it measures nothing.
**Do:**
1. In `.env`: `JUDGE_PROVIDER=gemini`, `JUDGE_MODEL=gemini-3.5-flash-lite` (a model the demo does
   not use, so quotas don't collide), or a paid model (e.g. Anthropic `claude-haiku-5-5`).
2. `python -m uv run python eval/judge.py eval/results/20261027-1925-v1-generation-generation.csv`
   (resumable; cached in `eval/results/judge_cache.jsonl`).
3. **Validate the judge:** hand-label 30 answers (supported / not, per claim) in a small JSONL, and
   compute judge-vs-human agreement (accuracy, Cohen's kappa). Add `eval/judge_agreement.py`.
**Done when:** README faithfulness column comes from the independent judge, with "judge agreement
with human labels: X% (kappa Y) on 30 answers".

### 1.4 Answer correctness (half a day)
**Why:** faithful != correct. An answer can faithfully repeat the wrong chunk.
**Do:** new metric in `eval/judge.py` (or `eval/correctness.py`): give the judge the question,
`expected_answer` and the system answer; ask "does the answer contain the expected facts?
(correct / partially / incorrect)". Score 1 / 0.5 / 0. Cache like the faithfulness judge.
**Done when:** README has a "Correctness" column per mode and per question type.

### 1.5 Chunk-size experiment (half a day + compute)
**Why:** it's in the original plan, it's a classic interview question, and labels are page-based so
results stay comparable.
**Do:** for each size in 256, 350, 450 (and overlap 32/64):
```powershell
$env:CHUNK_SIZE = "256"; python -m uv run python -m hybrid_rag.ingestion.pipeline; python -m uv run python -m hybrid_rag.indexing.build; python -m uv run python eval/run_eval.py --split dev --label chunk256
```
Run on the dev split only; pick the best; report on test once. Restore the chosen size in
`config.py` and rebuild + recommit the indexes (`data/processed/`).
(Needs `data/raw/` PDFs: run `scripts/download_data.py` first on a fresh clone.)
**Done when:** README has a chunk-size table (Recall@5, MRR, chunk count, index size) and the
default is justified by it.

---

## Phase 2: better retrieval (toward 9.5)

Measure every change with `--split dev`, keep it only if it helps, report on test at the end.

### 2.1 Stronger embedder (half a day)
**Why:** vector-only is the weakest part (0.625).
**Do:** `.env`: `EMBEDDING_MODEL=BAAI/bge-base-en-v1.5` (768 dims; slower, better) then rebuild the
index (`indexing.build` recreates the collection when the vector size changes). Also try
`BAAI/bge-m3` if fastembed supports it in your version (also multilingual, long context).
**Done when:** a table of embedder vs vector/hybrid Recall@5, MRR, index size and query latency;
the chosen default is committed.

### 2.2 Query rewriting / HyDE (1 day)
**Why:** vague or paraphrased questions; a frequent interview topic.
**Do:** new `src/hybrid_rag/retrieval/rewrite.py`:
- *Rewrite:* LLM turns the question into a keyword-rich search query (feed to BM25).
- *HyDE:* LLM writes a short hypothetical answer; embed it instead of the question (feed to dense).
Add modes `hybrid_rewrite` / `hybrid_hyde` in `retriever.py` (and the `Mode` type), config flag
`QUERY_REWRITE=none|rewrite|hyde`. Cache rewrites (LLM calls cost latency).
**Done when:** per-type table shows where it helps (expect paraphrase) and the latency cost.

### 2.3 Fine-tune the embedder (2-3 days; strong resume signal)
**Why:** "fine-tuned embeddings on domain data" is a standout line.
**Do:**
1. Training pairs: for ~2,000 chunks, have an LLM write 1-2 questions each (reuse
   `build_golden_set.py` prompts). **Exclude every golden-set page** from training (leakage).
2. Hard negatives: top BM25/vector hits that are not the source chunk.
3. Fine-tune bge-small with `sentence-transformers` (MultipleNegativesRankingLoss), on a free GPU
   (Colab/Kaggle) or your GTX 1650. This adds PyTorch, so keep it in a separate script
   (`scripts/finetune_embedder.py`) and a separate dependency group.
4. Export to ONNX for fastembed, or switch dense embedding to sentence-transformers for that model.
**Done when:** vector and hybrid Recall@5 on the **test split** before vs after, plus training
details (pairs, epochs, time) in the README.

### 2.4 Metadata filters (half a day)
**Why:** real users want "only SEBI" or "only Budget 2026-27".
**Do:** `QueryRequest` gets optional `publisher: list[str]`, `year`. Dense: Qdrant payload filter.
BM25: filter results by chunk metadata after scoring. UI: multiselect in the sidebar.
Tests in `tests/test_api.py`.
**Done when:** a filtered query only returns chunks from the selected publishers (tested).

### 2.5 Cheaper reranking (half a day)
**Why:** bge-reranker-base costs ~14 s per query on CPU.
**Do:** options to measure: `RERANK_CANDIDATES` 10/20/30; jina turbo model; rerank only when BM25
and vector top-5 disagree (a "conditional rerank" flag in `retriever.py`).
**Done when:** a quality-vs-latency table and a chosen default for both local and live use.

---

## Phase 3: production / LLMOps (toward 9.5)

### 3.1 Tracing, tokens and cost per query (1 day)
**Why:** "LLMOps / observability" is in most GenAI job descriptions.
**Do:** add Langfuse (free cloud tier or self-hosted) or OpenTelemetry. Trace spans: dense, bm25,
fusion, rerank, llm. Record prompt/completion tokens (from the provider responses in `llm.py`) and
cost per query (price table in config). Keys in `.env` only.
**Done when:** a screenshot of a trace in the README, and "average cost per query: $X" measured.

### 3.2 Eval in CI as a regression gate (half a day)
**Why:** very few portfolio projects do this; it shows engineering maturity.
**Do:** new job in `.github/workflows/`: `uv sync`, then
`python eval/run_eval.py --modes bm25 hybrid --label ci` (no LLM, no reranker: fast and free), then
a small script that fails if Recall@5 drops more than 0.02 below a committed baseline file
(`eval/baseline.json`). Model downloads need caching (`actions/cache` on the fastembed cache dir).
**Done when:** a PR that breaks retrieval fails CI (test it once on a branch).

### 3.3 Streaming answers (half a day)
**Do:** `llm.complete_stream()` for the providers; FastAPI `StreamingResponse` (or SSE) endpoint
`/query/stream`; Streamlit `st.write_stream`. Citations are validated after the stream ends.
**Done when:** text appears token by token in the UI; citations still validated.

### 3.4 Response cache (2 hours)
**Do:** cache key = normalised question + mode + config hash; store answer JSON in SQLite or a
JSON file; TTL; bypass flag. Log hits.
**Done when:** a repeated question returns in < 100 ms; cache hit rate shown in logs.

### 3.5 Feedback buttons (half a day)
**Do:** thumbs up/down + optional comment in the UI, stored with question, mode, answer, chunk ids
(SQLite locally; a free hosted DB such as Supabase for the live app; never collect personal data).
**Done when:** feedback rows are stored and exportable; enables Phase 4.

---

## Phase 4: real users and impact (to 10/10)

This is what separates 9 from 10: evidence that people used it and that you improved it from what
you learned.

### 4.1 Get 10-20 real users (1-2 weeks, mostly waiting)
**Who:** CA/CS/CFA students, finance colleagues, LinkedIn connections.
**Do:** share the live link with a 2-line ask ("ask it 3 questions you'd actually ask at work;
use the thumbs"). Ensure the demo model has enough daily quota first (0.1). Add a short privacy
note in the UI ("questions are logged anonymously to improve the system").
**Done when:** >= 100 real questions logged.

### 4.2 Learn from real questions (2-3 days)
**Do:**
1. Export questions + feedback. Categorise failures: retrieval miss, wrong answer, should have
   refused, out of corpus, slow.
2. Add 30-50 real questions (anonymised, reviewed) to the golden set as a new "real_user" slice.
3. Fix the top failure category; measure before/after on that slice.
**Done when:** README section "What real users taught me": failure breakdown, the fix, the
before/after number.

### 4.3 Public write-up (1-2 days)
**Do:** a Medium / LinkedIn article, e.g. "Why textbook RRF lost to BM25 on Indian legal text, and
how I fixed it". Include the results table, the RRF failure explanation, the dev/test protocol,
and one chart (Recall@5 by question type per mode).
**Done when:** published, linked from the README and your resume.

---

## Phase 5: breadth (optional, pick by target role)

| Task | For roles | Sketch |
|---|---|---|
| **Agentic layer** | AI agent engineer | LLM chooses tools: `search_documents`, `calculate` (tax/percentage math), `ask_clarifying_question`; max 3 steps; log the tool trace; evaluate on multi_hop questions |
| **Multi-turn conversation** | Product-facing GenAI | Keep last N turns; rewrite follow-ups into standalone questions before retrieval ("what about for NRIs?") |
| **Multilingual (Hindi)** | India-market roles | Stop dropping Hindi lines (`MAX_NON_ENGLISH_RATIO`), bge-m3 embeddings, Hindi golden questions; report English vs Hindi metrics |
| **OCR for image-only pages** | Document AI | Install Tesseract; re-ingest; count recovered pages; add questions on them |
| **RBI documents** | Completeness | Save the 3 RBI PDFs by hand into `data/raw/`; re-ingest; add RBI questions |
| **Scale test** | Platform roles | Index 500+ documents; Qdrant server with HNSW + quantization; measure latency and memory vs size |

---

## Resume line after each level

- **8.5:** current bullets in `docs/launch_posts.md`, plus "hand-verified" once 0.2 is done.
- **9:** "...on a 200-question hand-verified golden set; hybrid + rerank significantly outperforms
  vector-only (p < 0.01) and [BM25: significant or not]; faithfulness X by an independent judge
  validated against human labels (kappa Y)."
- **9.5:** add "fine-tuned bge embedder (+Z Recall@5), Langfuse tracing with cost per query, eval
  regression gate in CI."
- **10:** add "used by N people (M questions); real-user failure analysis drove a fix improving
  accuracy on real queries from A to B; published write-up."

## What not to do

- Don't switch to LangChain/LlamaIndex: owning the pipeline is the point of this project.
- Don't tune on the test split, and don't quote a number you didn't get from a report.
- Don't add features without a measurement attached; an unmeasured feature adds little here.
- Don't polish beyond 9 if job hunting is urgent: a second, different project (agent-based) adds
  more than the last point here.
