# RAG Evaluation Pipeline — Project Brief

> How to use this file: this project lives **inside your existing Hybrid Search RAG repo** (`hybrid-rag/`). Save this file as `docs/EVAL_CLAUDE.md` and add one line to your root `CLAUDE.md`: "For eval work, follow docs/EVAL_CLAUDE.md." Open Claude Code in the repo and paste the **Starter prompt** from the bottom. Work one step at a time, and ask Claude Code to explain anything you don't understand, because interviewers will dig into exactly this.

This is project **#2** from your Top 10 list. Project #1 built the system. This project **proves how good it is, shows which change made it better, and stops it from silently getting worse.**

---

## 1. Goal

Build a reusable evaluation pipeline for your Hybrid RAG that:

1. Has a **hand-verified golden set of 80 questions** (minimum 50) with expected answers, relevant pages and question types.
2. Scores **retrieval** with your own code: Hit@k, Recall@k, MRR, nDCG@10.
3. Scores **generation** with **RAGAS**: faithfulness, response relevancy, context precision, context recall. Plus two of your own: **citation accuracy** and **refusal accuracy**.
4. Runs **experiments**: one command evaluates any config (retrieval mode, chunk size, top-k, reranker, prompt version) and saves a versioned results file.
5. Produces a **before/after table** for each improvement, broken down by question type. This is the core of your README and resume.
6. Logs every run to **Langfuse** (or LangSmith) as a dataset experiment, so you can click into any bad answer and see its retrieved chunks.
7. **Calibrates the LLM judge** against your own labels on 30 answers, so you can say how much to trust it.
8. Runs in **CI**: a fast pytest gate on every PR (retrieval metrics on a 20-question smoke set) and a full nightly/manual RAGAS run, failing if quality drops past a threshold.

The point for the job switch: almost every candidate says "I built RAG". Very few can say "faithfulness went from 0.71 to 0.89, here's the experiment that did it, and CI blocks any PR that drops Recall@5 by more than 3 points." That sentence is the whole project.

## 2. What you already have (from project #1)

- Ingestion → chunks with `doc_id, file_name, page_start, page_end, chunk_id` metadata.
- `retriever.py` with modes `vector | bm25 | hybrid | hybrid_rerank`.
- Answer generation with citations `[file_name, p.X]` and an "I couldn't find this" refusal.
- Possibly a first `eval/` folder. If so, this project replaces and extends it.

If any of these are missing, Step 1 fixes them.

## 3. Architecture

```
eval/golden_set.jsonl ──┐
                        │      ┌────────────── experiments/*.yaml ──────────────┐
                        ▼      ▼                                                │
                 run_experiment.py  (loads config, freezes git sha + settings)  │
                        │                                                       │
            for each question:                                                  │
                        ▼                                                       │
        hybrid_rag.pipeline.run_rag(question, config)                           │
          → RagResult{answer, contexts[], citations[], latency, tokens}         │
                        │                                                       │
          ┌─────────────┼───────────────────────┬────────────────────┐          │
          ▼             ▼                       ▼                    ▼          │
   retrieval metrics   RAGAS (judge LLM)   citation check     refusal check     │
   Hit/Recall/MRR/     faithfulness,       cited page ∈       unanswerable →    │
   nDCG (own code)     relevancy, ctx P/R  relevant pages     refused?          │
          └─────────────┴───────────┬───────────┴────────────────────┘          │
                                    ▼                                           │
          eval/results/<run_id>/  per_question.jsonl + summary.json + report.md │
                                    │                                           │
                ┌───────────────────┼─────────────────────┐                     │
                ▼                   ▼                     ▼                     │
        compare.py (A vs B     Langfuse dataset      pytest gate (CI)  ◄────────┘
        table, by q-type)      run + traces          thresholds vs baseline
```

Key idea: **one function, `run_rag()`, is the only door into your system.** The API, the UI and the eval all call it, so you evaluate exactly what you ship.

## 4. Golden set

### 4.1 Schema (`eval/golden_set.jsonl`, one JSON per line)
```json
{
  "id": "q001",
  "question": "What is the maximum LTV ratio for gold loans given by NBFCs?",
  "type": "exact_term",
  "expected_answer": "75% of the value of the gold jewellery.",
  "relevant": [{"file_name": "rbi_gold_loan_nbfc.pdf", "pages": [3]}],
  "answerable": true,
  "difficulty": "easy",
  "source": "llm_draft_reviewed",
  "notes": ""
}
```

### 4.2 Mix (for 80 questions)
| Type | Count | What it tests |
|---|---|---|
| `exact_term` | 20 | section numbers, form names, acronyms (BM25 should win) |
| `paraphrase` | 20 | same meaning, different words (vectors should win) |
| `numeric_table` | 15 | figures from tables (tests parsing + chunking) |
| `multi_hop` | 12 | needs 2 chunks or 2 documents |
| `unanswerable` | 13 | not in corpus, or a trap (wrong year, made-up rule) |

### 4.3 How to build it
1. `scripts/build_golden_set.py` samples chunks (stratified by document, so one big PDF doesn't dominate) and asks an LLM to draft question + answer + type. You can also try RAGAS `TestsetGenerator`, and compare.
2. **Review every row by hand** in a simple Streamlit review page or a spreadsheet: fix wording, check the page, delete bad ones. Mark `source: llm_draft_reviewed` or `handwritten`.
3. Write at least **15 questions yourself** the way a real user would ask (short, vague, Hinglish-style English, typos). LLM-drafted questions are too clean and copy the chunk's words, which flatters BM25.
4. Split: `golden_set.jsonl` (80, used for reporting) and `smoke_set.jsonl` (20, subset, used in CI). Never tune prompts while looking at individual answers in a held-out slice: keep 20 questions as `holdout` and only look at them at the end.

## 5. Metrics

### 5.1 Retrieval (your own code, `eval/metrics/retrieval.py`, no LLM, free)
A retrieved chunk is "relevant" if its `file_name` matches and its page range overlaps a relevant page.
- **Hit@k**: is at least one relevant chunk in the top k? (k = 1, 3, 5, 10)
- **Recall@k**: fraction of the relevant pages covered in top k.
- **MRR**: 1 / rank of the first relevant chunk.
- **nDCG@10**: rewards relevant chunks ranked higher.
- Skip unanswerable questions for these.

Write these yourself (each is 5 to 15 lines) with unit tests on tiny hand-made examples. Interviewers ask you to explain MRR vs nDCG.

### 5.2 Generation (RAGAS, `eval/metrics/generation.py`)
- **Faithfulness**: are the answer's claims supported by the retrieved context? (hallucination)
- **Response relevancy**: does the answer address the question?
- **Context precision (with reference)**: are the relevant chunks ranked near the top?
- **Context recall**: does the context contain what's needed for the reference answer?

Notes:
- Use the current RAGAS API (`EvaluationDataset` / `SingleTurnSample`, metric classes, wrapped judge LLM). The API changed a lot between versions, so pin the version in `pyproject.toml`.
- Pick **one judge model** and keep it fixed across all experiments (e.g. a cheaper model than your generator, temperature 0). Changing the judge changes the numbers.
- Cache judge calls on disk (hash of inputs → score) so re-running is cheap.

### 5.3 Your own checks (`eval/metrics/custom.py`, no LLM)
- **Citation accuracy**: % of answers where every cited `[file, p.X]` is a relevant page (strict) or at least one is (lenient).
- **Refusal accuracy**: on unanswerable questions, % refused. On answerable ones, % *wrongly* refused (false refusals). Report both; it's a trade-off.
- **Latency** p50/p95 per stage and **cost per query** (tokens × price).

### 5.4 Judge calibration
Label 30 answers yourself (faithful: yes/no; correct: yes/partial/no). Compare with RAGAS faithfulness (threshold at 0.5) and report agreement % or Cohen's kappa. Note the cases where the judge is wrong. This takes 1 to 2 hours and is one of the strongest things you can say in an interview.

## 6. Experiments (the before/after ladder)

Each experiment is a YAML file in `experiments/`; only change **one thing** at a time.

| # | Experiment | Change | Hypothesis |
|---|---|---|---|
| E0 | `baseline_vector` | vector only, chunk 512, top-5 | baseline |
| E1 | `bm25_only` | bm25 | better on exact_term, worse on paraphrase |
| E2 | `hybrid_rrf` | RRF k=60 | best of both |
| E3 | `hybrid_rerank` | + bge-reranker, top-30 → 5 | higher precision, more latency |
| E4 | `chunk_256` / `chunk_1024` | chunk size | tables and multi_hop shift |
| E5 | `prompt_v2` | stricter "cite every claim, refuse if unsure" | faithfulness ↑, false refusals ↑? |
| E6 | `query_rewrite` or `hyde` | optional | paraphrase ↑ |

Run each with 3 repeats for generation metrics if budget allows, or at least note that LLM metrics have noise (±0.02 to 0.05). Don't claim a win smaller than the noise.

## 7. Langfuse / LangSmith integration
- Upload the golden set as a **Langfuse dataset** once (`scripts/upload_dataset.py`).
- Each experiment run = a **dataset run** named `<experiment>_<git_sha>`, with one trace per question (retrieval spans, rerank span, generation span) and scores attached (faithfulness, hit@5, citation_ok).
- In the README, add a screenshot of two runs compared side by side and one trace of a failure.
- If you already used Langfuse in the multi-agent project, reuse the same self-hosted docker-compose service.

## 8. Testing and CI

### 8.1 pytest layers
- `tests/unit/`: metrics on toy data (MRR, nDCG, overlap logic, citation parser), golden set schema validation (every row valid, ids unique, types balanced).
- `tests/eval/test_retrieval_gate.py` (marker `@pytest.mark.eval`): runs retrieval on the 20-question smoke set, no LLM, and asserts `hit@5 >= baseline - 0.03` and `mrr >= baseline - 0.03`, reading baselines from `eval/baselines.json`.
- `tests/eval/test_generation_gate.py` (marker `@pytest.mark.llm_eval`): RAGAS on the smoke set; only runs when an API key is present.

### 8.2 GitHub Actions
- `ci.yml` on every PR: ruff + unit tests + retrieval gate (needs a small prebuilt index: commit a tiny test corpus or cache the Qdrant snapshot as an artifact). Posts a markdown summary table to the job summary.
- `eval-full.yml`: manual (`workflow_dispatch`) or nightly, runs the full 80 with RAGAS, uploads `report.md` as an artifact, keeps cost under a budget (fail fast if estimated cost > ₹X).
- Updating `baselines.json` is a deliberate commit, never automatic.

## 9. Stack

| Layer | Choice | Notes |
|---|---|---|
| Eval framework | **RAGAS** (pinned) | generation metrics |
| Retrieval metrics | own code | numpy only |
| Tracking | **Langfuse** (self-hosted, Docker) or LangSmith | datasets + experiment runs |
| Judge LLM | one fixed model, temp 0 | config switch, disk cache |
| Configs | YAML + pydantic | one file per experiment |
| Tests | pytest + markers | `unit`, `eval`, `llm_eval` |
| CI | GitHub Actions | PR gate + full manual/nightly run |
| Reports | pandas → markdown | `compare.py A B` |
| Review UI (optional) | Streamlit | golden set review + failure browser |

## 10. Folder structure (added to `hybrid-rag/`)

```
hybrid-rag/
├── docs/EVAL_CLAUDE.md
├── src/hybrid_rag/
│   └── pipeline.py            # run_rag(question, config) -> RagResult  (Step 1)
├── eval/
│   ├── golden_set.jsonl
│   ├── smoke_set.jsonl
│   ├── baselines.json
│   ├── schema.py              # GoldenItem, RagResult, EvalRow (pydantic)
│   ├── metrics/
│   │   ├── retrieval.py
│   │   ├── generation.py      # RAGAS wrapper + judge cache
│   │   └── custom.py          # citation + refusal accuracy
│   ├── run_experiment.py      # python -m eval.run_experiment experiments/hybrid_rerank.yaml
│   ├── compare.py             # python -m eval.compare <run_a> <run_b>
│   ├── calibrate_judge.py
│   └── results/<run_id>/      # per_question.jsonl, summary.json, report.md (gitignored except summaries)
├── experiments/
│   ├── baseline_vector.yaml
│   ├── hybrid_rrf.yaml
│   └── hybrid_rerank.yaml
├── scripts/
│   ├── build_golden_set.py
│   └── upload_dataset.py
├── tests/
│   ├── unit/test_retrieval_metrics.py
│   ├── unit/test_golden_set_schema.py
│   └── eval/test_retrieval_gate.py
└── .github/workflows/{ci.yml, eval-full.yml}
```

## 11. Step-by-step plan (~2 weeks, ~2 hours on weekdays)

**Step 1 — Make the RAG evaluable (Day 1)**
`run_rag(question, config) -> RagResult` with `answer`, `contexts` (chunk_id, file_name, pages, text, rank, scores), `citations`, `refused`, `latency_ms` per stage, `tokens`. API and UI switched to call it. `RagConfig` pydantic model covering mode, top_k, rerank_top_n, chunk_size, prompt_version. ✅ Done when one script prints a full `RagResult` for 3 questions in all 4 modes.

**Step 2 — Golden set schema + draft (Day 2–3)**
`eval/schema.py`, `build_golden_set.py`, 100 drafts generated. Schema test passes.

**Step 3 — Human review (Day 4–5)**
Review all drafts, keep ~65, write 15 by hand, tag types, split smoke + holdout. ✅ Done when the type mix matches section 4.2.

**Step 4 — Retrieval metrics + first experiment runner (Day 6–7)**
Own metrics with unit tests, `run_experiment.py` saving `results/<run_id>/`, E0 to E3 retrieval-only. ✅ Done when you have a retrieval table for 4 modes by question type.

**Step 5 — RAGAS + custom metrics (Day 8–9)**
RAGAS wrapper with judge cache, citation and refusal accuracy, full E0 to E3 run. `compare.py`.

**Step 6 — Langfuse (Day 10)**
Dataset upload, runs as dataset experiments, scores on traces.

**Step 7 — Judge calibration + more experiments (Day 11–12)**
Label 30, compute agreement. Run E4 to E5 (and E6 if time).

**Step 8 — pytest gate + CI (Day 13)**
Markers, baselines.json, both workflows green.

**Step 9 — Write-up (Day 14)**
README eval section, results tables, Langfuse screenshots, "what I learned" (3 surprising failure cases), resume bullet, LinkedIn post.

## 12. Results table for the README
| Experiment | Hit@5 | MRR | nDCG@10 | Faithfulness | Ctx precision | Ctx recall | Citation acc. | Refusal acc. | p95 latency | ₹/query |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 vector | | | | | | | | | | |
| E1 BM25 | | | | | | | | | | |
| E2 hybrid RRF | | | | | | | | | | |
| E3 hybrid + rerank | | | | | | | | | | |
| E5 prompt v2 | | | | | | | | | | |

Plus a second table: Hit@5 and faithfulness **by question type** for E0 vs E3. Report real numbers, even if gains are small; an honest "reranking added +4 points Hit@5 but +350 ms p95" is a better interview story than a perfect table.

## 13. Resume bullet (fill in real numbers)
> Built an automated evaluation pipeline for a hybrid-search RAG system: an 80-question hand-verified golden set, custom retrieval metrics (Hit@k, MRR, nDCG) and RAGAS generation metrics tracked as Langfuse experiments. Used it to raise Hit@5 from X% to Y% and faithfulness from A to B through hybrid retrieval, reranking and prompt changes; calibrated the LLM judge to Z% agreement with human labels; added a GitHub Actions gate that blocks PRs which regress retrieval quality.

## 14. Interview questions to prepare
- Why measure retrieval and generation separately? Which one did you fix first and why?
- Explain MRR vs nDCG vs Recall@k. When would they disagree?
- How does RAGAS compute faithfulness? What are its failure modes?
- Context precision vs context recall: what does each tell you?
- How did you build the golden set? How do you avoid it being biased toward the chunker or toward BM25?
- How do you know the judge LLM is right? What biases do LLM judges have?
- LLM metrics are noisy: how do you know an improvement is real?
- How do you stop overfitting your prompts to the golden set?
- What does your CI gate check, how long does it take, what does it cost?
- How would you evaluate this in production without labels (online evals, user feedback, sampling)?

## 15. Rules for Claude Code (working agreement)
- Work one step at a time; at the end of each, summarize what was built, how to run it, and stop for my review.
- Explain non-obvious code in 2 to 3 sentences, especially metric formulas, RAGAS internals and the CI gate.
- Write the retrieval metrics myself with Claude's help, not from a library, with unit tests.
- Unit tests never call an LLM. LLM evals are behind a pytest marker and an API key check.
- All settings (judge model, thresholds, k values, experiment configs) in YAML / `config.py`, never hardcoded. Never commit secrets.
- Never edit `golden_set.jsonl` to make a metric go up. Changes to the golden set get their own commit with a reason.
- Don't fake eval numbers; always run the script and report what it prints.
- Prefer simple, readable code over clever abstractions.

---

## Starter prompt (paste into Claude Code)

```
I'm adding the evaluation pipeline described in docs/EVAL_CLAUDE.md to my existing Hybrid Search RAG repo (BM25 + vector with RRF, cross-encoder reranker, cited answers over Indian finance/policy PDFs). I have about 1 year of GenAI experience and I'm building this as a portfolio project for a job switch, so I want to understand every part.

Please read docs/EVAL_CLAUDE.md and the existing code fully, then:
1. Tell me what in the current repo already supports evaluation and what's missing. Propose any changes to the plan (including current RAGAS and Langfuse versions and APIs) and wait for my OK.
2. Do Step 1: create src/hybrid_rag/pipeline.py with RagConfig and run_rag(question, config) -> RagResult (answer, contexts with chunk ids/pages/ranks/scores, citations, refused flag, per-stage latency, tokens), and switch the API and UI to use it. Add a script that prints the RagResult for 3 questions in all 4 retrieval modes.
3. Explain what you built, how to run it, and what we do next.

Work one step at a time and stop for my review at the end of each.
```
