# Hybrid Search RAG over Indian finance PDFs

![Python 3.11](https://img.shields.io/badge/python-3.11-blue) ![FastAPI](https://img.shields.io/badge/API-FastAPI-009688) ![Qdrant](https://img.shields.io/badge/vectors-Qdrant-dc244c) ![BM25](https://img.shields.io/badge/keyword-BM25-555) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)

Question answering over SEBI, Union Budget and IRDAI PDFs that combines **keyword search (BM25)**
with **semantic search (vectors)**, merges them with **Reciprocal Rank Fusion**, re-scores with a
**cross-encoder**, and answers with a **citation to the document and page** for every claim, or
says it couldn't find the answer. Every retrieval choice is measured on a hand-reviewed golden set.

**Live demo:** [hybrid-rag-finance.streamlit.app](https://hybrid-rag-finance.streamlit.app/) (Streamlit Community Cloud; first load after idle takes ~1 min) · **2-minute walkthrough:** _coming soon_

## The problem: why plain vector RAG struggles here

Regulatory text mixes two kinds of questions:

- **Exact terms**: "What does circular `SEBI/HO/IMD/DF2/CIR/P/2021/668` say?", "section `148A`",
  "Form `26AS`". An embedding model reads a circular number as "some SEBI circular"; bge's
  tokenizer even splits `26AS` into `26` + `##as`. In our probes, vector search put **0 of 5**
  chunks containing that circular number in its top 5. BM25 put **4 of 5**.
- **Paraphrases**: "How much will the government borrow relative to the size of the economy?" The
  Budget speech says "debt-to-GDP ratio", sharing no words with the question. Vector search
  ranked the right chunk **#1**; BM25 found nothing.

Each retriever fails where the other succeeds, so this project fuses them and measures whether
the fusion (and a reranker on top) actually helps.

## Architecture

```mermaid
flowchart LR
    PDF[SEBI / Budget / IRDAI PDFs] --> P[Parse<br/>PyMuPDF, tables to markdown,<br/>OCR fallback]
    P --> C[Clean<br/>drop Hindi lines, headers/footers,<br/>chart-label soup]
    C --> K[Chunk<br/>450 BGE tokens, sentence and<br/>table-row units, page spans]
    K --> B[(BM25 index<br/>finance-aware tokenizer)]
    K --> E[bge-small embeddings] --> Q[(Qdrant)]
    U[Question] --> B & Q
    B -- top 30 --> F[RRF fusion<br/>sum of 1/k+rank]
    Q -- top 30 --> F
    F -- top 30 --> R[Cross-encoder rerank<br/>bge-reranker-base / Cohere]
    R -- top 5 --> L[LLM with numbered sources]
    L --> V[Citation check<br/>n to file, page; drop<br/>numbers never given]
    V --> A[Answer + citations +<br/>retrieval debug + latency]
```

| Stage | Choice | Code |
|---|---|---|
| Parse | PyMuPDF with `find_tables()` to markdown, Tesseract OCR for image-only pages | [parse.py](src/hybrid_rag/ingestion/parse.py) |
| Clean | Non-ASCII-letter filter per line (catches Unicode and legacy-font Hindi), digit-masked running headers, chart-label removal | [clean.py](src/hybrid_rag/ingestion/clean.py) |
| Chunk | 450 BGE tokens, 64 overlap; sentences and table row-groups with the header row repeated; headings start chunks | [chunk.py](src/hybrid_rag/ingestion/chunk.py) |
| Dense | `BAAI/bge-small-en-v1.5` via fastembed (ONNX, no PyTorch), cosine in Qdrant | [embed.py](src/hybrid_rag/indexing/embed.py) |
| Keyword | `rank_bm25` with a tokenizer that keeps `10(23C)`, `148A(d)`, circular numbers whole | [bm25_index.py](src/hybrid_rag/indexing/bm25_index.py) |
| Fusion | Reciprocal Rank Fusion, our own 8 lines | [fusion.py](src/hybrid_rag/retrieval/fusion.py) |
| Rerank | `BAAI/bge-reranker-base` (local) or Cohere Rerank, top 30 to top 5 | [rerank.py](src/hybrid_rag/retrieval/rerank.py) |
| Answer | Ollama / OpenAI / Anthropic / Gemini; cite-by-number, validated | [answer.py](src/hybrid_rag/generation/answer.py) |

## Results

Golden set v1: **74 questions** over 27 documents (64 answerable: 16 exact-term, 18 paraphrase,
22 numeric, 8 multi-hop; plus 10 unanswerable near-misses). 46 were drafted by llama3.1 and then
reviewed against the source pages (43 needed edits, 24 more drafts were dropped); 28 were written
from the page text. **All review so far was done by Claude, not yet by a human**: see
`reviewer` in [golden_set.jsonl](eval/golden_set.jsonl). Numbers are copied from
`eval/results/` reports, never typed by hand.

### Before / after, current settings

Shipped configuration: bge-small embeddings, BM25 with the finance tokenizer, weighted RRF (k=10,
BM25 x2, chosen on the dev half), bge-reranker-base on the top 30, llama3.1 answering from the top
5 chunks. Retrieval on the 64 answerable questions
([report](eval/results/20261028-0100-v1-final-retrieval.md)); answers on all 74
([report](eval/results/20261027-1925-v1-generation.md),
[faithfulness](eval/results/20261027-1925-v1-generation-faithfulness.md)):

| Mode | Recall@5 | MRR | Faithfulness* | Citation acc. | p95 retrieval | p95 end-to-end |
|---|---|---|---|---|---|---|
| Vector only | 0.625 | 0.499 | 0.917 | 0.774 | 34 ms | 82 s |
| BM25 only | 0.859 | 0.665 | 0.915 | 0.793 | 47 ms | 65 s |
| Hybrid (weighted RRF) | 0.836 | 0.683 | **0.920** | **0.828** | 79 ms | 73 s |
| **Hybrid + rerank** | **0.867** | **0.707** | 0.916 | 0.754 | 14.3 s | 93 s |

**Headline:** hybrid + rerank lifts Recall@5 from 0.625 to 0.867 (+24 points) and MRR from 0.499
to 0.707 over vector-only RAG. Against BM25 alone, the gain is in ranking (MRR +0.04) rather than
recall (+0.008, within noise), and it costs ~14 s of CPU reranking per query. The best-cited
answers come from plain weighted hybrid, which is also ~180x faster to retrieve than reranking.
Hybrid's numbers include the dev half its fusion weights were tuned on; on the untouched test
half alone (below) it trails BM25's Recall@5 slightly.

\* Faithfulness = share of an answer's claims supported by the context it was given, judged by
llama3.1 grading its own answers (no independent judge key was available). 219 of 233 answered
questions were scored; 102 of 836 claims (12%) were flagged unsupported, and the flagged ones I
spot-checked were genuine (e.g. an answer attributing a section 478 penalty to section 276C). The
score is flat across modes, which most likely means a self-grading 8B judge isn't sensitive
enough to separate them, not that the modes are equally faithful. Re-run `eval/judge.py` with
`JUDGE_PROVIDER`/`JUDGE_MODEL` set to a stronger model for a trustworthy comparison.

### Retrieval detail with the plan's default fusion (RRF k=60, equal weights)

Retrieval, 64 answerable questions, top 10 ([report](eval/results/20261026-1705-v1-retrieval.md)):

| Mode | Recall@5 | Recall@10 | MRR | nDCG@10 | p95 retrieval |
|---|---|---|---|---|---|
| Vector only (bge-small) | 0.625 | 0.750 | 0.499 | 0.551 | 42 ms |
| BM25 only | 0.859 | 0.914 | 0.665 | 0.721 | 66 ms |
| Hybrid (RRF, k=60) | 0.789 | 0.844 | 0.631 | 0.679 | 116 ms |
| **Hybrid + rerank** (bge-reranker-base, top 30) | **0.867** | **0.922** | **0.708** | **0.748** | 21.9 s (laptop CPU) |

Recall@5 by question type:

| Mode | exact_term (16) | paraphrase (18) | numeric (22) | multi_hop (8) |
|---|---|---|---|---|
| Vector only | 0.750 | 0.500 | 0.727 | 0.375 |
| BM25 only | **1.000** | 0.722 | **0.955** | **0.625** |
| Hybrid (RRF, k=60) | 0.875 | 0.667 | 0.909 | 0.562 |
| Hybrid + rerank | **1.000** | **0.889** | 0.864 | 0.562 |

**What this shows, honestly:** the cross-encoder is what makes hybrid search pay off. Plain RRF
fusion lands *below* BM25, and reranking its top 30 lifts the result past BM25: about the same
Recall@5 (0.867 vs 0.859, within noise) but clearly better ranking (MRR 0.708 vs 0.665), driven
by paraphrase questions (Recall@5 0.889 vs 0.722). It costs on numeric questions (0.864 vs
0.955: the reranker demotes table chunks) and about 20 s of CPU per query. On their own, BM25 beats dense retrieval on
every question type, including paraphrases, and RRF fusion with the weaker dense list pulls BM25
*down*. The RRF sweep (k 5 to 60 x depth 10 to 30) improves hybrid only to Recall@5 0.820 (k=5,
depth 20), still below BM25. Why the dense side is weak (from the per-question CSVs): for some
Budget questions its top hits are number tables from Budget at a Glance and the Receipt Budget
rather than the speech paragraph with the answer. Caveats: 64 questions means roughly +/-5 points
of noise per figure, and questions drafted from passages share wording with them, which favours
BM25.

**Testing that diagnosis** ([experiment reports and scripts](eval/results/experiments/), dev
half): two fixes it suggested both failed. Keeping number-only chunks (letter ratio < 0.3 or < 0.4,
4-8% of chunks, mostly Receipt Budget tables) out of the vector index changed no dev metric at all.
Embedding chunks without the document title (section heading + text only) made things worse:
vector Recall@5 0.688 -> 0.625, hybrid 0.844 -> 0.828, MRR 0.720 -> 0.689. The title prefix costs
a few Budget questions but anchors many more, so it stays. Neither change was adopted.

### Answers (generation), all 74 questions

llama3.1 (local, via Ollama) answers each question from the top 5 chunks of each mode
([report](eval/results/20261027-1925-v1-generation.md)). Citation accuracy = share of answered,
answerable questions with at least one citation on a labelled page; refusal accuracy = share of the
10 unanswerable questions refused; false refusals = answerable questions it refused.

| Mode | Citation acc. | Refusal acc. | False refusals | p50 / p95 total latency |
|---|---|---|---|---|
| Vector only | 0.774 | **1.000** | 0.172 | 41 s / 82 s |
| BM25 only | 0.793 | 0.900 | 0.094 | 41 s / 65 s |
| **Hybrid (weighted RRF)** | **0.828** | 0.900 | 0.094 | 41 s / 73 s |
| Hybrid + rerank | 0.754 | 0.900 | **0.047** | 56 s / 93 s |

Hybrid gives the best-cited answers. Reranking gives the LLM better context, so it refuses answerable
questions least often (4.7%), but it answers more of the hard ones, and those carry more citation
misses. Vector-only refuses every unanswerable question but also 17% of answerable ones: its weaker
context makes the model cautious. Latency is dominated by the 8B model on a laptop GPU (~40 s); the
reranker adds ~15 s. Faithfulness (claims supported by the answer's own context,
[judge.py](eval/judge.py)) is in the table above; note it measures grounding, not correctness: an
answer can faithfully repeat a wrong figure from a wrongly retrieved chunk (the judge marked a
vector-mode answer of "20% of NAV" as supported because a chunk in its context said 20% about a
different limit; the right answer is 10%).

### Tuning fusion without fooling ourselves

The table above uses the plan's default fusion (RRF k=60, both lists weighted equally), which
lands below BM25. To tune it honestly, the golden set is split 50/50 into **dev** and **test**,
stratified by question type (`run_eval.py --split`). Settings were chosen on dev only
([dev sweep](eval/results/20261026-1732-v1-dev-sweep.md): 48 combinations of k, list depth and a
BM25 weight) and then measured once on the untouched test half
([test report](eval/results/20261026-1732-v1-test.md),
[old-setting baseline](eval/results/20261026-1734-v1-test-equal-rrf.md)):

| Test half, 32 answerable questions | Recall@5 | MRR |
|---|---|---|
| Vector only | 0.562 | 0.498 |
| BM25 only | **0.875** | 0.633 |
| Hybrid, RRF k=60, equal weights (plan default) | 0.766 | 0.610 |
| Hybrid, weighted RRF k=10, BM25 x2 (chosen on dev, now the default) | 0.828 | 0.647 |
| Hybrid + rerank (on the weighted fusion) | **0.875** | **0.677** |

Weighting BM25's ranks 2x recovered most of what equal-weight fusion lost (+6 points Recall@5 on
unseen questions) and gives slightly better ranking than BM25, but BM25 alone still has the best
plain recall; on dev the two were tied, so part of that gain was dev noise (at 32 questions each
figure moves about +/-6 points). The cross-encoder matches BM25's recall with the best ranking and
got every test paraphrase question right, at ~16 s per query on a laptop CPU.

**How it's measured** ([metrics.py](eval/metrics.py)): relevance is labelled as `(document,
pages)`, not chunk ids, so the same golden set scores any chunk size. A retrieved chunk counts if
its page span overlaps a labelled page. Recall@k = share of required passages in the top k (two
for multi-hop questions); MRR = 1/rank of the first hit; nDCG@10 credits each passage once at the
rank it first appears. Generation adds citation accuracy, refusal accuracy on unanswerable
questions, false refusals, and p50/p95 latency per stage. Faithfulness ([judge.py](eval/judge.py))
has an LLM split each answer into claims and check every claim against the context that answer
was given: the RAGAS idea without its LangChain dependency tree, with judge outputs cached.

## Design decisions and trade-offs

- **RRF, not weighted score sums.** BM25 scores run 0 to 30+ and vary with query length; cosine
  similarities here sit between about 0.6 and 0.8. Ranks are comparable by construction, so RRF
  needs no per-query normalisation. Caveat we hit: with `k = 60` and 30-deep lists, a chunk ranked
  26th and 15th (score 0.025) outranks one ranked 1st by a single retriever (0.016). On the
  circular-number probe, BM25's correct ranks 1 to 5 became RRF ranks 10 to 17. `eval/run_eval.py
  --sweep` tests `k` and fusion depth on the golden set instead of trusting the default.
- **Cross-encoder reranking on 30 candidates.** The embedder encodes question and chunk
  separately (fast, approximate). The cross-encoder reads them together (slow, precise). On our
  probes it restored the circular-number chunks to ranks 1 to 5 and found the right chunk first on
  4 of 5 queries. Cost: about 15 s per query for 30 chunks of ~470 tokens on a laptop CPU, far
  from the "under a second" often quoted for short passages. `jina-reranker-v1-turbo-en` is ~4x
  faster but weaker on exact terms. Cohere Rerank removes the cost for the hosted demo.
- **450-token chunks, not 512 or 1024.** bge models truncate at 512 tokens, so a 1024-token chunk
  is mostly never embedded. 450 leaves room for the "title / section" prefix each chunk is
  embedded with.
- **Cite by number, then validate.** The LLM sees sources `[1]` to `[5]` and cites numbers;
  code maps them to `[file.pdf, p.X]` and drops any number it never received. Asking a model to
  reproduce file names invites typos, while a number is easy to emit and to check.
- **No LangChain / LlamaIndex.** Retrieval, fusion, prompting and citation checks are plain
  Python so every step can be inspected and tested.
- **Embedded Qdrant + fastembed.** The index is ~25 MB and the models run on ONNX, so the demo is
  one container with no external database and no PyTorch. `QDRANT_URL` switches to a Qdrant
  server or Qdrant Cloud without code changes.

## Quickstart

```bash
git clone <this repo> && cd hybrid-search-rag
uv sync
cp .env.example .env                              # defaults: embedded Qdrant + local Ollama
uv run python scripts/download_data.py            # PDFs listed in data/sources.csv
uv run python -m hybrid_rag.ingestion.pipeline    # PDFs -> pages.jsonl -> chunks.jsonl
uv run python -m hybrid_rag.indexing.build        # BM25 (~1 s) + vectors (~7 min on CPU)
ollama pull llama3.1                              # or set LLM_PROVIDER / LLM_MODEL / key in .env
uv run python -m hybrid_rag.cli "What is the debt-to-GDP ratio estimated for 2026-27?"
```

`--mode vector|bm25|hybrid|hybrid_rerank` picks the retriever; `--retrieve-only` prints the
ranking table without calling the LLM. RBI's site serves a bot challenge to scripts, so its three
PDFs must be saved by hand into `data/raw/` (the download script tells you which).

Make targets: `make data ingest index ask Q="..." serve ui eval eval-full test lint`.

## API

```bash
uv run uvicorn hybrid_rag.api.main:app --port 8000      # docs at http://localhost:8000/docs
uv run streamlit run ui/app.py                          # UI at http://localhost:8501
```

```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" -d '{"question": "What is the TCS rate on remittances under the Liberalised Remittance Scheme for education?", "mode": "hybrid"}'
```

A real response (snippet shortened):

```json
{
  "answer": "According to [budget_2026_27_memorandum_finance_bill.pdf, p.50], the TCS rate on remittances made under the Liberalised Remittance Scheme for education is proposed to be reduced to 2% if the remittance amount exceeds ten lakh rupees.",
  "refused": false,
  "citations": [
    {"doc": "budget_2026_27_memorandum_finance_bill.pdf", "page": "p.50",
     "chunk_id": "budget_2026_27_memorandum_finance_bill:0069", "snippet": "..."}
  ],
  "invalid_citations": [],
  "uncited": false,
  "retrieval_debug": {"mode": "hybrid", "chunks": ["... vector_rank, bm25_rank, rrf_rank, rerank_score per context chunk"]},
  "latency_ms": {"retrieve": 23, "generate": 54433, "total": 54455}
}
```

| Endpoint | What it does |
|---|---|
| `POST /query` | Question + mode -> cited answer. 422 on bad input, 429 over the per-IP rate limit, 503 if the LLM is unreachable |
| `GET /health` | Indexed chunk count, configured LLM and reranker |
| `POST /ingest` | Rebuild everything from `data/raw` in the background. Needs `X-Admin-Token`; disabled unless `ADMIN_TOKEN` is set |

Every query logs one JSON line with per-stage latency, citation count and refusal.

Embedded Qdrant allows one process at a time: stop the API before running the CLI or eval, or
point everything at a Qdrant server with `QDRANT_URL`.

## Evaluation workflow

```bash
uv run python scripts/build_golden_set.py            # LLM drafts questions per type
uv run python scripts/review_golden_set.py           # keep / edit / drop each one (resumable)
uv run python scripts/review_golden_set.py --add     # write ~30% of questions by hand
uv run python scripts/review_golden_set.py --check   # validate labels against the corpus
uv run python eval/run_eval.py --sweep               # retrieval, all 4 modes + RRF sweep
uv run python eval/run_eval.py --generate            # + answers: citation / refusal accuracy
uv run python eval/judge.py eval/results/<run>-generation.csv   # faithfulness (LLM judge)
uv run python eval/run_eval.py --split dev --sweep  # tune on dev; then --split test to report
```

Drafting guards against weak questions (numeric answers must contain a number, exact-term
questions must contain an identifier, tables of contents and blank forms are skipped), but every
question is still reviewed by hand. Hand-written questions limit the bias of questions generated
from our own chunks.

## Docker and deployment

```bash
docker compose up --build app      # UI http://localhost:7860, API http://localhost:8000/docs
uv run python scripts/deploy_space.py --space <you>/hybrid-rag --dry-run    # what would ship
uv run python scripts/deploy_space.py --space <you>/hybrid-rag --var LLM_PROVIDER=gemini --var LLM_MODEL=<model>
```

One image runs the API and the Streamlit UI (`scripts/start.sh`); the embedding and reranker
models are downloaded at build time, and the prebuilt indexes are copied in, so the container
never parses PDFs. LLM API keys go in the Space's secrets, never in the repo. (Docker Spaces
now need a Hugging Face PRO plan, so the public demo runs on Streamlit Community Cloud instead.)

**Streamlit Community Cloud (free, the live demo).** With `API_URL` empty the UI runs retrieval
and generation in its own process, so no FastAPI server is needed. The prebuilt indexes
(`chunks.jsonl`, `bm25.pkl`, the embedded Qdrant store, ~66 MB) are committed for this, and
Community Cloud installs dependencies from `uv.lock`. Deploy from share.streamlit.io: repo
`main`, entrypoint `ui/app.py`, Python 3.11, and these secrets (root-level secrets become
environment variables, which `config.py` reads):

```toml
API_URL = ""
LLM_PROVIDER = "gemini"
LLM_MODEL = "gemini-3.6-flash"
GEMINI_API_KEY = "..."
RERANKER_MODEL = "jinaai/jina-reranker-v1-turbo-en"  # ~4x faster, smaller; fits shared CPU/RAM
RERANK_CANDIDATES = "10"
```

## Failure cases found so far

- **RRF with k = 60 demoted single-retriever wins** (see design decisions); the eval sweep decides.
- **A valid citation on an unsupported claim**: llama3.1 called a circular "rescinded", citing a
  page that only lists it in a footnote. Citation checks can't catch this; faithfulness
  evaluation (RAGAS) can.
- **Charts lose their numbers.** Bar-chart labels in Budget at a Glance extract as unlabelled
  number soup, so they're dropped; values that exist only in charts are unanswerable.
- **LLM-drafted "paraphrase" questions reuse the passage's words**, which flatters BM25. Review
  rewrites them and hand-written questions balance the set.
- **Some table headers are junk**: a page title split across cells becomes the "header" repeated
  on each table piece.

## What's next

Experiments still open (tune on `--split dev`, report on `--split test`): a non-quantized or
larger embedder (bge-base), since the dense side is the weak link; rerank fewer candidates or use
Cohere to cut the ~16 s reranking cost; a bigger golden set so the test half is less noisy. Then:
query rewriting and HyDE for short questions; RAGAS faithfulness with a stronger judge model;
reading charts with a vision model; a fine-tuned or long-context embedder (bge-m3) to test larger
chunks; Hindi documents with a multilingual embedder; Qdrant sparse vectors to keep both indexes
in one database.

## Tech stack

Python 3.11 · uv · PyMuPDF · fastembed (bge-small-en-v1.5, bge-reranker-base) · rank_bm25 +
PyStemmer · Qdrant · Ollama / OpenAI / Anthropic / Gemini · FastAPI · Streamlit · Docker ·
pytest + ruff + GitHub Actions
