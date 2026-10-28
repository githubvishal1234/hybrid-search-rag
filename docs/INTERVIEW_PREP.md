# Interview prep

Answers to the questions this project invites, using this project's own numbers and failures.
Say them in your own words; the specifics are what make them convincing. Numbers are from
`eval/results/20261028-0100-v1-final-retrieval.md` unless noted.

## The 30-second pitch

"I built a question-answering system over 27 Indian regulatory PDFs: SEBI circulars, the Union
Budget, IRDAI regulations. It combines BM25 keyword search with vector search, fuses them with
Reciprocal Rank Fusion, reranks with a cross-encoder, and answers with page-level citations or
refuses. The part I'm proudest of is the evaluation: a 74-question golden set split into dev and
test halves. Vector-only search found the right page in the top 5 only 62.5% of the time; the
final pipeline gets 86.7%. And I found that the textbook RRF setting made results worse than BM25
alone, which I fixed by tuning on the dev half only."

---

## Retrieval

**Why does BM25 beat embeddings on some queries? When does it lose?**
BM25 scores exact token overlap, weighted by how rare a term is. Legal text is full of rare,
exact identifiers: "section 148A", "Rule 114B", "10(23C)", circular numbers. A query containing
one of them has a near-unique match, and BM25 ranks it first. A small embedding model compresses
the passage into 384 numbers; "148A" and "148B" end up almost identical, so the right chunk drifts
down. Here BM25 scored 0.859 Recall@5 vs 0.625 for vectors.
It loses when the user's words differ from the document's ("how much will the government
borrow" vs "fiscal deficit... per cent of GDP"), on synonyms, and on short, vague queries. I also
built a tokenizer so BM25 keeps "10(23C)" whole and also indexes its parts, and normalises
"50,00,000" and "per cent".
Honest twist: on my set BM25 even won on "paraphrase" questions, because LLM-drafted paraphrases
reused the document's wording. That's why ~30% of questions were written from the PDFs by hand.

**Why did vectors do so badly here?**
Two diagnosed causes: (1) for Budget questions, the dense top hits were number tables that look
alike to the embedder; (2) every chunk is embedded with its document title, so chunks from the
same document look similar. I tested two fixes on the dev half and rejected both: excluding
number-only chunks changed nothing; dropping the title prefix made things worse (0.688 -> 0.625).
Next step would be a bigger or fine-tuned embedder (bge-base, bge-m3).

**How does RRF work, and why not just add normalized scores?**
Each retriever returns a ranked list. Each document gets `sum over lists of w / (k + rank)`.
Only ranks are used, never raw scores. That matters because BM25 scores are unbounded (5, 30, 80
depending on the query) while cosine similarities sit in a narrow band like 0.6-0.8. Normalising
them (min-max per query) is fragile: one outlier stretches the scale, and the "best" score is
always 1.0 even when the whole list is bad. Ranks are comparable across retrievers with no
calibration. `k` controls how much the top ranks dominate: a small k rewards being #1, a large k
flattens the curve.

**You said default RRF made things worse. Explain.**
With k=60 and 30-deep lists, a chunk ranked mediocre in both lists (say 26th and 15th) outscores a
chunk ranked 1st in only one list. In legal text BM25's #1 is often exactly right and the vector
retriever doesn't find it at all, so RRF buried it. Hybrid at k=60 scored 0.789 Recall@5, below
BM25 alone (0.859). I swept k, fusion depth and a BM25 weight on the dev half only, chose k=10 with
BM25 weighted 2x, and reported on the untouched test half: 0.766 -> 0.828 Recall@5, MRR 0.610 ->
0.647.

**Bi-encoder vs cross-encoder: why rerank only the top 30?**
A bi-encoder embeds the query and each passage separately, so passage vectors are computed once
and search is a fast nearest-neighbour lookup over millions. It never sees query and passage
together, so it misses fine interactions. A cross-encoder reads the query and passage together
through a transformer and outputs one relevance score: much more accurate, but one full forward
pass per pair at query time. You can't do that for 6,000 chunks per query, so the cheap retrievers
pick 30 candidates and the cross-encoder reorders them.
Measured cost: bge-reranker-base took ~15 s for 30 pairs of ~470 tokens on my laptop CPU, not the
sub-second numbers quoted for short passages. Gain: Recall@5 0.836 -> 0.867, MRR 0.683 -> 0.707.
For the free live demo I used a smaller reranker (jina turbo) and 10 candidates.

**How did you pick chunk size? What happens with tables?**
bge models only read 512 tokens; anything after is silently truncated. I used 450 tokens to leave
room for a "title / section" prefix, with 64 tokens of overlap so a sentence cut at a boundary
appears in both chunks. Units are whole sentences, and headings close a chunk so one chunk doesn't
mix two sections. Tables are extracted as markdown with PyMuPDF and split into row groups with the
header row repeated in every piece, so numbers stay next to their column names. Relevance is
labelled by (document, page), not chunk id, so chunk-size experiments stay comparable.

---

## Evaluation

**How did you build and validate the golden set? How do you avoid leakage?**
An LLM drafted questions from sampled chunks with per-type prompts; drafts were auto-rejected
for obvious problems (numeric answers without digits, exact_term questions without an
identifier, table-of-contents chunks). Every kept draft was reviewed against the source page:
46 kept (43 edited), 24 dropped, and 28 more written straight from the PDFs. Five types:
exact_term, paraphrase, numeric, multi_hop, unanswerable (near-misses that are on-topic but not
answered).
Leakage controls: hand-written questions not derived from my chunks; page-level labels; a
dev/test split stratified by type, where all tuning happens on dev and test is used once.
Be upfront: the review was done with an AI assistant; a human spot-check of a sample is the
remaining step before calling it hand-verified.

**What metrics, and why those?**
Recall@5: is a correct page in what the LLM will see (it gets 5 chunks)? MRR: how high is the first
correct hit? nDCG@10: rewards correct hits near the top, crediting each required passage once (so
multi-hop questions need both passages). For answers: citation accuracy (cited page is a correct
page), refusal accuracy on unanswerable questions (0.90 for hybrid), false refusals, latency p50/p95
per stage.

**What about faithfulness?**
I wrote my own claim-level judge (split the answer into claims, check each against the sources)
instead of pulling in RAGAS and its LangChain dependencies. With llama3.1 grading its own answers it
scored ~0.92 in every mode, which tells me the judge isn't discriminating, not that all modes are
equal. The fix is a stronger, independent judge model; the code supports a separate judge provider.
Also: faithfulness measures grounding, not correctness. An answer can faithfully repeat a wrong
chunk.

**Your gain over BM25 is tiny. Is the system worth it?**
On Recall@5, hybrid + rerank beats BM25 by under one point, which is within noise on 64 answerable
questions. The real gains are ranking quality (MRR 0.707 vs 0.665) and robustness to paraphrase.
The bigger lesson: on identifier-heavy text, BM25 is a strong baseline that many RAG systems skip.
If latency mattered most I'd ship weighted hybrid without the reranker (79 ms p95).

---

## Generation and safety

**How do you stop hallucinated citations?**
Sources go into the prompt numbered [1]..[5]. The model cites numbers, not file names, which it
can't mistype. After generation, code maps each number to `[file, p.X]` and drops any number that
wasn't in the context; the UI shows a warning for dropped ones. Answers with no citation are
flagged as unverified. Limitation: this proves the citation points at real context, not that the
cited text supports the claim. I saw llama3.1 add "rescinded" with a valid citation; only a
faithfulness check catches that.

**How does it refuse?**
The prompt says to answer only from the sources and otherwise reply with a fixed sentence; the
code detects that sentence. Unanswerable questions in the eval are near-misses, so refusing them
requires actually reading the context. Refusal accuracy 0.90; false refusals 0.047-0.17 depending
on mode, the main cost of being strict.

---

## Engineering

**Walk me through a request.**
Query -> BM25 top 30 and vector top 30 (Qdrant, cosine) -> weighted RRF -> top 30 to the
cross-encoder -> top 5 chunks -> prompt with numbered sources -> LLM -> citation mapping and
validation -> JSON response with answer, citations, per-retriever ranks, rerank scores and
per-stage latency. FastAPI logs one JSON line per query.

**Why no LangChain or LlamaIndex?**
The pipeline is a few hundred lines; owning it means I can explain and test every step (RRF,
citation checks) and swap providers with one config value. Frameworks are worth it for many
integrations; here they would hide the parts I wanted to show.

**How would you scale to 1 million documents? Handle updates?**
- Ingestion as a queue of jobs, parallel workers; it's already idempotent (file hash) so retries
  are safe.
- Vectors: a Qdrant cluster with HNSW and quantization (int8 cuts memory ~4x). I use exact search
  today because 6,000 vectors is tiny.
- BM25: move from in-memory rank_bm25 to Qdrant sparse vectors, Elasticsearch or OpenSearch.
- Reranker on GPU, or a hosted reranker; cache frequent queries.
- Metadata filters (publisher, year) to shrink the search space.
- Updates: re-ingest changed files by hash, upsert their chunks, delete stale chunk ids (my sync
  already does this), and version documents so amended circulars don't conflict with old ones.

**What would you do next?**
Stronger independent faithfulness judge; bigger or fine-tuned embedder for paraphrases; query
rewriting or HyDE for vague questions; multilingual (Hindi) with bge-m3; human-verified larger
golden set; OCR for the ~20 image-only pages.

**What went wrong during the build?**
- Default RRF lost to BM25 (diagnosed and fixed with a dev/test protocol).
- A stale-points bug: re-indexing left old chunks in Qdrant; fixed with an upsert-then-delete sync.
- Ollama with an 8k context grew to 14.6 GB and crashed a multi-hour eval with MemoryError; made
  the eval resumable and used a 4k context.
- Merged table cells were duplicated across columns by the PDF parser; added de-duplication.
- Hugging Face started charging for Docker Spaces, so I moved the demo to Streamlit Cloud by
  letting the UI run the pipeline in-process.
