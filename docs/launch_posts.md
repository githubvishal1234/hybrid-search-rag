# Resume bullets and LinkedIn post

Every number below comes from `eval/results/20261028-0100-v1-final-retrieval.md` and
`eval/results/20261027-1925-v1-generation.md`. Don't round them up. Don't write "hand-verified"
until the golden-set spot-check is done; faithfulness is left out on purpose (self-graded, flat
across modes).

## Resume

**Project title line**

Hybrid Search RAG over Indian Finance Regulations | Python, Qdrant, BM25, FastAPI, Streamlit, Docker
Live: hybrid-rag-finance.streamlit.app · Code: github.com/githubvishal1234/hybrid-search-rag

**Bullets (pick 3-4)**

- Built a question-answering system over 27 Indian regulatory PDFs (SEBI, Union Budget 2026-27,
  IRDAI, PRS income-tax summary; 6,100 chunks) that combines BM25 keyword search and dense vector search with
  weighted Reciprocal Rank Fusion and a cross-encoder reranker.
- Raised Recall@5 from 62.5% (vector-only) to 86.7% and MRR from 0.50 to 0.71 on a 74-question
  golden set, tuning fusion on a dev split and reporting on a held-out test split to avoid
  overfitting the eval.
- Wrote a finance-aware BM25 tokenizer (keeps identifiers like "10(23C)", "114B", "26AS" intact,
  normalises Indian number formats) and a PDF pipeline that converts tables to markdown and strips
  bilingual Hindi text, repeated headers and chart noise.
- Generated answers with page-level citations and a post-check that drops citations to sources
  the model was never given; measured citation accuracy (0.83 for hybrid), refusal accuracy on
  unanswerable questions (0.90) and p95 latency per stage.
- Shipped as a Dockerized FastAPI service (rate limiting, admin-only re-ingest, structured
  per-stage latency logs) with a Streamlit UI and GitHub Actions CI; deployed free on Streamlit
  Community Cloud with Gemini.

**One-liner (if space is tight)**

Hybrid BM25 + vector RAG with RRF and reranking over 27 Indian regulatory PDFs; Recall@5 62.5% ->
86.7% vs vector-only on a held-out eval split; cited answers, FastAPI + Docker, live demo.

## LinkedIn post

> Most "chat with your PDF" demos use only vector search. I wanted to know how much that leaves
> on the table, so I measured it.
>
> I built a RAG system over 27 Indian finance documents (SEBI circulars, the Union Budget
> 2026-27, IRDAI regulations, an income-tax bill summary) and wrote a 74-question test set: exact-term questions,
> paraphrases, numbers from tables, multi-document questions, and questions the documents
> can't answer.
>
> What I found:
> 🔹 Vector-only search found the right page in the top 5 only 62.5% of the time. Legal text is
> full of identifiers like "Section 148A" or "Rule 114B", and embeddings blur them.
> 🔹 Plain BM25 keyword search hit 85.9%. A strong baseline that's easy to skip.
> 🔹 Textbook Reciprocal Rank Fusion (k=60) actually made things worse than BM25 alone. Tuning
> it on a dev split (k=10, BM25 weighted 2x) and checking on a held-out split fixed most of that.
> 🔹 Hybrid + a cross-encoder reranker reached 86.7% Recall@5 and the best ranking quality
> (MRR 0.71 vs 0.50 for vector-only).
>
> Every answer cites the document and page, and citations to sources the model never saw are
> dropped automatically. When the documents don't cover a question, it says so.
>
> Honest caveats: the gain over BM25 alone is small on a test set this size, and the reranker
> costs seconds per query on CPU. Knowing that is half the point of measuring.
>
> Stack: Python, PyMuPDF, Qdrant, rank-bm25, fastembed (bge-small, bge-reranker), FastAPI,
> Streamlit, Docker, GitHub Actions. Runs free on Streamlit Community Cloud with Gemini.
>
> Try it: https://hybrid-rag-finance.streamlit.app/
> Code and full results: https://github.com/githubvishal1234/hybrid-search-rag
>
> #RAG #GenAI #LLM #InformationRetrieval #Python #MachineLearning
