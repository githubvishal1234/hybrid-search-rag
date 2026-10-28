# Eval run 2026-10-28 01:16

- golden set: `golden_set.jsonl` (split: all): 74 questions (64 answerable)
- embedding `BAAI/bge-small-en-v1.5`, chunk_size 450, rrf_k 10, BM25 weight 2.0, depth 30/30, reranker `local:BAAI/bge-reranker-base` on top 30

## Retrieval (answerable questions, top 10)

| Mode | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| vector | 0.625 | 0.750 | 0.499 | 0.551 | 26 | 34 |
| bm25 | 0.859 | 0.914 | 0.665 | 0.721 | 33 | 47 |
| hybrid | 0.836 | 0.883 | 0.683 | 0.727 | 60 | 79 |
| hybrid_rerank | 0.867 | 0.922 | 0.707 | 0.748 | 13956 | 14310 |

## Recall@5 by question type

| Mode | exact_term (n=16) | paraphrase (n=18) | numeric (n=22) | multi_hop (n=8) |
|---|---|---|---|---|
| vector | 0.750 | 0.500 | 0.727 | 0.375 |
| bm25 | 1.000 | 0.722 | 0.955 | 0.625 |
| hybrid | 0.938 | 0.722 | 0.955 | 0.562 |
| hybrid_rerank | 1.000 | 0.889 | 0.864 | 0.562 |
