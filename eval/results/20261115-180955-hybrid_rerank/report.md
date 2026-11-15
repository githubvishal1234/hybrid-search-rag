# hybrid_rerank (20261115-180955-hybrid_rerank)

Hypothesis: Higher precision at the top (Hit@1, MRR) because the cross-encoder reads question and passage together; costs ~15 s per query on CPU.

- split `dev`: 49 questions, 43 answerable scored
- git `a37db1b-dirty`, golden set `golden_set.jsonl` (sha1 70b416d2d43f)
- config: `{"mode": "hybrid_rerank", "dense_top_k": 30, "bm25_top_k": 30, "rrf_k": 10, "rrf_bm25_weight": 2.0, "rerank_candidates": 30, "reranker_provider": "local", "reranker_model": "BAAI/bge-reranker-base", "context_top_n": 5}` (all resolved settings in config.json)

## Overall

| hit@1 | hit@3 | hit@5 | hit@10 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| 0.721 | 0.837 | 0.907 | 0.930 | 0.872 | 0.919 | 0.793 | 0.808 |

## By question type and difficulty

| group | n | hit@5 | recall@5 | mrr | ndcg@10 |
|---|---|---|---|---|---|
| exact_term | 12 | 1.000 | 1.000 | 0.958 | 0.969 |
| paraphrase | 12 | 0.833 | 0.833 | 0.628 | 0.679 |
| numeric | 12 | 0.917 | 0.917 | 0.863 | 0.894 |
| multi_hop | 7 | 0.857 | 0.643 | 0.671 | 0.607 |
| easy | 15 | 0.933 | 0.933 | 0.941 | 0.953 |
| medium | 19 | 0.947 | 0.921 | 0.794 | 0.812 |
| hard | 9 | 0.778 | 0.667 | 0.544 | 0.558 |

## Latency (ms)

| stage | p50 | p95 |
|---|---|---|
| bm25 | 41 | 73 |
| dense | 31 | 43 |
| fusion | 0 | 0 |
| rerank | 14956 | 17202 |
| retrieve | 15043 | 17323 |
| total | 15043 | 17323 |
