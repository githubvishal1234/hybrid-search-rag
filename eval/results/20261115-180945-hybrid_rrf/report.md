# hybrid_rrf (20261115-180945-hybrid_rrf)

Hypothesis: Best of both retrievers. Risk seen on v1 probes - with k=60 a chunk ranked mediocre in both lists outscores one ranked first in only one.

- split `dev`: 49 questions, 43 answerable scored
- git `a37db1b-dirty`, golden set `golden_set.jsonl` (sha1 70b416d2d43f)
- config: `{"mode": "hybrid", "dense_top_k": 30, "bm25_top_k": 30, "rrf_k": 60, "rrf_bm25_weight": 1.0, "context_top_n": 5}` (all resolved settings in config.json)

## Overall

| hit@1 | hit@3 | hit@5 | hit@10 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| 0.488 | 0.744 | 0.837 | 0.907 | 0.814 | 0.884 | 0.635 | 0.691 |

## By question type and difficulty

| group | n | hit@5 | recall@5 | mrr | ndcg@10 |
|---|---|---|---|---|---|
| exact_term | 12 | 0.917 | 0.917 | 0.762 | 0.821 |
| paraphrase | 12 | 0.667 | 0.667 | 0.458 | 0.528 |
| numeric | 12 | 1.000 | 1.000 | 0.787 | 0.840 |
| multi_hop | 7 | 0.714 | 0.571 | 0.458 | 0.491 |
| easy | 15 | 0.933 | 0.933 | 0.810 | 0.857 |
| medium | 19 | 0.842 | 0.816 | 0.571 | 0.651 |
| hard | 9 | 0.667 | 0.611 | 0.477 | 0.499 |

## Latency (ms)

| stage | p50 | p95 |
|---|---|---|
| bm25 | 53 | 78 |
| dense | 37 | 48 |
| fusion | 0 | 0 |
| retrieve | 90 | 124 |
| total | 90 | 124 |
