# hybrid_weighted (20261115-180950-hybrid_weighted)

Hypothesis: A smaller k lets a top-ranked chunk in one list win, and the extra BM25 weight trusts the stronger retriever on this corpus; beats E2.

- split `dev`: 49 questions, 43 answerable scored
- git `a37db1b-dirty`, golden set `golden_set.jsonl` (sha1 70b416d2d43f)
- config: `{"mode": "hybrid", "dense_top_k": 30, "bm25_top_k": 30, "rrf_k": 10, "rrf_bm25_weight": 2.0, "context_top_n": 5}` (all resolved settings in config.json)

## Overall

| hit@1 | hit@3 | hit@5 | hit@10 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| 0.581 | 0.791 | 0.884 | 0.907 | 0.860 | 0.907 | 0.703 | 0.752 |

## By question type and difficulty

| group | n | hit@5 | recall@5 | mrr | ndcg@10 |
|---|---|---|---|---|---|
| exact_term | 12 | 1.000 | 1.000 | 0.896 | 0.922 |
| paraphrase | 12 | 0.750 | 0.750 | 0.461 | 0.552 |
| numeric | 12 | 1.000 | 1.000 | 0.875 | 0.908 |
| multi_hop | 7 | 0.714 | 0.571 | 0.493 | 0.537 |
| easy | 15 | 1.000 | 1.000 | 0.883 | 0.913 |
| medium | 19 | 0.895 | 0.868 | 0.655 | 0.728 |
| hard | 9 | 0.667 | 0.611 | 0.504 | 0.534 |

## Latency (ms)

| stage | p50 | p95 |
|---|---|---|
| bm25 | 51 | 91 |
| dense | 36 | 45 |
| fusion | 0 | 0 |
| retrieve | 86 | 137 |
| total | 86 | 137 |
