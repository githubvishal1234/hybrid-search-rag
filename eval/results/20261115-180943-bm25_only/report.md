# bm25_only (20261115-180943-bm25_only)

Hypothesis: Better than vector on exact_term and numeric (identifiers and figures match literally), worse on paraphrase.

- split `dev`: 49 questions, 43 answerable scored
- git `a37db1b-dirty`, golden set `golden_set.jsonl` (sha1 70b416d2d43f)
- config: `{"mode": "bm25", "bm25_top_k": 30, "context_top_n": 5}` (all resolved settings in config.json)

## Overall

| hit@1 | hit@3 | hit@5 | hit@10 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| 0.535 | 0.791 | 0.837 | 0.930 | 0.814 | 0.930 | 0.681 | 0.737 |

## By question type and difficulty

| group | n | hit@5 | recall@5 | mrr | ndcg@10 |
|---|---|---|---|---|---|
| exact_term | 12 | 1.000 | 1.000 | 0.875 | 0.908 |
| paraphrase | 12 | 0.583 | 0.583 | 0.384 | 0.491 |
| numeric | 12 | 1.000 | 1.000 | 0.861 | 0.897 |
| multi_hop | 7 | 0.714 | 0.571 | 0.546 | 0.594 |
| easy | 15 | 0.933 | 0.933 | 0.831 | 0.872 |
| medium | 19 | 0.842 | 0.816 | 0.628 | 0.708 |
| hard | 9 | 0.667 | 0.611 | 0.542 | 0.575 |

## Latency (ms)

| stage | p50 | p95 |
|---|---|---|
| bm25 | 46 | 82 |
| retrieve | 46 | 82 |
| total | 46 | 82 |
