# baseline_vector (20261115-184547-baseline_vector)

Hypothesis: Dense retrieval alone misses exact identifiers (section numbers, form names) and returns number tables for figure questions.

- split `dev`: 49 questions, 43 answerable scored
- git `cba34ef-dirty`, golden set `golden_set.jsonl` (sha1 70b416d2d43f)
- config: `{"mode": "vector", "dense_top_k": 30, "context_top_n": 5}` (all resolved settings in config.json)

## Overall

| hit@1 | hit@3 | hit@5 | hit@10 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| 0.326 | 0.628 | 0.721 | 0.767 | 0.663 | 0.733 | 0.492 | 0.546 |

## By question type and difficulty

| group | n | hit@5 | recall@5 | mrr | ndcg@10 |
|---|---|---|---|---|---|
| exact_term | 12 | 0.833 | 0.833 | 0.639 | 0.709 |
| paraphrase | 12 | 0.500 | 0.500 | 0.417 | 0.457 |
| numeric | 12 | 0.833 | 0.833 | 0.556 | 0.624 |
| multi_hop | 7 | 0.714 | 0.357 | 0.262 | 0.283 |
| easy | 15 | 0.867 | 0.867 | 0.639 | 0.712 |
| medium | 19 | 0.632 | 0.605 | 0.456 | 0.504 |
| hard | 9 | 0.667 | 0.444 | 0.324 | 0.357 |

## Latency (ms)

| stage | p50 | p95 |
|---|---|---|
| dense | 28 | 66 |
| generate | 39930 | 77887 |
| retrieve | 28 | 67 |
| total | 40062 | 77916 |

## Generation (ollama:llama3.1, judge gemini:gemini-3.5-flash-lite)

| metric | value | n |
|---|---|---|
| refusal_accuracy | 1.000 | 6 |
| false_refusal_rate | 0.163 | 43 |
| citation_strict | 0.639 | 36 |
| citation_lenient | 0.806 | 36 |
| uncited_rate | 0.000 | 36 |
| invalid_citation_rate | 0.028 | 36 |
| faithfulness | 0.898 | 36 |
| answer_relevancy | 0.806 | 36 |
| context_precision | 0.531 | 43 |
| context_recall | 0.721 | 43 |

Tokens per answer: input_tokens 2183, output_tokens 56

Judge `gemini:gemini-3.5-flash-lite` this session: 22 calls, 37258 input / 1684 output tokens, 0 errors (cached scores not counted).
