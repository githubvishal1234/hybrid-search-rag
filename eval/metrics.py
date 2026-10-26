"""Retrieval and answer metrics, computed at page level.

A golden question lists its requirements as `relevant: [{"doc", "pages"}, ...]` (one per passage
it needs; two for most multi_hop questions). A retrieved chunk satisfies a requirement when it
comes from that doc and its page span overlaps those pages. Working on pages, not chunk ids, is
what lets the same golden set score any chunk size.

- Recall@k:  share of requirements satisfied by the top k chunks. For a single-passage question
             this is simply "is a relevant page in the top k?" (hit rate).
- MRR:       1 / rank of the first chunk that satisfies any requirement (0 if none in the list).
             Rewards putting the right chunk first, which matters because the LLM only sees ~5.
- nDCG@k:    each requirement counts once, at the rank where it is first satisfied, discounted by
             1/log2(rank+1); divided by the best possible ordering. Unlike recall it cares about
             order, unlike MRR it credits finding the second passage of a multi_hop question.
"""

import math
import re


def overlaps(chunk: dict, req: dict) -> bool:
    return chunk["file_name"] == req["doc"] and any(
        chunk["page_start"] <= p <= chunk["page_end"] for p in req["pages"]
    )


def satisfied(chunk: dict, relevant: list[dict]) -> set[int]:
    """Indexes of the requirements this chunk satisfies."""
    return {i for i, req in enumerate(relevant) if overlaps(chunk, req)}


def recall_at(chunks: list[dict], relevant: list[dict], k: int) -> float:
    found = set().union(*(satisfied(c, relevant) for c in chunks[:k]))
    return len(found) / len(relevant)


def first_hit_rank(chunks: list[dict], relevant: list[dict]) -> int | None:
    return next((r for r, c in enumerate(chunks, start=1) if satisfied(c, relevant)), None)


def reciprocal_rank(chunks: list[dict], relevant: list[dict]) -> float:
    rank = first_hit_rank(chunks, relevant)
    return 1 / rank if rank else 0.0


def ndcg_at(chunks: list[dict], relevant: list[dict], k: int) -> float:
    seen: set[int] = set()
    dcg = 0.0
    for rank, c in enumerate(chunks[:k], start=1):
        new = satisfied(c, relevant) - seen
        if new:
            dcg += len(new) / math.log2(rank + 1)  # a chunk spanning both passages earns both
            seen |= new
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def retrieval_scores(chunks: list[dict], relevant: list[dict]) -> dict[str, float]:
    return {
        "recall@5": recall_at(chunks, relevant, 5),
        "recall@10": recall_at(chunks, relevant, 10),
        "mrr": reciprocal_rank(chunks, relevant),
        "ndcg@10": ndcg_at(chunks, relevant, 10),
    }


PAGE_LABEL = re.compile(r"p\.(\d+)(?:-(\d+))?")


def citation_correct(citation: dict, relevant: list[dict]) -> bool:
    """citation: {"doc": "file.pdf", "page": "p.12-13", ...} as returned by answer()."""
    start, end = PAGE_LABEL.match(citation["page"]).groups()
    span = {"file_name": citation["doc"], "page_start": int(start), "page_end": int(end or start)}
    return bool(satisfied(span, relevant))


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100). No numpy needed for a few hundred values."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]
