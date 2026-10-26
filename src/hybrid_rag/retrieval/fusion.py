"""Reciprocal Rank Fusion (RRF): merge ranked lists using only each item's rank.

    score(d) = sum over lists of 1 / (k + rank_i(d))     (rank starts at 1; absent = adds 0)

Why ranks and not scores: BM25 scores are unbounded (0..30+, depends on query length and term
rarity) while cosine similarities sit in a narrow band (~0.6..0.8 here). Adding or normalising
them needs per-query tuning; ranks are on the same scale by construction.

Why k = 60: it flattens the curve, so rank 1 (1/61) is not much more than rank 5 (1/65). A chunk
that is decent in BOTH lists (e.g. ranks 3 and 4: 1/63 + 1/64 = 0.0315) beats one that is first
in a single list (1/61 = 0.0164). That agreement bonus is the whole point of hybrid search.
"""


def rrf(ranked_lists: dict[str, list[dict]], k: int) -> list[dict]:
    """ranked_lists: {"vector": [chunk, ...], "bm25": [chunk, ...]}, each best first.

    Returns one list, best first. Each chunk gets "score" (its RRF score) and "ranks", e.g.
    {"vector": 3, "bm25": 1}, so the debug panel can show why it was ranked where it is.
    Ties keep the order in which chunks were first seen.
    """
    fused: dict[str, dict] = {}
    for name, hits in ranked_lists.items():
        for rank, hit in enumerate(hits, start=1):
            entry = fused.setdefault(hit["chunk_id"], {**hit, "score": 0.0, "ranks": {}})
            entry["score"] += 1 / (k + rank)
            entry["ranks"][name] = rank
    return sorted(fused.values(), key=lambda h: h["score"], reverse=True)
