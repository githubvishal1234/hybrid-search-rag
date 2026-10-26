"""Compare retrieval modes on the golden set. Prints a report and saves it to eval/results/.

uv run python eval/run_eval.py                       # retrieval metrics, all 4 modes
uv run python eval/run_eval.py --sweep               # + RRF k x fusion depth sweep (hybrid)
uv run python eval/run_eval.py --generate --modes vector hybrid_rerank   # + LLM answers (slow)
uv run python eval/run_eval.py --golden eval/golden_set.draft.jsonl --label drafts

Retrieval metrics are free and take minutes (the cross-encoder dominates), so tune with them.
--generate calls the LLM once per question per mode: ~30-60 s each with local llama3.1.
"""

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from metrics import (
    citation_correct,
    first_hit_rank,
    percentile,
    recall_at,
    reciprocal_rank,
    retrieval_scores,
)

from hybrid_rag.config import ROOT, settings
from hybrid_rag.retrieval.dense import dense_search
from hybrid_rag.retrieval.fusion import rrf
from hybrid_rag.retrieval.retriever import MODES, retrieve
from hybrid_rag.retrieval.sparse import bm25_search

GOLDEN = ROOT / "eval" / "golden_set.jsonl"
RESULTS = ROOT / "eval" / "results"
TYPES = ("exact_term", "paraphrase", "numeric", "multi_hop")
METRICS = ("recall@5", "recall@10", "mrr", "ndcg@10")


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(xs) -> float:
    xs = list(xs)
    return statistics.fmean(xs) if xs else float("nan")


def run_retrieval(questions: list[dict], modes: list[str]) -> list[dict]:
    rows = []
    for mode in modes:
        retrieve("warm-up query", mode, top_n=10)  # load models first so p95 isn't load time
        for n, q in enumerate(questions, start=1):
            t0 = time.perf_counter()
            chunks = retrieve(q["question"], mode, top_n=10)
            ms = (time.perf_counter() - t0) * 1000
            rows.append(
                {
                    "mode": mode,
                    "id": q["id"],
                    "type": q["type"],
                    **retrieval_scores(chunks, q["relevant"]),
                    "first_hit_rank": first_hit_rank(chunks, q["relevant"]),
                    "retrieve_ms": round(ms),
                }
            )
            print(f"\r  {mode:14} {n}/{len(questions)}", end="", flush=True)
        print()
    return rows


def run_sweep(questions: list[dict]) -> list[dict]:
    """Retrieve each list once at full depth, then re-fuse with every (k, depth) offline."""
    lists = [
        (dense_search(q["question"], 30), bm25_search(q["question"], 30), q["relevant"])
        for q in questions
    ]
    out = []
    for depth in (10, 20, 30):
        for k in (5, 10, 20, 60):
            recalls, rrs = [], []
            for v, b, relevant in lists:
                fused = rrf({"vector": v[:depth], "bm25": b[:depth]}, k)
                recalls.append(recall_at(fused, relevant, 5))
                rrs.append(reciprocal_rank(fused, relevant))
            out.append({"depth": depth, "rrf_k": k, "recall@5": mean(recalls), "mrr": mean(rrs)})
    return out


def run_generation(questions: list[dict], modes: list[str]) -> list[dict]:
    from hybrid_rag.generation.answer import answer  # needs the LLM; import only when asked

    rows = []
    for mode in modes:
        for n, q in enumerate(questions, start=1):
            print(f"\r  {mode:14} {n}/{len(questions)}", end="", flush=True)
            try:
                res = answer(q["question"], mode)
            except Exception as e:  # one failed call shouldn't sink a 2-hour run
                rows.append({"mode": mode, "id": q["id"], "type": q["type"], "error": repr(e)})
                continue
            answerable = q["type"] != "unanswerable"
            rows.append(
                {
                    "mode": mode,
                    "id": q["id"],
                    "type": q["type"],
                    "refused": res["refused"],
                    "uncited": res["uncited"],
                    "n_citations": len(res["citations"]),
                    "citation_ok": answerable
                    and any(citation_correct(c, q["relevant"]) for c in res["citations"]),
                    "invalid_citations": len(res["invalid_citations"]),
                    "total_ms": res["latency_ms"]["total"],
                    "answer": res["answer"],
                    "expected": q["expected_answer"],
                }
            )
        print()
    return rows


def fmt(x: float) -> str:
    return "–" if x != x else f"{x:.3f}"  # x != x is True only for NaN (no questions)


def report(golden: Path, questions, retr, sweep, gen, unreviewed: bool) -> str:
    answerable = [q for q in questions if q["type"] != "unanswerable"]
    lines = [f"# Eval run {datetime.now():%Y-%m-%d %H:%M}", ""]
    if unreviewed:
        lines += ["> **UNREVIEWED golden set: numbers are for debugging, not for the README.**", ""]
    lines += [
        f"- golden set: `{golden.name}`: {len(questions)} questions ({len(answerable)} answerable)",
        f"- embedding `{settings.embedding_model}`, chunk_size {settings.chunk_size}, "
        f"rrf_k {settings.rrf_k}, depth {settings.dense_top_k}/{settings.bm25_top_k}, "
        f"reranker `{settings.reranker_provider}:{settings.reranker_model}` "
        f"on top {settings.rerank_candidates}",
        "",
        "## Retrieval (answerable questions, top 10)",
        "",
        "| Mode | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    by_mode = defaultdict(list)
    for r in retr:
        by_mode[r["mode"]].append(r)
    for mode, rows in by_mode.items():
        ms = [r["retrieve_ms"] for r in rows]
        cells = [fmt(mean(r[m] for r in rows)) for m in METRICS]
        cells += [f"{percentile(ms, 50):.0f}", f"{percentile(ms, 95):.0f}"]
        lines.append(f"| {mode} | {' | '.join(cells)} |")

    counts = {t: sum(q["type"] == t for q in answerable) for t in TYPES}
    lines += ["", "## Recall@5 by question type", ""]
    lines.append("| Mode | " + " | ".join(f"{t} (n={counts[t]})" for t in TYPES) + " |")
    lines.append("|---" * (len(TYPES) + 1) + "|")
    for mode, rows in by_mode.items():
        cells = [fmt(mean(r["recall@5"] for r in rows if r["type"] == t)) for t in TYPES]
        lines.append(f"| {mode} | {' | '.join(cells)} |")

    if sweep:
        lines += ["", "## RRF sweep (hybrid, no rerank)", "", "| depth | rrf_k | Recall@5 | MRR |"]
        lines.append("|---|---|---|---|")
        for s in sweep:
            lines.append(
                f"| {s['depth']} | {s['rrf_k']} | {fmt(s['recall@5'])} | {fmt(s['mrr'])} |"
            )

    if gen:
        lines += [
            "",
            f"## Generation (LLM `{settings.llm_provider}:{settings.llm_model}`, "
            f"top {settings.context_top_n} chunks)",
            "",
            "Citation acc. = answered, answerable questions with at least one citation on a "
            "relevant page. Refusal acc. = unanswerable questions refused. Faithfulness (RAGAS) "
            "is not computed yet.",
            "",
            "| Mode | Citation acc. | Refusal acc. | False refusals | Uncited | Errors "
            "| p50 ms | p95 ms |",
            "|---|---|---|---|---|---|---|---|",
        ]
        by_mode = defaultdict(list)
        for r in gen:
            by_mode[r["mode"]].append(r)
        for mode, rows in by_mode.items():
            ok = [r for r in rows if "error" not in r]
            ans = [r for r in ok if r["type"] != "unanswerable"]
            una = [r for r in ok if r["type"] == "unanswerable"]
            answered = [r for r in ans if not r["refused"]]
            ms = [r["total_ms"] for r in ok]
            lines.append(
                f"| {mode} | {fmt(mean(r['citation_ok'] for r in answered))} "
                f"| {fmt(mean(r['refused'] for r in una))} "
                f"| {fmt(mean(r['refused'] for r in ans))} "
                f"| {fmt(mean(r['uncited'] for r in ok))} | {len(rows) - len(ok)} "
                f"| {percentile(ms, 50):.0f} | {percentile(ms, 95):.0f} |"
            )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--golden", type=Path, default=GOLDEN)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--sweep", action="store_true", help="also sweep rrf_k x fusion depth")
    p.add_argument("--generate", action="store_true", help="also generate answers (slow)")
    p.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    p.add_argument("--label", default="run")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    questions = load_golden(args.golden)[: args.limit]
    unreviewed = any(not q.get("reviewed") for q in questions)
    answerable = [q for q in questions if q["type"] != "unanswerable"]
    print(f"{len(questions)} questions ({len(answerable)} answerable) from {args.golden}")
    if unreviewed:
        print("WARNING: golden set contains unreviewed drafts; don't report these numbers.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = RESULTS / f"{datetime.now():%Y%m%d-%H%M}-{args.label}"
    print("retrieval:")
    retr = run_retrieval(answerable, args.modes)
    write_csv(stem.with_name(stem.name + "-retrieval.csv"), retr)  # saved before anything else
    sweep = run_sweep(answerable) if args.sweep else []
    gen = []
    if args.generate:
        print("generation:")
        gen = run_generation(questions, args.modes)
        write_csv(stem.with_name(stem.name + "-generation.csv"), gen)

    text = report(args.golden, questions, retr, sweep, gen, unreviewed)
    print("\n" + text)
    stem.with_suffix(".md").write_text(text, encoding="utf-8")
    print(f"saved {stem.relative_to(ROOT)}.md (+ per-question CSVs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
