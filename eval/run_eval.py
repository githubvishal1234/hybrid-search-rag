"""Compare retrieval modes on the golden set. Prints a report and saves it to eval/results/.

uv run python eval/run_eval.py                       # retrieval metrics, all 4 modes
uv run python eval/run_eval.py --sweep               # + RRF k x depth x BM25-weight sweep
uv run python eval/run_eval.py --split dev --sweep   # tune on the dev half; report on --split test
uv run python eval/run_eval.py --generate --modes vector hybrid_rerank   # + LLM answers (slow)
uv run python eval/run_eval.py --golden eval/golden_set.draft.jsonl --label drafts

Retrieval metrics are free and take minutes (the cross-encoder dominates), so tune with them.
--generate calls the LLM once per question per mode: ~30-60 s each with local llama3.1.
"""

import argparse
import csv
import hashlib
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


def split(questions: list[dict], which: str) -> list[dict]:
    """Stratified, deterministic 50/50 dev/test split: within each question type, order by a hash
    of the id and alternate. Choose settings on dev and report on test, so the reported numbers
    aren't tuned on the questions that measure them."""
    if which == "all":
        return questions
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_type[q["type"]].append(q)
    keep = set()
    for group in by_type.values():
        group.sort(key=lambda q: hashlib.md5(q["id"].encode()).hexdigest())
        keep |= {q["id"] for i, q in enumerate(group) if (i % 2 == 0) == (which == "dev")}
    return [q for q in questions if q["id"] in keep]


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
            for w in (1.0, 1.5, 2.0, 3.0):
                recalls, rrs = [], []
                for v, b, relevant in lists:
                    fused = rrf({"vector": v[:depth], "bm25": b[:depth]}, k, {"bm25": w})
                    recalls.append(recall_at(fused, relevant, 5))
                    rrs.append(reciprocal_rank(fused, relevant))
                row = {"depth": depth, "rrf_k": k, "bm25_weight": w}
                out.append(row | {"recall@5": mean(recalls), "mrr": mean(rrs)})
    return sorted(out, key=lambda r: (r["recall@5"], r["mrr"]), reverse=True)


def load_progress(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {(r["mode"], r["id"]): r for r in map(json.loads, f)}


def run_generation(questions: list[dict], modes: list[str], progress: Path) -> list[dict]:
    """Answers every question in every mode. Each answer is appended to `progress` as soon as it
    exists, and a rerun with the same label skips answers already there, so a crash or reboot
    during a multi-hour run loses at most one answer. Failed calls aren't saved, so they retry."""
    from hybrid_rag.generation.answer import answer  # needs the LLM; import only when asked

    done = load_progress(progress)
    if done:
        print(f"  resuming: {len(done)} answers already in {progress.name}")
    rows = []
    log = progress.open("a", encoding="utf-8")
    for mode in modes:
        for n, q in enumerate(questions, start=1):
            print(f"\r  {mode:14} {n}/{len(questions)}", end="", flush=True)
            if (mode, q["id"]) in done:
                rows.append(done[(mode, q["id"])])
                continue
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
                    # the chunks the LLM was shown, so eval/judge.py can score faithfulness later
                    "context_ids": "|".join(
                        c["chunk_id"] for c in res["retrieval_debug"]["chunks"]
                    ),
                }
            )
            log.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
            log.flush()
        print()
    log.close()
    return rows


def fmt(x: float) -> str:
    return "–" if x != x else f"{x:.3f}"  # x != x is True only for NaN (no questions)


def retrieval_tables(answerable: list[dict], retr: list[dict]) -> list[str]:
    lines = [
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
    return lines


def report(golden: Path, questions, retr, sweep, gen, unreviewed: bool, split: str) -> str:
    answerable = [q for q in questions if q["type"] != "unanswerable"]
    lines = [f"# Eval run {datetime.now():%Y-%m-%d %H:%M}", ""]
    if unreviewed:
        lines += ["> **UNREVIEWED golden set: numbers are for debugging, not for the README.**", ""]
    lines += [
        f"- golden set: `{golden.name}` (split: {split}): {len(questions)} questions "
        f"({len(answerable)} answerable)",
        f"- embedding `{settings.embedding_model}`, chunk_size {settings.chunk_size}, "
        f"rrf_k {settings.rrf_k}, BM25 weight {settings.rrf_bm25_weight}, "
        f"depth {settings.dense_top_k}/{settings.bm25_top_k}, "
        f"reranker `{settings.reranker_provider}:{settings.reranker_model}` "
        f"on top {settings.rerank_candidates}",
    ]
    if retr:
        lines += retrieval_tables(answerable, retr)

    if sweep:
        lines += ["", "## RRF sweep (hybrid, no rerank; best first, top 12)", ""]
        lines += ["| depth | rrf_k | BM25 weight | Recall@5 | MRR |", "|---|---|---|---|---|"]
        for s in sweep[:12]:
            cells = [s["depth"], s["rrf_k"], s["bm25_weight"], fmt(s["recall@5"]), fmt(s["mrr"])]
            lines.append("| " + " | ".join(map(str, cells)) + " |")

    if gen:
        lines += [
            "",
            f"## Generation (LLM `{settings.llm_provider}:{settings.llm_model}`, "
            f"top {settings.context_top_n} chunks)",
            "",
            "Citation acc. = answered, answerable questions with at least one citation on a "
            "relevant page. Refusal acc. = unanswerable questions refused. Faithfulness: run "
            "eval/judge.py on this run's -generation.csv.",
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
    p.add_argument("--no-retrieval", action="store_true", help="skip retrieval metrics")
    p.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    p.add_argument("--split", choices=["all", "dev", "test"], default="all")
    p.add_argument("--label", default="run")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    questions = load_golden(args.golden)[: args.limit]
    questions = split(questions, args.split)
    unreviewed = any(not q.get("reviewed") for q in questions)
    answerable = [q for q in questions if q["type"] != "unanswerable"]
    print(f"{len(questions)} questions ({len(answerable)} answerable, split {args.split})")
    if unreviewed:
        print("WARNING: golden set contains unreviewed drafts; don't report these numbers.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = RESULTS / f"{datetime.now():%Y%m%d-%H%M}-{args.label}"
    retr = []
    if not args.no_retrieval:
        print("retrieval:")
        retr = run_retrieval(answerable, args.modes)
        write_csv(stem.with_name(stem.name + "-retrieval.csv"), retr)  # saved before anything else
    sweep = run_sweep(answerable) if args.sweep else []
    gen = []
    if args.generate:
        print("generation:")
        progress = RESULTS / f"{args.label}-generation.progress.jsonl"
        gen = run_generation(questions, args.modes, progress)
        write_csv(stem.with_name(stem.name + "-generation.csv"), gen)
        failed = sum("error" in r for r in gen)
        if failed:  # keep the progress file: rerunning the same command retries only these
            print(f"{failed} calls failed; rerun the same command to retry just those")
        else:
            progress.unlink()  # finished: the next run with this label starts fresh

    text = report(args.golden, questions, retr, sweep, gen, unreviewed, args.split)
    print("\n" + text)
    stem.with_suffix(".md").write_text(text, encoding="utf-8")
    print(f"saved {stem.relative_to(ROOT)}.md (+ per-question CSVs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
