"""Run experiments (experiments/*.yaml) over the golden set; one results folder per run.

uv run python eval/run_experiment.py experiments/e0_baseline_vector.yaml
uv run python eval/run_experiment.py experiments/*.yaml               # the whole ladder, dev half
uv run python eval/run_experiment.py experiments/*.yaml --split test  # only to report a phase

Each run writes eval/results/<run_id>/ with config.json (experiment + every resolved setting +
git sha + golden-set hash), per_question.jsonl (metrics, ranked chunk ids and pages, latency),
summary.json (overall, by type, by difficulty) and report.md. Several experiments in one call
also write a side-by-side table, eval/results/ladder-<split>-<time>.md.

An experiment file is {name, hypothesis, config}; config holds RagConfig fields, so a typo fails
before anything runs. Everything goes through run_rag(), the same door the API and UI use.
Retrieval only for now: generation metrics (RAGAS, citations, refusals) come in Step 5.

Metrics need the top 10 chunks, so retrieval is asked for max(10, context_top_n); context_top_n
still says what the LLM would see. Unanswerable questions are skipped (nothing to retrieve).
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml
from metrics import percentile, retrieval_scores
from run_eval import split
from schema import TYPES, load_golden

from hybrid_rag.config import ROOT
from hybrid_rag.pipeline import RagConfig, run_rag

GOLDEN = ROOT / "eval" / "golden_set.jsonl"
RESULTS = ROOT / "eval" / "results"
METRICS = ("hit@1", "hit@3", "hit@5", "hit@10", "recall@5", "recall@10", "mrr", "ndcg@10")
CONTEXT_FIELDS = ("rank", "chunk_id", "file_name", "page_start", "page_end", "score")
RANK_FIELDS = ("vector_rank", "bm25_rank", "rrf_rank", "rerank_score")
ANSWERABLE = tuple(t for t in TYPES if t != "unanswerable")
LEVELS = ("easy", "medium", "hard")


def load_experiment(path: Path) -> tuple[dict, RagConfig]:
    exp = yaml.safe_load(path.read_text(encoding="utf-8"))
    if set(exp) != {"name", "hypothesis", "config"}:
        raise SystemExit(f"{path}: needs exactly name, hypothesis, config (got {sorted(exp)})")
    return exp, RagConfig(**exp["config"])


def git_sha() -> str:
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout

    sha = git("rev-parse", "--short", "HEAD").strip() or "nogit"
    return sha + ("-dirty" if git("status", "--porcelain", "--untracked-files=no").strip() else "")


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["metrics"]]

    def block(group):
        return {"n": len(group), **{m: mean(r["metrics"][m] for r in group) for m in METRICS}}

    stages = sorted({s for r in rows for s in r["latency_ms"]})
    return {
        "n_questions": len(rows),
        "overall": block(scored),
        "by_type": {t: block([r for r in scored if r["type"] == t]) for t in ANSWERABLE},
        "by_difficulty": {d: block([r for r in scored if r["difficulty"] == d]) for d in LEVELS},
        "latency_ms": {
            s: {
                "p50": percentile([r["latency_ms"][s] for r in rows if s in r["latency_ms"]], 50),
                "p95": percentile([r["latency_ms"][s] for r in rows if s in r["latency_ms"]], 95),
            }
            for s in stages
        },
    }


def run_one(exp: dict, cfg: RagConfig, questions: list[dict], split_name: str) -> Path:
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{exp['name']}"
    out = RESULTS / run_id
    out.mkdir(parents=True)
    retrieve_cfg = cfg.model_copy(update={"context_top_n": max(10, cfg.context_top_n)})
    meta = {
        "run_id": run_id,
        "experiment": exp,
        "config": cfg.model_dump(),  # every knob, defaults included
        "git_sha": git_sha(),
        "golden_set": GOLDEN.name,
        "golden_sha1": hashlib.sha1(GOLDEN.read_bytes()).hexdigest()[:12],
        "split": split_name,
    }
    (out / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rows = []
    with (out / "per_question.jsonl").open("w", encoding="utf-8") as f:
        for n, q in enumerate(questions, start=1):
            result = run_rag(q["question"], retrieve_cfg, generate=False)
            contexts = [c.model_dump() for c in result.contexts]
            row = {
                **{k: q[k] for k in ("id", "type", "difficulty", "split")},
                "metrics": retrieval_scores(contexts, q["relevant"]) if q["answerable"] else None,
                "latency_ms": {k: v for k, v in result.latency_ms.items() if k != "generate"},
                "contexts": [
                    {k: c[k] for k in CONTEXT_FIELDS + RANK_FIELDS if c[k] is not None}
                    for c in contexts
                ],
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"\r  {exp['name']:16} {n}/{len(questions)}", end="", flush=True)
    print()

    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "report.md").write_text(report(meta, summary), encoding="utf-8")
    return out


def fmt(x: float) -> str:
    return "-" if x != x else f"{x:.3f}"  # x != x is NaN


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def report(meta: dict, s: dict) -> str:
    exp, o = meta["experiment"], s["overall"]
    lines = [
        f"# {exp['name']} ({meta['run_id']})",
        "",
        f"Hypothesis: {exp['hypothesis']}",
        "",
        f"- split `{meta['split']}`: {s['n_questions']} questions, {o['n']} answerable scored",
        f"- git `{meta['git_sha']}`, golden set `{meta['golden_set']}` "
        f"(sha1 {meta['golden_sha1']})",
        f"- config: `{json.dumps(exp['config'])}` (all resolved settings in config.json)",
        "",
        "## Overall",
        "",
        *table(list(METRICS), [[fmt(o[m]) for m in METRICS]]),
        "",
        "## By question type and difficulty",
        "",
        *table(
            ["group", "n", "hit@5", "recall@5", "mrr", "ndcg@10"],
            [
                [g, str(b["n"]), *(fmt(b[m]) for m in ("hit@5", "recall@5", "mrr", "ndcg@10"))]
                for g, b in {**s["by_type"], **s["by_difficulty"]}.items()
            ],
        ),
        "",
        "## Latency (ms)",
        "",
        *table(
            ["stage", "p50", "p95"],
            [[k, str(v["p50"]), str(v["p95"])] for k, v in s["latency_ms"].items()],
        ),
    ]
    return "\n".join(lines) + "\n"


def ladder(runs: list[tuple[dict, dict]], split_name: str) -> str:
    cols = ("hit@1", "hit@5", "recall@5", "recall@10", "mrr", "ndcg@10")
    lines = [f"# Retrieval ladder ({split_name} split, {datetime.now():%Y-%m-%d %H:%M})", ""]
    lines += table(
        ["experiment", *cols, "p95 retrieve ms"],
        [
            [m["experiment"]["name"], *(fmt(s["overall"][c]) for c in cols)]
            + [str(s["latency_ms"]["retrieve"]["p95"])]
            for m, s in runs
        ],
    )
    for metric in ("hit@5", "mrr"):
        groups = [*ANSWERABLE, *LEVELS]
        n = {g: {**runs[0][1]["by_type"], **runs[0][1]["by_difficulty"]}[g]["n"] for g in groups}
        lines += ["", f"## {metric} by question type and difficulty", ""]
        lines += table(
            ["experiment", *(f"{g} (n={n[g]})" for g in groups)],
            [
                [m["experiment"]["name"]]
                + [fmt({**s["by_type"], **s["by_difficulty"]}[g][metric]) for g in groups]
                for m, s in runs
            ],
        )
    lines += ["", "Runs: " + ", ".join(f"`{m['run_id']}`" for m, _ in runs)]
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("experiments", nargs="+", type=Path)
    p.add_argument("--split", choices=["dev", "test", "smoke", "all"], default="dev")
    p.add_argument("--limit", type=int, help="first N questions only (a quick try)")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    loaded = [load_experiment(path) for path in args.experiments]  # fail on a typo before running
    questions = split([q.model_dump() for q in load_golden(GOLDEN)], args.split)[: args.limit]
    print(f"{len(questions)} questions ({args.split} split), {len(loaded)} experiment(s)")
    runs = []
    for exp, cfg in loaded:
        out = run_one(exp, cfg, questions, args.split)
        meta = json.loads((out / "config.json").read_text(encoding="utf-8"))
        runs.append((meta, json.loads((out / "summary.json").read_text(encoding="utf-8"))))
        print(f"  -> {out.relative_to(ROOT)}")
    if len(runs) > 1:
        path = RESULTS / f"ladder-{args.split}-{datetime.now():%Y%m%d-%H%M}.md"
        path.write_text(ladder(runs, args.split), encoding="utf-8")
        print("\n" + path.read_text(encoding="utf-8") + f"\nsaved {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
