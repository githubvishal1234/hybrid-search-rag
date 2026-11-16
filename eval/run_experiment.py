"""Run experiments (experiments/*.yaml) over the golden set; one results folder per run.

uv run python eval/run_experiment.py experiments/e0_baseline_vector.yaml
uv run python eval/run_experiment.py experiments/*.yaml               # the whole ladder, dev half
uv run python eval/run_experiment.py experiments/*.yaml --split test  # only to report a phase
uv run python eval/run_experiment.py experiments/e1_bm25_only.yaml --generate --llm ollama:llama3.1
uv run python eval/run_experiment.py --rejudge eval/results/<run> --judge gemini:<model>

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
from metrics import citation_correct, percentile, retrieval_scores
from run_eval import split
from schema import TYPES, load_golden

from hybrid_rag.config import ROOT, settings
from hybrid_rag.pipeline import RagConfig, run_rag

GOLDEN = ROOT / "eval" / "golden_set.jsonl"
RESULTS = ROOT / "eval" / "results"
METRICS = ("hit@1", "hit@3", "hit@5", "hit@10", "recall@5", "recall@10", "mrr", "ndcg@10")
CONTEXT_FIELDS = ("rank", "chunk_id", "file_name", "page_start", "page_end", "score")
RANK_FIELDS = ("vector_rank", "bm25_rank", "rrf_rank", "rerank_score")
ANSWERABLE = tuple(t for t in TYPES if t != "unanswerable")
LEVELS = ("easy", "medium", "hard")
RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
GEN_COLS = (
    "refusal_accuracy",
    "false_refusal_rate",
    "citation_strict",
    "citation_lenient",
    "uncited_rate",
    "invalid_citation_rate",
    *RAGAS_METRICS,
)


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


def generation_row(q: dict, result, cfg: RagConfig, scorer) -> dict:
    """Answer-side checks for one question: citations against the golden pages, refusals, tokens,
    and (with a judge) RAGAS on the contexts the LLM actually saw."""
    cites = [c.model_dump() for c in result.citations]
    correct = [citation_correct(c, q["relevant"]) for c in cites] if q["answerable"] else []
    row = {
        "answer": result.answer,
        "refused": result.refused,
        "uncited": result.uncited,
        "citations": [{k: c[k] for k in ("doc", "page", "chunk_id")} for c in cites],
        "invalid_citations": result.invalid_citations,  # numbers the LLM made up, dropped
        "citation_strict": all(correct) if correct else None,  # every cited page relevant
        "citation_lenient": any(correct) if correct else None,  # at least one is
        "tokens": result.tokens.model_dump(),
    }
    if scorer:
        seen = [c.text for c in result.contexts[: cfg.context_top_n]]
        row["ragas"] = scorer.score(
            q["question"], result.answer, seen, q["expected_answer"], result.refused
        )
    return row


def summarize_generation(rows: list[dict]) -> dict:
    gen = [r for r in rows if r.get("gen")]
    if not gen:
        return {}
    ans = [r for r in gen if r["metrics"] is not None]
    una = [r for r in gen if r["metrics"] is None]
    answered = [r for r in gen if not r["gen"]["refused"]]

    def rate(group, key):
        vals = [r["gen"][key] for r in group if r["gen"][key] is not None]
        return {"value": mean(map(float, vals)), "n": len(vals)}

    out = {
        "n": len(gen),
        "refusal_accuracy": rate(una, "refused"),  # unanswerable and refused: good
        "false_refusal_rate": rate(ans, "refused"),  # answerable but refused: bad
        "citation_strict": rate(ans, "citation_strict"),
        "citation_lenient": rate(ans, "citation_lenient"),
        "uncited_rate": rate(answered, "uncited"),
        "invalid_citation_rate": {
            "value": mean(float(bool(r["gen"]["invalid_citations"])) for r in answered),
            "n": len(answered),
        },
        "tokens_per_answer": {
            k: mean(r["gen"]["tokens"][k] or 0 for r in gen)
            for k in ("input_tokens", "output_tokens")
        },
    }
    if any("ragas" in r["gen"] for r in gen):
        for m in RAGAS_METRICS:
            scores = [r["gen"].get("ragas") or {} for r in gen]  # partly judged runs too
            vals = [s[m] for s in scores if s.get(m) is not None]
            out[m] = {"value": mean(vals), "n": len(vals)}
    return out


def run_one(
    exp: dict,
    cfg: RagConfig,
    questions: list[dict],
    split_name: str,
    *,
    generate: bool = False,
    scorer=None,
    resume: Path | None = None,
) -> Path:
    """generate: also answer each question with cfg's LLM. scorer: a RagasScorer (judge).
    resume: an earlier run's folder; questions already in its per_question.jsonl are skipped."""
    rows: list[dict] = []
    if resume:
        out, meta = resume, json.loads((resume / "config.json").read_text(encoding="utf-8"))
        if meta["config"] != cfg.model_dump() or meta["generate"] != generate:
            raise SystemExit(f"{resume.name} was run with other settings; start a new run")
        with (out / "per_question.jsonl").open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{exp['name']}"
        out = RESULTS / run_id
        out.mkdir(parents=True)
        meta = {
            "run_id": run_id,
            "experiment": exp,
            "config": cfg.model_dump(),  # every knob, defaults included
            "generate": generate,
            "judge": scorer.judge if scorer else None,
            "git_sha": git_sha(),
            "golden_set": GOLDEN.name,
            "golden_sha1": hashlib.sha1(GOLDEN.read_bytes()).hexdigest()[:12],
            "split": split_name,
        }
        (out / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    done = {r["id"] for r in rows}
    with (out / "per_question.jsonl").open("a", encoding="utf-8") as f:
        for n, q in enumerate(questions, start=1):
            if q["id"] in done:
                continue
            result = run_rag(q["question"], cfg, generate=generate, retrieve_n=10)
            contexts = [c.model_dump() for c in result.contexts]
            row = {
                **{k: q[k] for k in ("id", "type", "difficulty", "split")},
                "metrics": retrieval_scores(contexts, q["relevant"]) if q["answerable"] else None,
                "latency_ms": result.latency_ms if generate else _no_generate(result.latency_ms),
                "contexts": [
                    {k: c[k] for k in CONTEXT_FIELDS + RANK_FIELDS if c[k] is not None}
                    for c in contexts
                ],
                "gen": generation_row(q, result, cfg, scorer) if generate else None,
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()  # an interrupted run keeps what it has (see --resume)
            print(f"\r  {exp['name']:16} {n}/{len(questions)}", end="", flush=True)
    print()

    write_summary(out, meta, rows, scorer)
    return out


def write_summary(out: Path, meta: dict, rows: list[dict], scorer) -> None:
    summary = summarize(rows)
    summary["generation"] = summarize_generation(rows)
    if scorer:
        m = scorer.meter
        summary["judge"] = {
            "judge": scorer.judge,
            "calls_this_session": m.calls,  # cached scores cost nothing and aren't counted
            "input_tokens_this_session": m.input_tokens,
            "output_tokens_this_session": m.output_tokens,
            "errors": scorer.errors,
            "daily_quota_hit": scorer.quota_exhausted,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "report.md").write_text(report(meta, summary), encoding="utf-8")


def rejudge(run: Path, scorer) -> bool:
    """Score (or finish scoring) the answers of a saved --generate run with RAGAS.

    Every row is sent to the scorer; scores already in the cache cost nothing, so this only pays
    for what's missing. It stops when the judge's daily quota runs out; run it again tomorrow.
    Returns True when every row was scored. Contexts are looked up by chunk id in chunks.jsonl,
    so the index must be the one the run used."""
    meta = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if not meta["generate"]:
        raise SystemExit(f"{run.name} has no answers to judge (it was run without --generate)")
    if meta.get("judge") not in (None, scorer.judge):
        raise SystemExit(f"{run.name} was judged by {meta['judge']}; keep one judge per ladder")
    if meta["golden_sha1"] != hashlib.sha1(GOLDEN.read_bytes()).hexdigest()[:12]:
        raise SystemExit("the golden set changed since this run; its references would differ")
    golden = {q.id: q.model_dump() for q in load_golden(GOLDEN)}
    with (settings.processed_dir / "chunks.jsonl").open(encoding="utf-8") as f:
        texts = {c["chunk_id"]: c["text"] for c in map(json.loads, f)}
    path = run / "per_question.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    top_n = meta["config"]["context_top_n"]
    for n, row in enumerate(rows, start=1):
        if scorer.quota_exhausted:
            break
        q, gen = golden[row["id"]], row["gen"]
        seen = [texts[c["chunk_id"]] for c in row["contexts"][:top_n]]
        gen["ragas"] = scorer.score(
            q["question"], gen["answer"], seen, q["expected_answer"], gen["refused"]
        )
        print(f"\r  rejudge {run.name} {n}/{len(rows)}", end="", flush=True)
    print()

    tmp = path.with_suffix(".tmp")  # replace in one step: a crash never leaves half a file
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    tmp.replace(path)
    meta["judge"] = scorer.judge
    (run / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_summary(run, meta, rows, scorer)
    return not scorer.quota_exhausted and not scorer.errors


def _no_generate(latency: dict) -> dict:
    return {k: v for k, v in latency.items() if k != "generate"}


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
    if g := s.get("generation"):
        lines += [
            "",
            f"## Generation ({meta['config']['llm_provider']}:{meta['config']['llm_model']}, "
            f"judge {meta.get('judge') or 'none'})",
            "",
            *table(
                ["metric", "value", "n"],
                [[k, fmt(g[k]["value"]), str(g[k]["n"])] for k in GEN_COLS if k in g],
            ),
            "",
            "Tokens per answer: "
            + ", ".join(f"{k} {v:.0f}" for k, v in g["tokens_per_answer"].items()),
        ]
    if j := s.get("judge"):
        lines += [
            "",
            f"Judge `{j['judge']}` this session: {j['calls_this_session']} calls, "
            f"{j['input_tokens_this_session']} input / {j['output_tokens_this_session']} output "
            f"tokens, {len(j['errors'])} errors (cached scores not counted).",
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
    gen_runs = [(m, s) for m, s in runs if s.get("generation")]
    if gen_runs:
        cols = [k for k in GEN_COLS if k in gen_runs[0][1]["generation"]]
        lines += ["", "## Generation", ""]
        lines += table(
            ["experiment", *cols, "p95 total ms"],
            [
                [m["experiment"]["name"], *(fmt(s["generation"][c]["value"]) for c in cols)]
                + [str(s["latency_ms"]["total"]["p95"])]
                for m, s in gen_runs
            ],
        )
    lines += ["", "Runs: " + ", ".join(f"`{m['run_id']}`" for m, _ in runs)]
    return "\n".join(lines) + "\n"


def _load_run(run: Path) -> tuple[dict, dict]:
    meta = json.loads((run / "config.json").read_text(encoding="utf-8"))
    return meta, json.loads((run / "summary.json").read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("experiments", nargs="*", type=Path)
    p.add_argument("--split", choices=["dev", "test", "smoke", "all"], default="dev")
    p.add_argument("--limit", type=int, help="first N questions only (a quick try)")
    p.add_argument("--generate", action="store_true", help="also answer with the config's LLM")
    p.add_argument("--llm", metavar="PROVIDER:MODEL", help="override the answering LLM")
    p.add_argument("--judge", metavar="PROVIDER:MODEL", help="score answers with RAGAS")
    p.add_argument("--resume", type=Path, metavar="RUN_DIR", help="continue an interrupted run")
    p.add_argument(
        "--rejudge", nargs="+", type=Path, metavar="RUN_DIR",
        help="score saved --generate runs with --judge (resumable; stops at the daily quota)",
    )  # fmt: skip
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.rejudge:
        if not args.judge or args.experiments:
            raise SystemExit("--rejudge takes run folders and --judge, no experiment files")
        from generation_metrics import RagasScorer  # needs the eval extra

        scorer = RagasScorer(*args.judge.split(":", 1))
        done = [rejudge(run, scorer) for run in args.rejudge if not scorer.quota_exhausted]
        runs = [_load_run(run) for run in args.rejudge]
        path = RESULTS / f"ladder-{runs[0][0]['split']}-{datetime.now():%Y%m%d-%H%M}.md"
        path.write_text(ladder(runs, runs[0][0]["split"]), encoding="utf-8")
        print(f"{scorer.meter.calls} judge calls; saved {path.relative_to(ROOT)}")
        if scorer.quota_exhausted:
            print("Daily judge quota used up: run the same command again tomorrow.")
        return 0 if len(done) == len(args.rejudge) and all(done) else 1
    if not args.experiments:
        raise SystemExit("pass experiment files (or --rejudge RUN_DIR...)")
    if args.judge and not args.generate:
        raise SystemExit("--judge scores answers, so it needs --generate")
    if args.resume and len(args.experiments) != 1:
        raise SystemExit("--resume continues one run: pass exactly one experiment file")

    loaded = [load_experiment(path) for path in args.experiments]  # fail on a typo before running
    if args.llm:
        provider, _, model = args.llm.partition(":")
        loaded = [(e, c.model_copy(update={"llm_provider": provider, "llm_model": model}))
                  for e, c in loaded]  # fmt: skip
    scorer = None
    if args.judge:
        from generation_metrics import RagasScorer  # needs the eval extra

        scorer = RagasScorer(*args.judge.split(":", 1))
    questions = split([q.model_dump() for q in load_golden(GOLDEN)], args.split)[: args.limit]
    print(f"{len(questions)} questions ({args.split} split), {len(loaded)} experiment(s)")
    runs = []
    for exp, cfg in loaded:
        out = run_one(
            exp, cfg, questions, args.split, generate=args.generate, scorer=scorer,
            resume=args.resume,
        )  # fmt: skip
        runs.append(_load_run(out))
        print(f"  -> {out.resolve().relative_to(ROOT)}")
    if len(runs) > 1:
        path = RESULTS / f"ladder-{args.split}-{datetime.now():%Y%m%d-%H%M}.md"
        path.write_text(ladder(runs, args.split), encoding="utf-8")
        print("\n" + path.read_text(encoding="utf-8") + f"\nsaved {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
