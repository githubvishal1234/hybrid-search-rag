"""Compare two experiment runs: metric deltas, then the questions that changed.

uv run python eval/compare.py eval/results/<run_a> eval/results/<run_b>

Deltas smaller than --noise (default 0.03) are marked "~": on ~45 questions one question moves
a rate by ~0.02, and LLM-judged scores wobble by 0.02-0.05 between identical runs, so a smaller
"win" isn't one. Both runs should use the same split and golden set (checked).
"""

import argparse
import json
import sys
from pathlib import Path

RETRIEVAL = ("hit@1", "hit@5", "recall@5", "recall@10", "mrr", "ndcg@10")
GENERATION = (
    "refusal_accuracy",
    "false_refusal_rate",
    "citation_strict",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def load(run: Path) -> tuple[dict, dict, dict[str, dict]]:
    meta = json.loads((run / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    rows = {}
    if (run / "per_question.jsonl").exists():  # gitignored: only on the machine that ran it
        with (run / "per_question.jsonl").open(encoding="utf-8") as f:
            rows = {r["id"]: r for r in map(json.loads, f)}
    return meta, summary, rows


def delta(a: float | None, b: float | None, noise: float) -> str:
    if a is None or b is None or a != a or b != b:
        return "-"
    d = b - a
    return f"{d:+.3f}" + (" ~" if abs(d) < noise else "")


def compare(run_a: Path, run_b: Path, noise: float = 0.03) -> str:
    (ma, sa, ra), (mb, sb, rb) = load(run_a), load(run_b)
    if (ma["split"], ma["golden_sha1"]) != (mb["split"], mb["golden_sha1"]):
        raise SystemExit("runs used a different split or golden set: not comparable")
    a_name, b_name = ma["experiment"]["name"], mb["experiment"]["name"]
    lines = [f"# {a_name} -> {b_name} ({ma['split']} split)", "", "| metric | A | B | B - A |"]
    lines.append("|---|---|---|---|")
    for m in RETRIEVAL:
        a, b = sa["overall"][m], sb["overall"][m]
        lines.append(f"| {m} | {a:.3f} | {b:.3f} | {delta(a, b, noise)} |")
    ga, gb = sa.get("generation") or {}, sb.get("generation") or {}
    for m in GENERATION:
        if m in ga and m in gb:
            a, b = ga[m]["value"], gb[m]["value"]
            lines.append(f"| {m} | {a:.3f} | {b:.3f} | {delta(a, b, noise)} |")

    lines += ["", "## Hit@5 by question type", "", "| type | A | B | B - A |", "|---|---|---|---|"]
    for t, block in sa["by_type"].items():
        a, b = block["hit@5"], sb["by_type"][t]["hit@5"]
        lines.append(f"| {t} (n={block['n']}) | {a:.3f} | {b:.3f} | {delta(a, b, noise)} |")

    if ra and rb:  # which questions flipped, so a delta can be read, not just quoted
        common = [i for i in ra if i in rb and ra[i]["metrics"] and rb[i]["metrics"]]
        won = [i for i in common if rb[i]["metrics"]["hit@5"] > ra[i]["metrics"]["hit@5"]]
        lost = [i for i in common if rb[i]["metrics"]["hit@5"] < ra[i]["metrics"]["hit@5"]]
        lines += ["", "## Questions that changed (Hit@5)", ""]
        lines.append(f"- found by B only ({len(won)}): {', '.join(won) or 'none'}")
        lines.append(f"- found by A only ({len(lost)}): {', '.join(lost) or 'none'}")
        gen = [i for i in ra if i in rb and ra[i].get("gen") and rb[i].get("gen")]
        flips = [i for i in gen if ra[i]["gen"]["refused"] != rb[i]["gen"]["refused"]]
        if flips:
            lines.append(f"- refusal changed: {', '.join(flips)}")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    p.add_argument("--noise", type=float, default=0.03)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    print(compare(args.run_a, args.run_b, args.noise))
    return 0


if __name__ == "__main__":
    sys.exit(main())
