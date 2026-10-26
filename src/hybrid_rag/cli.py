"""Ask a question from the terminal.

uv run python -m hybrid_rag.cli "What is the maximum LTV ratio for gold loans?"
uv run python -m hybrid_rag.cli --mode vector "..."  # vector | bm25 | hybrid (default)
uv run python -m hybrid_rag.cli --retrieve-only "..."  # show the ranking, skip the LLM
uv run python -m hybrid_rag.cli --json "..."         # full response, as the API will return it
"""

import argparse
import json
import sys

import anthropic
import openai

from hybrid_rag.config import settings
from hybrid_rag.generation.answer import answer
from hybrid_rag.retrieval.retriever import MODES, retrieve


def print_ranking(question: str, mode: str) -> None:
    print(f"\n{'#':>2}  {'vector':>6}  {'bm25':>4}  {'rrf':>3}  {'score':>7}  chunk")
    for i, c in enumerate(retrieve(question, mode), start=1):
        v, b, r = (c["ranks"].get(k, "-") for k in ("vector", "bm25", "rrf"))
        line = f"{i:>2}  {v:>6}  {b:>4}  {r:>3}  {c['score']:>7.4f}  {c['chunk_id']}"
        print(f"{line} p{c['page_start']}")
        print(f"{'':>31}{c['text'][:60].replace(chr(10), ' ')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("question")
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.add_argument("--retrieve-only", action="store_true", help="print the ranking, no LLM call")
    p.add_argument("--json", action="store_true", help="print the full JSON response")
    args = p.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252 (₹, “ ”)
    if args.retrieve_only:
        print_ranking(args.question, args.mode)
        return
    try:
        res = answer(args.question, args.mode)
    except (openai.APIConnectionError, anthropic.APIConnectionError):
        hint = "start it with `ollama serve`" if settings.llm_provider == "ollama" else "check it"
        sys.exit(f"Can't reach the {settings.llm_provider} LLM; {hint} (or set LLM_PROVIDER).")
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    print(f"\n{res['answer']}\n")
    for c in res["citations"]:
        print(f"  - {c['doc']}, {c['page']}  ({c['chunk_id']})")
    if res["invalid_citations"]:
        print(f"  ! dropped hallucinated citations: {res['invalid_citations']}")
    if res["uncited"]:
        print("  ! answer has no citations; treat it as unverified")
    timings = " | ".join(f"{stage} {ms} ms" for stage, ms in res["latency_ms"].items())
    print(f"\n  {timings}")


if __name__ == "__main__":
    main()
