"""Print the full RagResult for a few questions in every retrieval mode (Step 1 check).

uv run python scripts/show_rag_result.py                      # 3 questions x 4 modes, with the LLM
uv run python scripts/show_rag_result.py --retrieve-only      # contexts + latency only, seconds
uv run python scripts/show_rag_result.py --provider gemini --model gemini-3.5-flash-lite
uv run python scripts/show_rag_result.py -q "your question" --modes bm25 hybrid

Context text is cut to 160 characters for readability; --full-text prints all of it.
"""

import argparse
import json
import sys

from hybrid_rag.pipeline import RagConfig, run_rag
from hybrid_rag.retrieval.retriever import MODES

# One of each kind that matters: an exact identifier, a paraphrase, and one the corpus can't answer.
QUESTIONS = [
    "What does section 148A of the Income-tax Act provide?",
    "How much will the government borrow relative to the size of the economy next year?",
    "What is the GST rate on cryptocurrency trading?",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("-q", "--question", action="append", help="repeatable; default: 3 built-in")
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--retrieve-only", action="store_true", help="skip the LLM")
    p.add_argument("--provider", choices=["ollama", "gemini", "openai", "anthropic"])
    p.add_argument("--model")
    p.add_argument("--full-text", action="store_true")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    overrides = {k: v for k, v in (("llm_provider", args.provider), ("llm_model", args.model)) if v}
    for question in args.question or QUESTIONS:
        for mode in args.modes:
            print(f"\n=== {mode} | {question}", flush=True)
            result = run_rag(
                question, RagConfig(mode=mode, **overrides), generate=not args.retrieve_only
            )
            data = result.model_dump()
            if not args.full_text:
                for c in data["contexts"]:
                    c["text"] = c["text"][:160].replace("\n", " ") + " ..."
            print(json.dumps(data, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
