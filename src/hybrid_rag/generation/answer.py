"""Citation validation for generated answers (the pipeline itself is hybrid_rag.pipeline).

Citation check: the model cites sources by number. Numbers that point to a source we actually
gave it become "[file_name, p.X]"; any other number is a hallucinated citation, so it is removed
from the answer and reported in `invalid_citations`.
"""

import re

CITATION = re.compile(r"\[(\d+)\]")


def page_label(chunk: dict) -> str:
    start, end = chunk["page_start"], chunk["page_end"]
    return f"p.{start}" if start == end else f"p.{start}-{end}"


def resolve_citations(raw: str, chunks: list[dict]) -> tuple[str, list[dict], list[int]]:
    """'... 75% [2].' -> ('... 75% [file.pdf, p.3].', citations, invalid_numbers)"""
    cited, invalid = [], []

    def replace(m: re.Match) -> str:
        n = int(m.group(1))
        if not 1 <= n <= len(chunks):
            invalid.append(n)
            return ""
        if n not in cited:
            cited.append(n)
        c = chunks[n - 1]
        return f"[{c['file_name']}, {page_label(c)}]"

    text = re.sub(r"\s+([.,;])", r"\1", CITATION.sub(replace, raw)).strip()
    citations = [
        {
            "doc": chunks[n - 1]["file_name"],
            "page": page_label(chunks[n - 1]),
            "chunk_id": chunks[n - 1]["chunk_id"],
            "snippet": chunks[n - 1]["text"][:300],
        }
        for n in cited
    ]
    return text, citations, invalid


def answer(question: str, mode: str = "hybrid") -> dict:
    """/query-shaped dict for eval/run_eval.py, which Step 4 of docs/EVAL_CLAUDE.md replaces.
    New code calls hybrid_rag.pipeline.run_rag directly."""
    from hybrid_rag.pipeline import RagConfig, run_rag  # the pipeline imports this module

    return run_rag(question, RagConfig(mode=mode)).to_response()
