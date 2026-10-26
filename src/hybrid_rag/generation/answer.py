"""Retrieve -> prompt -> LLM -> validated, cited answer.

Citation check: the model cites sources by number. Numbers that point to a source we actually
gave it become "[file_name, p.X]"; any other number is a hallucinated citation, so it is removed
from the answer and reported in `invalid_citations`.
"""

import re
import time

from hybrid_rag.generation import llm, prompts
from hybrid_rag.retrieval.retriever import Mode, retrieve

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


def answer(question: str, mode: Mode = "hybrid") -> dict:
    t0 = time.perf_counter()
    chunks = retrieve(question, mode)
    t1 = time.perf_counter()
    raw = llm.complete(prompts.SYSTEM, prompts.user_prompt(question, chunks))
    t2 = time.perf_counter()

    text, citations, invalid = resolve_citations(raw, chunks)
    refused = prompts.REFUSAL.lower().rstrip(".") in raw.lower()
    return {
        "answer": text,
        "refused": refused,
        "citations": citations,
        "invalid_citations": invalid,
        "uncited": not refused and not citations,  # answered without grounding: flag it
        "retrieval_debug": {
            "mode": mode,
            # Per context chunk: where each retriever ranked it (None = not in its top-k).
            "chunks": [
                {
                    "rank": i,
                    "chunk_id": c["chunk_id"],
                    "vector_rank": c["ranks"].get("vector"),
                    "bm25_rank": c["ranks"].get("bm25"),
                    "rrf_rank": c["ranks"].get("rrf"),  # only in hybrid_rerank
                    "rerank_score": c.get("rerank_score"),
                    "score": round(c["score"], 4),
                }
                for i, c in enumerate(chunks, start=1)
            ],
        },
        "latency_ms": {
            "retrieve": round((t1 - t0) * 1000),
            "generate": round((t2 - t1) * 1000),
            "total": round((t2 - t0) * 1000),
        },
    }
