"""Draft golden-set questions with an LLM. Every draft is then reviewed by hand
(scripts/review_golden_set.py) before it counts.

uv run python scripts/build_golden_set.py            # ~70 drafts -> eval/golden_set.draft.jsonl
uv run python scripts/build_golden_set.py --n 10     # quick trial

How it works:
- Chunks are sampled across documents, weighted by sqrt(chunk count) so the 1,100-chunk mutual
  fund circular doesn't swamp the 39-chunk Budget at a Glance. Blank form templates, rescinded-
  circular lists and tiny chunks are skipped.
- The type mix follows CLAUDE.md: exact_term 25%, paraphrase 25%, numeric 20%, multi_hop 15%,
  unanswerable 15%. Each type has its own instruction (see TYPE_RULES).
- multi_hop pairs a chunk with its nearest neighbour (by embedding) from another document or a
  distant page, and asks for a question that needs both.
- Labels are (doc, pages) from the source chunks' page span, never chunk ids, so chunk-size
  experiments can reuse the same golden set.

Leakage caveat: questions written from our own chunks favour our chunker. That's why ~30% of
the final set is written by hand from the PDFs (review_golden_set.py --add).
"""

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

from hybrid_rag.config import ROOT, settings
from hybrid_rag.generation import llm
from hybrid_rag.indexing import embed, vector_store

DRAFT_PATH = ROOT / "eval" / "golden_set.draft.jsonl"
MIX = {
    "exact_term": 0.25,
    "paraphrase": 0.25,
    "numeric": 0.20,
    "multi_hop": 0.15,
    "unanswerable": 0.15,
}
SKIP_HEADINGS = ("RESCINDED", "INDEX", "CONTENTS")
NUMBER = re.compile(r"\d[\d,.]*\d")
DOT_LEADERS = re.compile(r"\.{5,}")  # table-of-contents lines: "Chapter 3 ........ 46"
# 148A, 2(u), LRS, clause (d)(i)
IDENTIFIER = re.compile(r"\b(?=\w*\d)\w+(?:[(./-]\w+\)?)*|\b[A-Z]{2,}\b|\(\w{1,4}\)")
BLANKS = re.compile(r"_{3,}")  # fill-in-the-blank form templates: "______ subsidiaries"

SYSTEM = """You write evaluation questions for a question-answering system over Indian \
financial regulations and Union Budget documents. Every question must make sense on its own \
to someone who has NOT seen the passage: name the document, scheme, regulator or entity it is \
about (e.g. "under SEBI's mutual fund master circular", "in the Union Budget 2026-27"). Never \
write "the passage", "the table", "column C" or "the said section". \
Reply with a single JSON object only."""

TYPE_RULES = {
    "exact_term": "The question MUST contain a specific identifier copied exactly from the "
    "passage: a section/regulation/clause number, circular number, form name, or acronym "
    "(e.g. 'section 148A', 'Regulation 2(u)'). Answer from the passage.",
    "paraphrase": "Ask about a fact in the passage the way a non-expert would, WITHOUT reusing "
    "its distinctive words, numbers or identifiers (say 'borrowing compared to the size of the "
    "economy', not 'debt-to-GDP ratio'). Answer from the passage.",
    "numeric": "Ask for one specific figure (amount, rate, percentage, limit, date) stated in "
    "the passage, preferably from a table. The answer must quote the figure exactly, with units.",
    "multi_hop": "Ask ONE question whose answer needs facts from BOTH passages A and B. If the "
    'two passages are unrelated, reply {"skip": true}.',
    "unanswerable": "Ask a realistic question on the SAME topic that the passage does NOT answer "
    "(a different year, figure, entity or condition), so it sounds answerable but isn't. Set "
    '"answer" to "NOT_IN_CORPUS".',
}

FORMAT = '{"question": "...", "answer": "short answer, figures exactly as written"}'


def load_chunks() -> list[dict]:
    with (settings.processed_dir / "chunks.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def is_table_heavy(chunk: dict) -> bool:
    lines = chunk["text"].splitlines()
    return sum(ln.startswith("|") for ln in lines) > len(lines) / 2


def usable(chunk: dict, qtype: str) -> bool:
    if chunk["n_tokens"] < 150 or any(s in chunk["section_heading"].upper() for s in SKIP_HEADINGS):
        return False
    if len(DOT_LEADERS.findall(chunk["text"])) >= 3 or len(BLANKS.findall(chunk["text"])) >= 2:
        return False  # table of contents, or a blank form template
    letters = sum(c.isalpha() for c in chunk["text"])
    if letters < 0.4 * len(chunk["text"]):  # mostly pipes/digits: blank forms, broken tables
        return qtype == "numeric" and len(NUMBER.findall(chunk["text"])) >= 8
    if qtype == "numeric":
        return len(NUMBER.findall(chunk["text"])) >= 4
    return not is_table_heavy(chunk)


def sample_docs(chunks: list[dict], n: int, rng: random.Random) -> list[str]:
    """n doc_ids drawn with probability proportional to sqrt(#chunks in doc)."""
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c["doc_id"]] = counts.get(c["doc_id"], 0) + 1
    docs = sorted(counts)
    return rng.choices(docs, weights=[math.sqrt(counts[d]) for d in docs], k=n)


def type_plan(n: int) -> list[str]:
    plan = [t for t, share in MIX.items() for _ in range(round(n * share))]
    return (plan + ["paraphrase"] * n)[:n]


def partner(chunk: dict, used: set[str]) -> dict | None:
    """Nearest chunk by embedding from another doc, or the same doc at least 5 pages away."""
    vec = embed.embed_passages([embed.passage_text(chunk)])[0]
    for hit in vector_store.search(vec, 20):
        far = hit["doc_id"] != chunk["doc_id"] or abs(hit["page_start"] - chunk["page_start"]) > 5
        if far and hit["chunk_id"] not in used and hit["n_tokens"] >= 150:
            return hit
    return None


def parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.S)  # models often wrap JSON in prose or ``` fences
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def fits_type(qtype: str, out: dict) -> bool:
    """Cheap checks that catch the most common bad drafts before a human has to."""
    if qtype == "numeric":
        return any(c.isdigit() for c in str(out["answer"]))
    if qtype == "exact_term":
        return bool(IDENTIFIER.search(out["question"]))
    return True


def passage_block(label: str, c: dict) -> str:
    return f"Passage {label} ({c['title']}, p.{c['page_start']}-{c['page_end']}):\n{c['text']}"


def label(chunks: list[dict]) -> list[dict]:
    """One {doc, pages} requirement per source chunk. Each must be retrieved for full recall,
    so a multi_hop question isn't "found" when only one of its two passages is."""
    return [
        {"doc": c["file_name"], "pages": list(range(c["page_start"], c["page_end"] + 1))}
        for c in chunks
    ]


def draft_one(qtype: str, sources: list[dict]) -> dict | None:
    names = "AB" if len(sources) == 2 else "A"
    passages = "\n\n".join(passage_block(n, c) for n, c in zip(names, sources, strict=True))
    user = f"{passages}\n\nTask: {TYPE_RULES[qtype]}\n\nReply as JSON: {FORMAT}"
    for _ in range(2):  # one retry if the JSON is broken or the draft doesn't fit its type
        out = parse_json(llm.complete(SYSTEM, user))
        if out and out.get("skip"):
            return None
        if out and out.get("question") and out.get("answer") and fits_type(qtype, out):
            return out
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=70, help="number of drafts to aim for")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=DRAFT_PATH)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    rng = random.Random(args.seed)
    chunks = load_chunks()
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)

    queue = type_plan(args.n)
    rng.shuffle(queue)
    used: set[str] = set()  # chunks already tried, so a failed chunk isn't drawn again
    n_drafts = attempts = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        while queue and attempts < 3 * args.n:
            attempts += 1
            qtype = queue.pop(0)
            doc_id = sample_docs(chunks, 1, rng)[0]
            pool = [c for c in by_doc[doc_id] if c["chunk_id"] not in used and usable(c, qtype)]
            sources = [rng.choice(pool)] if pool else []
            if sources and qtype == "multi_hop":
                other = partner(sources[0], used)
                sources = sources + [other] if other else []
            used.update(c["chunk_id"] for c in sources)

            out = draft_one(qtype, sources) if sources else None
            if out is None:
                queue.append(qtype)  # try this type again later with another chunk
                print(f"[{attempts}] {qtype:12} no usable draft, re-queued")
                continue
            n_drafts += 1
            unanswerable = qtype == "unanswerable"
            draft = {
                "id": f"d{n_drafts:03d}",
                "question": out["question"].strip(),
                "type": qtype,
                "expected_answer": "" if unanswerable else str(out["answer"]).strip(),
                "relevant": [] if unanswerable else label(sources),
                "source_chunks": [c["chunk_id"] for c in sources],  # for the reviewer only
                "origin": "llm_draft",
                "drafted_by": f"{settings.llm_provider}:{settings.llm_model}",
            }
            f.write(json.dumps(draft, ensure_ascii=False) + "\n")
            f.flush()  # a long run that gets interrupted keeps what it has
            print(f"[{attempts}] {qtype:12} {n_drafts}/{args.n}: {draft['question'][:80]}")

    print(f"\nwrote {n_drafts} drafts ({attempts} attempts) to {args.out}")
    print("Next: uv run python scripts/review_golden_set.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
