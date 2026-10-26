"""Cleaned pages of one document -> chunks of at most `size` tokens.

How it works:
1. Every paragraph is broken into units: sentences for prose, groups of rows for tables (each
   group repeats the table's header row, so "2025-26 BE" still labels the numbers in a later
   piece). A unit longer than `size` on its own is cut into token windows.
2. Units are packed greedily into a chunk until the next one would overflow `size`.
3. On overflow, the next chunk starts with the last few units of the previous one (up to
   `overlap` tokens), so a fact sitting on a boundary is fully inside at least one chunk.
4. A heading ("15.21. Guidelines for ...", "ANNEXURE 16", all-caps titles) closes the current
   chunk if it already has a reasonable amount of text, so chunks rarely straddle two sections.
   The heading in effect is stored as `section_heading`.
Chunks may cross page breaks; `page_start`/`page_end` record the span for citations and eval.

Token counts come from a function passed in (the embedding model's tokenizer in the pipeline, a
word counter in tests), because bge models truncate input beyond 512 tokens.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

CountTokens = Callable[[str], int]

SENTENCE_END = re.compile(r"(?<=[.;?!])\s+(?=[A-Z(“\"'])")
# "CHAPTER IV", "Annexure 17A", "PART B". Not "section": legal sentences start with "section 12 ..."
HEADING_KEYWORD = re.compile(
    r"^(?i:chapter|part|annexure|schedule|appendix)\s+([\dIVXLC]+[A-Z]?|[A-Z])\b"
)
# "15.21. Guidelines for Digital Accessibility", "7. In section 144C of the Income-tax Act,––"
NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")
MAX_HEADING_WORDS = 14
MAX_NUMBERED_HEADING_WORDS = 10  # longer numbered lines are paragraphs: "5. The tabular format..."


@dataclass
class Unit:
    text: str
    page: int
    tokens: int
    new_paragraph: bool
    table_header: str = ""  # set on table pieces; lets _join skip a header already in the chunk


def is_heading(paragraph: str) -> bool:
    words = paragraph.split()
    if not words or len(words) > MAX_HEADING_WORDS or paragraph.startswith("|"):
        return False
    if paragraph.rstrip().endswith((".", ",", ";")):
        return False
    letters = [c for c in paragraph if c.isalpha()]
    all_caps = len(letters) >= 6 and all(c.isupper() for c in letters)
    numbered = (
        NUMBERED_HEADING.match(paragraph)
        and len(words) <= MAX_NUMBERED_HEADING_WORDS
        and ", " not in paragraph
    )
    return bool(HEADING_KEYWORD.match(paragraph) or numbered or all_caps)


def _token_windows(text: str, size: int, count: CountTokens) -> list[str]:
    """Last resort for a single sentence longer than `size`: cut on word boundaries."""
    pieces, current = [], []
    for word in text.split():
        if current and count(" ".join([*current, word])) > size:
            pieces.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _table_header(table: str) -> str:
    rows = table.splitlines()
    return "\n".join(rows[:2]) if len(rows) > 2 and "---" in rows[1] else ""


def _table_pieces(table: str, size: int, count: CountTokens) -> list[str]:
    rows = table.splitlines()
    header = _table_header(table).splitlines()
    body = rows[len(header) :]
    pieces, current = [], list(header)
    for row in body:
        if len(current) > len(header) and count("\n".join([*current, row])) > size:
            pieces.append("\n".join(current))
            current = list(header)
        current.append(row)
    pieces.append("\n".join(current))
    # A single row (or header) can still be too big; window it rather than overflow.
    fitted = []
    for p in pieces:
        fitted.extend([p] if count(p) <= size else _token_windows(p, size, count))
    return fitted


def _units(pages: list[dict], size: int, count: CountTokens):
    """Yield (heading_or_None, Unit) for every unit of the document, in order.

    Units are capped at half a chunk. Then a chunk can only overflow once it is over half full,
    so a heading followed by a big table never ends up alone in a tiny chunk.
    """
    size = size // 2
    for page in pages:
        for para in page["text"].split("\n\n"):
            if is_heading(para):
                yield para, Unit(para, page["page"], count(para), True)
                continue
            if para.startswith("|"):
                parts = _table_pieces(para, size, count)
            else:
                parts = []
                for sentence in SENTENCE_END.split(para):
                    if count(sentence) <= size:
                        parts.append(sentence)
                    else:
                        parts.extend(_token_windows(sentence, size, count))
            header = _table_header(para) if para.startswith("|") else ""
            for i, part in enumerate(parts):
                unit = Unit(part, page["page"], count(part), i == 0 or bool(header), header)
                yield None, unit


def _join(units: list[Unit]) -> str:
    text, prev = "", None
    for u in units:
        part = u.text
        if u.table_header and prev and prev.table_header == u.table_header:
            # Next piece of the same table: continue it instead of repeating its header.
            part = part.removeprefix(u.table_header).lstrip("\n")
            text += "\n" + part
        else:
            text = part if not text else text + ("\n\n" if u.new_paragraph else " ") + part
        prev = u
    return text


def chunk_document(pages: list[dict], size: int, overlap: int, count: CountTokens) -> list[dict]:
    """pages: [{"page": int, "text": str}, ...] for one document, in page order."""
    chunks: list[dict] = []
    current: list[Unit] = []
    current_heading = section = ""

    def flush(carry_overlap: bool) -> list[Unit]:
        if not current:
            return []
        chunks.append(
            {
                "text": _join(current),
                "page_start": current[0].page,
                "page_end": current[-1].page,
                "section_heading": section,
                "n_tokens": sum(u.tokens for u in current),
            }
        )
        if not carry_overlap:
            return []
        tail, budget = [], overlap
        for u in reversed(current):
            if u.tokens > budget:
                break
            tail.insert(0, u)
            budget -= u.tokens
        return tail

    for heading, unit in _units(pages, size, count):
        used = sum(u.tokens for u in current)
        if heading is not None:
            current_heading = heading
            if used >= size // 4:  # don't leave a tiny chunk behind just because a heading came
                current = flush(carry_overlap=False)
                section = current_heading
        elif current and used + unit.tokens > size:
            current = flush(carry_overlap=True)
            section = current_heading
            # Overlap + this unit could still overflow; drop overlap units until it fits.
            while current and sum(u.tokens for u in current) + unit.tokens > size:
                current.pop(0)
        if not current:
            section = current_heading
        current.append(unit)
    flush(carry_overlap=False)

    for i, c in enumerate(chunks):
        c["chunk_index"] = i
    return chunks
