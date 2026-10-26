"""Turn raw PyMuPDF blocks into clean paragraphs.

Two levels:
- per block: drop Hindi lines, re-join hyphenated compounds split across lines, unwrap lines into
  one paragraph, collapse whitespace, drop chart-label soup. Markdown tables keep their rows.
- per document: drop running headers/footers. A line counts as one if it sits in the first or
  last few blocks of a page and, with digits masked ("Page 12 of 99" -> "page # of #"), repeats on
  a large share of pages. Masking digits is what lets page counters match each other.
"""

import re
from collections import Counter

from hybrid_rag.config import settings

DEVANAGARI = re.compile(r"[ऀ-ॿ]")  # U+0900..U+097F, the Devanagari (Hindi) block
PAGE_NUMBER = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.I)
EDGE_BLOCKS = 3  # how many blocks at the top/bottom of a page can be header/footer
# **bold** and _italic_ markers that PyMuPDF's to_markdown() adds inside table cells
MD_EMPHASIS = re.compile(r"\*\*|(?<!\w)_(?=\S)|(?<=\S)_(?!\w)")


def non_english_ratio(text: str) -> float:
    """Share of letters that aren't ASCII. Catches Unicode Hindi (Devanagari) and also Hindi typed
    in legacy fonts, which PyMuPDF extracts as Latin-1 junk like 'ºÉiªÉàÉä´É'."""
    letters = [c for c in text if c.isalpha()]
    return sum(not c.isascii() for c in letters) / len(letters) if letters else 0.0


def is_table(block: str) -> bool:
    return block.lstrip().startswith("|")


def clean_table(block: str) -> str:
    rows = []
    for ln in block.splitlines():
        ln = ln.replace("-<br>", "-").replace("<br>", " ")  # <br> = line break inside a cell
        ln = DEVANAGARI.sub("", ln)  # bilingual cells: keep the English half
        ln = re.sub(r"[ \t]+", " ", MD_EMPHASIS.sub("", ln)).strip()
        if ln.strip("| -") or "---" in ln:  # drop rows left with only empty cells
            rows.append(ln)
    return "\n".join(rows) if any(r.strip("| -") for r in rows) else ""


def clean_block(block: str) -> str:
    if is_table(block):
        return clean_table(block)
    lines = [ln.strip() for ln in block.splitlines()]
    lines = [ln for ln in lines if ln and non_english_ratio(ln) <= settings.max_non_english_ratio]

    text = ""
    for ln in lines:
        # A line ending in "word-" is joined without a space. In this corpus those are real
        # compounds ("non-" + "compliance", "paid-" + "up"), not soft hyphenation, so the hyphen
        # stays. (Measured on the parsed output: no "regu-lation" style splits found.)
        glue = "" if text.endswith("-") and text[-2:-1].isalpha() else " "
        text = f"{text}{glue}{ln}" if text else ln
    text = re.sub(r"\s+", " ", text).strip()
    # Charts are vector drawings whose labels come out as number soup ("9.2% 7.3% 2017-18").
    # ponytail: letter-ratio heuristic; also drops a lone "Rs. 50,00,000" block, which is rare.
    non_space = sum(not c.isspace() for c in text)
    if non_space and sum(c.isalpha() for c in text) / non_space < settings.min_letter_ratio:
        return ""
    return text


def _edge_key(block: str) -> str:
    return re.sub(r"\d+", "#", block.lower()).strip()


def clean_document(pages: list[list[str]]) -> list[list[str]]:
    """pages: raw blocks per page -> cleaned paragraphs per page (page order and count kept)."""
    pages = [[b for b in (clean_block(b) for b in page) if b] for page in pages]

    edge_counts = Counter()
    for page in pages:
        edges = page[:EDGE_BLOCKS] + page[-EDGE_BLOCKS:]
        edge_counts.update({_edge_key(b) for b in edges})
    min_pages = max(3, settings.header_footer_min_share * len(pages))
    repeated = {k for k, n in edge_counts.items() if n >= min_pages}

    out = []
    for page in pages:
        keep = []
        for i, b in enumerate(page):
            at_edge = i < EDGE_BLOCKS or i >= len(page) - EDGE_BLOCKS
            if at_edge and (_edge_key(b) in repeated or PAGE_NUMBER.match(b)):
                continue
            keep.append(b)
        out.append(keep)
    return out
