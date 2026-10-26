"""PDF -> list of pages, each page a list of raw text blocks in reading order.

A "block" is what PyMuPDF considers one paragraph-ish region of text. Tables are detected with
PyMuPDF's find_tables() and emitted as one markdown block each, so row/column structure (and the
numbers in it) survives; text blocks that sit inside a table's box are dropped to avoid duplicates.
Pages with almost no text but an embedded image are likely scans and go through OCR (Tesseract).
"""

import logging
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import pymupdf

from hybrid_rag.config import settings

log = logging.getLogger(__name__)


@cache
def ocr_available() -> bool:
    """Tesseract + language data installed? (winget install UB-Mannheim.TesseractOCR)"""
    try:
        pymupdf.get_tessdata()
    except RuntimeError:
        log.warning("Tesseract not found: image-only pages will have no text")
        return False
    return True


@dataclass
class Page:
    number: int  # 1-based, as printed by a PDF viewer; this is what citations use
    blocks: list[str] = field(default_factory=list)
    n_tables: int = 0
    ocr: bool = False


def _inside(block_bbox, table_bbox) -> bool:
    x0, y0, x1, y1 = block_bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    tx0, ty0, tx1, ty1 = table_bbox
    return tx0 <= cx <= tx1 and ty0 <= cy <= ty1


def _ocr_text(page: pymupdf.Page) -> str | None:
    if not ocr_available():
        return None
    tp = page.get_textpage_ocr(dpi=settings.ocr_dpi, full=True)
    return page.get_text(textpage=tp)


def parse_page(page: pymupdf.Page) -> Page:
    out = Page(number=page.number + 1)
    raw = page.get_text()

    if sum(c.isalnum() for c in raw) < settings.ocr_min_chars and page.get_images():
        text = _ocr_text(page)
        if text:
            out.blocks, out.ocr = [text], True
        return out

    tables = [t for t in page.find_tables().tables if t.row_count >= 2 and t.col_count >= 2]
    items = [(t.bbox[1], t.to_markdown()) for t in tables]
    for x0, y0, x1, y1, text, _no, kind in page.get_text("blocks", sort=True):
        if kind != 0 or not text.strip():  # kind 1 = image block
            continue
        if any(_inside((x0, y0, x1, y1), t.bbox) for t in tables):
            continue
        items.append((y0, text))

    items.sort(key=lambda it: it[0])  # top-to-bottom; tables slot in where they appear
    out.blocks = [text for _, text in items]
    out.n_tables = len(tables)
    return out


def parse_pdf(path: Path) -> list[Page]:
    with pymupdf.open(path) as doc:
        return [parse_page(p) for p in doc]
