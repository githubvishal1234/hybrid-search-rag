"""Ingestion entry point:
data/raw/*.pdf -> data/processed/pages.jsonl (one line per page)
               -> data/processed/chunks.jsonl (one line per chunk, what gets indexed)

Parsing is the slow step (table detection, ~0.1-0.3 s/page), so each PDF's raw parse is cached in
data/processed/parsed/<doc_id>.json together with the PDF's SHA-256. Unchanged PDFs are never
parsed twice; cleaning and chunking (fast, and what we iterate on) always rerun from the cache.

Run: uv run python -m hybrid_rag.ingestion.pipeline
"""

import csv
import hashlib
import itertools
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from tokenizers import Tokenizer

from hybrid_rag.config import settings
from hybrid_rag.ingestion.chunk import chunk_document
from hybrid_rag.ingestion.clean import clean_document
from hybrid_rag.ingestion.parse import Page, ocr_available, parse_pdf

DOC_FIELDS = ("doc_id", "file_name", "title", "publisher", "date", "source_url")

log = logging.getLogger(__name__)


def load_sources() -> dict[str, dict]:
    with settings.sources_csv.open(newline="", encoding="utf-8") as f:
        return {row["file_name"]: row for row in csv.DictReader(f)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_cached(pdf: Path) -> list[Page]:
    cache = settings.processed_dir / "parsed" / f"{pdf.stem}.json"
    digest = sha256(pdf)
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        # A parse made without Tesseract is redone once Tesseract is installed, so scanned
        # pages get their text; otherwise the cache would hide them forever.
        if data["sha256"] == digest and (data.get("ocr", False) or not ocr_available()):
            return [Page(**p) for p in data["pages"]]

    t0 = time.perf_counter()
    pages = parse_pdf(pdf)
    log.info("  parsed %s in %.1fs", pdf.name, time.perf_counter() - t0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sha256": digest, "ocr": ocr_available(), "pages": [asdict(p) for p in pages]}
    cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return pages


def parse_all() -> None:
    sources = load_sources()
    out_path = settings.processed_dir / "pages.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for pdf in sorted(settings.raw_dir.glob("*.pdf")):
            meta = sources.get(pdf.name, {})
            pages = parse_cached(pdf)
            cleaned = clean_document([p.blocks for p in pages])
            kept = 0
            for page, paragraphs in zip(pages, cleaned, strict=True):
                text = "\n\n".join(paragraphs)
                if not text:  # blank, separator, all-Hindi, or unreadable page
                    continue
                kept += 1
                record = {
                    "doc_id": pdf.stem,
                    "file_name": pdf.name,
                    "title": meta.get("title", pdf.stem),
                    "publisher": meta.get("publisher", ""),
                    "date": meta.get("date", ""),
                    "source_url": meta.get("source_url", ""),
                    "page": page.number,
                    "n_tables": page.n_tables,
                    "ocr": page.ocr,
                    "text": text,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.info("%-50s %4d/%4d pages kept", pdf.name, kept, len(pages))
    log.info("wrote %s", out_path)


def chunk_all() -> None:
    tokenizer = Tokenizer.from_pretrained(settings.embedding_model)

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    in_path = settings.processed_dir / "pages.jsonl"
    out_path = settings.processed_dir / "chunks.jsonl"
    with in_path.open(encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]

    n_total = 0
    with out_path.open("w", encoding="utf-8") as out:
        # pages.jsonl is written one document at a time, so groupby sees each doc once
        for doc_id, doc_pages in itertools.groupby(pages, key=lambda p: p["doc_id"]):
            doc_pages = list(doc_pages)
            meta = {k: doc_pages[0][k] for k in DOC_FIELDS}
            chunks = chunk_document(doc_pages, settings.chunk_size, settings.chunk_overlap, count)
            for c in chunks:
                record = {"chunk_id": f"{doc_id}:{c['chunk_index']:04d}", **meta, **c}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_total += len(chunks)
            log.info("%-50s %5d chunks", doc_id, len(chunks))
    log.info("wrote %d chunks to %s", n_total, out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parse_all()
    chunk_all()
