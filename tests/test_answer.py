from hybrid_rag.generation.answer import resolve_citations
from hybrid_rag.generation.prompts import format_sources
from hybrid_rag.indexing.vector_store import point_id


def chunk(i: int, start: int, end: int) -> dict:
    return {
        "chunk_id": f"doc{i}:000{i}",
        "file_name": f"doc{i}.pdf",
        "title": f"Doc {i}",
        "section_heading": "",
        "page_start": start,
        "page_end": end,
        "text": f"text of chunk {i}",
    }


CHUNKS = [chunk(1, 3, 3), chunk(2, 7, 8)]


def test_valid_citations_become_file_and_page():
    raw = "LTV is 75% [1]. Tenor is 12 months [2][1]."
    text, citations, invalid = resolve_citations(raw, CHUNKS)
    assert text == (
        "LTV is 75% [doc1.pdf, p.3]. Tenor is 12 months [doc2.pdf, p.7-8][doc1.pdf, p.3]."
    )
    assert [c["chunk_id"] for c in citations] == ["doc1:0001", "doc2:0002"]  # deduped, in order
    assert invalid == []


def test_hallucinated_citation_is_dropped_and_reported():
    text, citations, invalid = resolve_citations("Rate is 20% [3].", CHUNKS)
    assert text == "Rate is 20%."
    assert citations == []
    assert invalid == [3]


def test_sources_are_numbered_with_page_spans():
    s = format_sources(CHUNKS)
    assert s.startswith("[1] Doc 1 (p.3)\ntext of chunk 1")
    assert "[2] Doc 2 (p.7-8)" in s


def test_point_id_is_stable():
    assert point_id("doc1:0001") == point_id("doc1:0001") != point_id("doc1:0002")
