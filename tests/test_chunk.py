import pytest

from hybrid_rag.ingestion.chunk import chunk_document, is_heading


def words(text: str) -> int:  # stand-in for the BGE tokenizer: 1 word = 1 token
    return len(text.split())


def sentence(i: int) -> str:
    return f"Sentence number {i} talks about the KYC rules for regulated entities."  # 11 words


def test_is_heading():
    for h in [
        "15.21. Guidelines for Digital Accessibility",
        "ANNEXURE 16",
        "Annexure 17A",
        "CHAPTER IV",
        "BUSINESS RESPONSIBILITY & SUSTAINABILITY REPORTING FORMAT",
        "7. In section 144C of the Income-tax Act,––",
    ]:
        assert is_heading(h), h
    for p in [
        "9. Therefore, rationalisation of TCS rates is proposed as follows:—",
        "section 172 apply.",
        "Part of the income is exempt",
        "5. The tabular format for disclosure of holding of specified securities is as follows:-",
        "|Col|Col|\n|---|---|\n|a|b|",
    ]:
        assert not is_heading(p), p


def test_chunks_respect_size_and_cover_all_text():
    pages = [{"page": 1, "text": " ".join(sentence(i) for i in range(40))}]
    chunks = chunk_document(pages, size=50, overlap=12, count=words)
    assert all(words(c["text"]) <= 50 for c in chunks)
    joined = " ".join(c["text"] for c in chunks)
    assert all(sentence(i) in joined for i in range(40))


def test_overflow_chunks_overlap():
    pages = [{"page": 1, "text": " ".join(sentence(i) for i in range(20))}]
    a, b = chunk_document(pages, size=50, overlap=12, count=words)[:2]
    last_sentence_of_a = a["text"].split(". ")[-1]
    assert b["text"].startswith(last_sentence_of_a.rstrip("."))


def test_heading_closes_chunk_and_labels_the_next():
    body = " ".join(sentence(i) for i in range(3))  # 33 words, over size // 4
    pages = [{"page": 1, "text": f"{body}\n\n3.8.1. Equity Schemes\n\n{sentence(99)}"}]
    first, second = chunk_document(pages, size=100, overlap=12, count=words)
    assert "Equity Schemes" not in first["text"]
    assert second["text"].startswith("3.8.1. Equity Schemes")
    assert second["section_heading"] == "3.8.1. Equity Schemes"
    assert not second["text"].count(sentence(2))  # no overlap carried across a heading


def test_chunk_spanning_pages_records_both():
    pages = [{"page": 4, "text": sentence(1)}, {"page": 5, "text": sentence(2)}]
    (only,) = chunk_document(pages, size=100, overlap=12, count=words)
    assert (only["page_start"], only["page_end"]) == (4, 5)


def test_big_table_is_split_with_header_repeated_once_per_chunk():
    header = "|Item|2025-26 BE|\n|---|---|"
    rows = "\n".join(f"|Scheme {i} outlay|{i}00|" for i in range(30))  # 3 words per row
    pages = [{"page": 2, "text": f"{header}\n{rows}"}]
    chunks = chunk_document(pages, size=40, overlap=0, count=words)
    assert len(chunks) > 1
    for c in chunks:
        assert c["text"].startswith(header)
        assert c["text"].count("2025-26 BE") == 1
    joined = "\n".join(c["text"] for c in chunks)
    assert all(f"|Scheme {i} outlay|" in joined for i in range(30))


@pytest.mark.parametrize("n_words", [7, 300])
def test_oversized_sentence_is_windowed(n_words):
    pages = [{"page": 1, "text": " ".join(["clause"] * n_words)}]
    chunks = chunk_document(pages, size=20, overlap=0, count=words)
    assert all(words(c["text"]) <= 20 for c in chunks)
    assert sum(words(c["text"]) for c in chunks) == n_words
