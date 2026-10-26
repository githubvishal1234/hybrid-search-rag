from hybrid_rag.indexing import bm25_index
from hybrid_rag.indexing.bm25_index import tokenize


def test_tokenize_keeps_identifiers_whole_and_split():
    assert tokenize("Form 26AS under section 10(23C)") == [
        "form", "26as", "section", "10(23c)", "10", "23c",
    ]  # fmt: skip
    toks = tokenize("Circular SEBI/HO/IMD/DF2/CIR/P/2021/668")
    assert "sebi/ho/imd/df2/cir/p/2021/668" in toks and "df2" in toks


def test_tokenize_normalises_numbers_and_words():
    assert tokenize("exceeds Rs. 50,00,000 at 10 per cent.") == [
        "exceed", "rs", "5000000", "10", "percent",
    ]  # fmt: skip
    assert tokenize("Loans to the AMCs") == ["loan", "amc"]  # stemmed, stopwords dropped
    assert tokenize("148A") == ["148a"]  # identifiers with digits are never stemmed


def chunk(i: int, text: str) -> dict:
    return {"chunk_id": f"d:{i}", "title": "Doc", "section_heading": "", "text": text}


def test_search_ranks_exact_term_first_and_skips_non_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(bm25_index, "path", lambda: tmp_path / "bm25.pkl")
    bm25_index.load.cache_clear()
    bm25_index.build(
        [
            chunk(0, "General provisions about reassessment of income."),
            chunk(1, "A notice under section 148A shall be issued before reassessment."),
            chunk(2, "Mutual funds shall follow digital accessibility guidelines."),
            chunk(3, "Rules for stock brokers and depositories."),
        ]
    )
    hits = bm25_index.search("section 148A notice", top_k=5)
    bm25_index.load.cache_clear()
    assert hits[0]["chunk_id"] == "d:1"
    assert all(h["score"] > 0 for h in hits)
    assert "d:2" not in [h["chunk_id"] for h in hits]  # shares no term with the query
