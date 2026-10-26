from hybrid_rag.ingestion.clean import clean_block, clean_document, non_english_ratio


def test_non_english_ratio():
    assert non_english_ratio("Know Your Customer") == 0
    assert non_english_ratio("बजट अनुमान") == 1
    assert non_english_ratio("ºÉiªÉàÉä´É VÉªÉiÉä") > 0.3  # legacy Hindi font junk
    assert non_english_ratio("₹ 50,000 crore") == 0  # rupee sign is not a letter


def test_clean_block_unwraps_lines_and_drops_hindi():
    block = "The LTV ratio shall not\n  exceed 75   per cent.\nबजट अनुमान\n"
    assert clean_block(block) == "The LTV ratio shall not exceed 75 per cent."


def test_clean_block_keeps_hyphenated_compounds():
    assert clean_block("in case of non-\ncompliance with sub-\nsection (2)") == (
        "in case of non-compliance with sub-section (2)"
    )


def test_clean_block_drops_chart_label_soup():
    assert clean_block("9.2%\n7.3% 6.7% 2017-18 2018-19") == ""
    assert clean_block("Fiscal deficit is 4.4% of GDP") != ""


def test_clean_table_strips_markdown_artifacts_and_hindi():
    table = "|Item|**2024-25**<br>**वास्तविक**|\n|---|---|\n|(_a_) Revenue<br>Receipts|3036619|"
    assert clean_block(table) == "|Item|2024-25|\n|---|---|\n|(a) Revenue Receipts|3036619|"


def test_clean_document_removes_running_headers_and_page_numbers():
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    pages = [
        [f"Page {i} of 10", "Master Circular as on March 20, 2026", f"Clause on {w}.", str(i)]
        for i, w in enumerate(words, start=1)
    ]
    cleaned = clean_document(pages)
    assert len(cleaned) == 10  # page count preserved, so page numbers stay correct
    assert cleaned[4] == ["Clause on epsilon."]


def test_dedupe_cells_collapses_merged_cell_copies_but_keeps_numbers():
    from hybrid_rag.ingestion.clean import dedupe_cells

    merged = "|241|NH-703 road near Moga|NH-703 road near Moga|NH-703 road near Moga|12.5|"
    assert dedupe_cells(merged) == "|241|NH-703 road near Moga|12.5|"
    assert dedupe_cells("|Col1|Item|Col3|Col4|2025-26|") == "|Item||2025-26|"
    assert dedupe_cells("|Total Receipts|4652867|4652867|") == "|Total Receipts|4652867|4652867|"
