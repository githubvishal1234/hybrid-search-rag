import random

from build_golden_set import fits_type, label, parse_json, sample_docs, type_plan, usable
from review_golden_set import validate

PAGES = {"a.pdf": {1, 2, 3, 7}, "b.pdf": {10}}


def item(**kw) -> dict:
    base = {
        "id": "d001",
        "question": "What is the LTV cap?",
        "type": "numeric",
        "expected_answer": "75%",
        "relevant": [{"doc": "a.pdf", "pages": [3]}],
        "origin": "llm_draft",
    }
    return base | kw


def test_type_plan_follows_the_mix():
    plan = type_plan(20)
    assert len(plan) == 20
    assert plan.count("exact_term") == 5 and plan.count("numeric") == 4
    assert plan.count("multi_hop") == 3 and plan.count("unanswerable") == 3


def test_sample_docs_flattens_big_documents():
    chunks = [{"doc_id": "big"}] * 900 + [{"doc_id": "small"}] * 100
    draws = sample_docs(chunks, 4000, random.Random(0))
    share = draws.count("small") / len(draws)
    assert 0.2 < share < 0.3  # sqrt weighting: 10/(30+10) = 25%, not the raw 10%


def test_usable_skips_forms_and_matches_type():
    prose = {"n_tokens": 300, "section_heading": "", "text": "The AMC shall disclose " * 30}
    form = {"n_tokens": 300, "section_heading": "", "text": "|Col1|Col2|\n|---|---|\n| | |" * 30}
    rescinded = {**prose, "section_heading": "APPENDIX: LIST OF RESCINDED CIRCULARS"}
    assert usable(prose, "paraphrase") and not usable(prose, "numeric")
    assert not usable(form, "paraphrase") and not usable(form, "numeric")
    assert not usable(rescinded, "exact_term")


def test_usable_skips_table_of_contents():
    toc = {
        "n_tokens": 300,
        "section_heading": "",
        "text": "Chapter 3 Schemes .......... 46\nChapter 4 Plans .......... 52\n"
        "Chapter 5 Risk .......... 60\nThe AMC shall disclose these matters " * 10,
    }
    assert not usable(toc, "paraphrase")


def test_fits_type():
    assert not fits_type("numeric", {"question": "Q", "answer": "the rate in column C"})
    assert fits_type("numeric", {"question": "Q", "answer": "2% above ₹10 lakh"})
    assert fits_type("exact_term", {"question": "What does section 148A require?", "answer": ""})
    assert fits_type("exact_term", {"question": "What is the TCS rate under LRS?", "answer": ""})
    assert not fits_type("exact_term", {"question": "What is the fiscal deficit?", "answer": ""})


def test_parse_json_tolerates_prose_and_fences():
    assert parse_json('Sure!\n```json\n{"question": "Q?", "answer": "A"}\n```') == {
        "question": "Q?",
        "answer": "A",
    }
    assert parse_json("no json here") is None
    assert parse_json('{"question": "broken",}') is None


def test_label_keeps_one_requirement_per_source_chunk():
    chunks = [
        {"file_name": "a.pdf", "page_start": 3, "page_end": 4},
        {"file_name": "a.pdf", "page_start": 40, "page_end": 40},
    ]
    assert label(chunks) == [{"doc": "a.pdf", "pages": [3, 4]}, {"doc": "a.pdf", "pages": [40]}]


def test_validate():
    assert validate(item(), PAGES) == []
    assert validate(item(type="unanswerable", expected_answer="", relevant=[]), PAGES) == []
    assert "unknown doc c.pdf" in validate(item(relevant=[{"doc": "c.pdf", "pages": [1]}]), PAGES)
    errors = validate(item(relevant=[{"doc": "a.pdf", "pages": [5]}]), PAGES)
    assert any("pages [5]" in e for e in errors)  # page 5 was dropped during cleaning
    assert validate(item(type="unanswerable"), PAGES)  # unanswerable can't have pages
    assert validate(item(relevant=[]), PAGES)  # answerable needs pages
    assert validate(item(type="trivia"), PAGES)
