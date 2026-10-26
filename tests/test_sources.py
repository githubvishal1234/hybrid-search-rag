import csv

from hybrid_rag.config import settings


def test_sources_csv_is_well_formed():
    with settings.sources_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows, "sources.csv is empty"
    assert set(rows[0]) == {"file_name", "title", "source_url", "publisher", "date"}
    names = [r["file_name"] for r in rows]
    assert len(names) == len(set(names)), "duplicate file_name would overwrite a PDF"
    for r in rows:
        assert r["file_name"].endswith(".pdf"), r["file_name"]
        assert r["source_url"].startswith("https://"), r["source_url"]
        assert r["publisher"], r["file_name"]
