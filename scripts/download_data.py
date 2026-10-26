"""Download every PDF listed in data/sources.csv into data/raw/.

Re-running is safe: files that already exist (and are real PDFs) are skipped.
Government sites often block default Python clients and sometimes return an HTML error page
with status 200, so we send a browser User-Agent, retry, and check the %PDF magic bytes.
"""

import csv
import sys
import time
import urllib.request
from pathlib import Path

from hybrid_rag.config import settings

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
RETRIES = 3


def is_pdf(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(5) == b"%PDF-"


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if not data.startswith(b"%PDF-"):
                raise ValueError("response is not a PDF (probably an HTML error page)")
            tmp = dest.with_suffix(".part")  # write then rename, so a crash never leaves half a PDF
            tmp.write_bytes(data)
            tmp.replace(dest)
            return
        except Exception as e:
            if attempt == RETRIES:
                raise
            print(f"  retry {attempt}/{RETRIES - 1} after error: {e}")
            time.sleep(2**attempt)


def main() -> int:
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    with settings.sources_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    failed = []
    for row in rows:
        dest = settings.raw_dir / row["file_name"]
        if dest.exists() and is_pdf(dest):
            print(f"skip  {row['file_name']}")
            continue
        print(f"get   {row['file_name']}")
        try:
            download(row["source_url"], dest)
            print(f"  ok  {dest.stat().st_size / 1e6:.1f} MB")
        except Exception as e:
            print(f"  FAIL {e}")
            failed.append(row["file_name"])

    print(f"\n{len(rows) - len(failed)}/{len(rows)} PDFs in {settings.raw_dir}")
    if failed:
        print("failed:", ", ".join(failed))
        # rbidocs.rbi.org.in serves a JavaScript bot challenge to scripts; a browser gets the PDF.
        print(f"Open those URLs in a browser and save them into {settings.raw_dir} under the")
        print("file_name from sources.csv. Re-running this script will then skip them.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
