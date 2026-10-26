"""Review LLM drafts by hand, and add hand-written questions. Only reviewed items reach
eval/golden_set.jsonl, which is what the eval reads.

uv run python scripts/review_golden_set.py          # walk the drafts: keep / edit / drop
uv run python scripts/review_golden_set.py --add    # write your own questions from the PDFs
uv run python scripts/review_golden_set.py --check  # validate golden_set.jsonl, print the mix

Progress is saved after every decision, so you can quit and resume any time.

What to check for each draft (this review is the part to talk about in interviews):
- Is the question something a real user would ask, and unambiguous?
- Is the expected answer correct and complete according to the PDF? Open the page.
- Are the pages right? Add any other page that also answers it.
- Is the type right? exact_term must contain an identifier; paraphrase must NOT reuse the
  passage's distinctive words.
- unanswerable: is it really not answered anywhere? The top search hits are shown to help.
"""

import argparse
import json
import sys
from collections import Counter

from hybrid_rag.config import ROOT, settings

DRAFT_PATH = ROOT / "eval" / "golden_set.draft.jsonl"
GOLDEN_PATH = ROOT / "eval" / "golden_set.jsonl"
DROPPED_PATH = ROOT / "eval" / "golden_set.dropped.jsonl"
TYPES = ("exact_term", "paraphrase", "numeric", "multi_hop", "unanswerable")
FIELDS = ("id", "question", "type", "expected_answer", "relevant", "origin")


def read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def kept_pages() -> dict[str, set[int]]:
    """file_name -> pages that survived cleaning (a label pointing elsewhere can never be hit)."""
    out: dict[str, set[int]] = {}
    with (settings.processed_dir / "pages.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.setdefault(r["file_name"], set()).add(r["page"])
    return out


def validate(item: dict, pages: dict[str, set[int]]) -> list[str]:
    errors = [f"missing {k}" for k in FIELDS if k not in item]
    if errors:
        return errors
    if item["type"] not in TYPES:
        errors.append(f"type must be one of {TYPES}")
    if not item["question"].strip():
        errors.append("empty question")
    if item["type"] == "unanswerable":
        if item["relevant"]:
            errors.append("unanswerable questions have no relevant pages")
    else:
        if not item["relevant"]:
            errors.append("answerable questions need at least one relevant (doc, pages)")
        if not item["expected_answer"].strip():
            errors.append("answerable questions need an expected_answer")
        for r in item["relevant"]:
            if r["doc"] not in pages:
                errors.append(f"unknown doc {r['doc']}")
            elif bad := [p for p in r["pages"] if p not in pages[r["doc"]]]:
                errors.append(f"{r['doc']}: pages {bad} not in the parsed corpus")
    return errors


def ask(prompt: str, default: str = "") -> str:
    shown = f" [{default}]" if default else ""
    reply = input(f"{prompt}{shown}: ").strip()
    return reply or default


def ask_relevant(current: list[dict]) -> list[dict]:
    """'file.pdf:3,4; other.pdf:12' <-> [{"doc": "file.pdf", "pages": [3, 4]}, ...]"""
    shown = "; ".join(f"{r['doc']}:{','.join(map(str, r['pages']))}" for r in current)
    reply = ask("relevant (doc:pages; ...), '-' for none", shown)
    if reply == "-":
        return []
    out = []
    for part in filter(None, (p.strip() for p in reply.split(";"))):
        doc, _, pages = part.partition(":")
        out.append({"doc": doc.strip(), "pages": [int(p) for p in pages.split(",") if p.strip()]})
    return out


def edit(item: dict) -> dict:
    item = dict(item)
    item["question"] = ask("question", item["question"])
    item["type"] = ask(f"type {TYPES}", item["type"])
    if item["type"] == "unanswerable":
        item["expected_answer"], item["relevant"] = "", []
    else:
        item["expected_answer"] = ask("expected answer", item["expected_answer"])
        item["relevant"] = ask_relevant(item["relevant"])
    return item


def show(item: dict, page_text: dict[tuple[str, int], str]) -> None:
    print("\n" + "=" * 100)
    print(f"{item['id']}  [{item['type']}]")
    print(f"Q: {item['question']}")
    print(f"A: {item['expected_answer'] or '(unanswerable)'}")
    for r in item["relevant"]:
        pdf = settings.raw_dir / r["doc"]
        print(f"   {r['doc']} pages {r['pages']}   {pdf.as_uri()}#page={r['pages'][0]}")
    # Show the labelled pages, not the source chunks: chunk ids change whenever chunking does,
    # pages don't (same reason the labels are pages).
    for r in item["relevant"]:
        text = "\n".join(page_text.get((r["doc"], p), "") for p in r["pages"])
        print(f"\n--- {r['doc']} p.{','.join(map(str, r['pages']))} ---")
        print(text[:1500] + (" [...]" if len(text) > 1500 else ""))
    if item["type"] == "unanswerable":
        from hybrid_rag.retrieval.retriever import retrieve  # loads models; only needed here

        print("\n--- top hybrid hits (check none of them answers it) ---")
        for h in retrieve(item["question"], "hybrid", 3):
            print(f"  {h['chunk_id']} p.{h['page_start']}: {h['text'][:200]!r}")


def review() -> None:
    page_text = {}
    with (settings.processed_dir / "pages.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            page_text[(r["file_name"], r["page"])] = r["text"]
    pages = kept_pages()
    done = {d["id"] for d in read_jsonl(GOLDEN_PATH) + read_jsonl(DROPPED_PATH)}
    todo = [d for d in read_jsonl(DRAFT_PATH) if d["id"] not in done]
    print(f"{len(todo)} drafts to review ({len(done)} already done)")

    for n, item in enumerate(todo, start=1):
        show(item, page_text)
        while True:
            choice = ask(f"\n({n}/{len(todo)}) [k]eep  [e]dit  [d]rop  [q]uit", "").lower()
            if choice == "q":
                return
            if choice == "d":
                reason = ask("why drop (helps you spot patterns)", "")
                append_jsonl(DROPPED_PATH, {**item, "drop_reason": reason})
                break
            if choice in ("k", "e"):
                final = edit(item) if choice == "e" else item
                final = {k: final[k] for k in FIELDS} | {"reviewed": True, "edited": choice == "e"}
                if errors := validate(final, pages):
                    print("  ! " + "; ".join(errors) + " - try [e]dit again")
                    continue
                append_jsonl(GOLDEN_PATH, final)
                break


def add() -> None:
    pages = kept_pages()
    existing = read_jsonl(GOLDEN_PATH)
    n = sum(1 for g in existing if g["origin"] == "hand")
    print("Write questions from the PDFs themselves (not from chunks). Docs:")
    for doc in sorted(pages):
        print(f"  {doc}")
    while True:
        print()
        q = ask("question (empty to stop)")
        if not q:
            return
        n += 1
        item = edit(
            {"id": f"h{n:03d}", "question": q, "type": "", "expected_answer": "", "relevant": []}
        )
        item = {**item, "origin": "hand", "reviewed": True, "edited": False}
        if errors := validate(item, pages):
            print("  ! not saved: " + "; ".join(errors))
            n -= 1
            continue
        append_jsonl(GOLDEN_PATH, item)


def check() -> int:
    pages = kept_pages()
    items = read_jsonl(GOLDEN_PATH)
    bad = 0
    for item in items:
        if errors := validate(item, pages):
            bad += 1
            print(f"{item.get('id')}: {'; '.join(errors)}")
    ids = Counter(i["id"] for i in items)
    for dup in [i for i, c in ids.items() if c > 1]:
        bad += 1
        print(f"duplicate id {dup}")
    total = max(len(items), 1)
    types = Counter(i["type"] for i in items)
    hand = sum(i["origin"] == "hand" for i in items)
    print(f"\n{len(items)} questions, {bad} invalid, {hand} hand-written ({hand / total:.0%})")
    for t in TYPES:
        print(f"  {t:13} {types[t]:3}  ({types[t] / total:.0%})")
    return 1 if bad else 0


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--add", action="store_true")
    g.add_argument("--check", action="store_true")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.add:
        add()
    elif not args.check:
        review()
    return check()


if __name__ == "__main__":
    sys.exit(main())
