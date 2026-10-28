"""Deploy to a Hugging Face Space (Docker SDK): code + prebuilt indexes, no PDFs.

    uv run python scripts/deploy_space.py --space you/hybrid-rag --dry-run   # show what ships
    uv run python scripts/deploy_space.py --space you/hybrid-rag \
        --var LLM_PROVIDER=gemini --var LLM_MODEL=<model> --var RERANK_CANDIDATES=15

Needs a write token: `hf auth login` or HF_TOKEN in the environment.
API keys are NOT uploaded by this script: add them in the Space under
Settings -> Variables and secrets (e.g. GEMINI_API_KEY) so they never touch the repo.

Why ship the embedded Qdrant store instead of Qdrant Cloud: the whole index is ~25 MB, so it
fits in the image and the demo has no external database to keep alive. Set QDRANT_URL (plus
QDRANT_API_KEY as a secret) to switch to Qdrant Cloud later without code changes.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from hybrid_rag.config import ROOT

# Tracked files the Space doesn't need (eval and tests aren't run there).
SKIP_PREFIXES = ("eval/", "tests/", ".github/", ".claude/", "data/raw/")
# Built locally by `make ingest index`, gitignored, but required at runtime.
INDEX_FILES = ("data/processed/chunks.jsonl", "data/processed/bm25.pkl")
INDEX_DIRS = ("data/processed/qdrant",)

SPACE_HEADER = """---
title: Hybrid Search RAG (Indian finance)
emoji: 📑
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: BM25 + vector RAG with RRF and reranking, cited answers
---

"""


def keep(path: str) -> bool:
    """Ship it to the Space? Not eval/tests/CI files, not Qdrant's process lock (but uv.lock)."""
    return not path.startswith(SKIP_PREFIXES) and Path(path).name != ".lock"


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def stage(dest: Path) -> list[Path]:
    missing = [p for p in (*INDEX_FILES, *INDEX_DIRS) if not (ROOT / p).exists()]
    if missing:
        sys.exit(f"missing {missing}: run `make ingest index` first")

    files = [*tracked_files(), *INDEX_FILES]
    for d in INDEX_DIRS:
        files += [p.relative_to(ROOT).as_posix() for p in (ROOT / d).rglob("*") if p.is_file()]
    files = [f for f in files if keep(f)]

    for f in files:
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest / f)
    readme = dest / "README.md"
    readme.write_text(SPACE_HEADER + readme.read_text(encoding="utf-8"), encoding="utf-8")
    return [dest / f for f in files]


def parse_vars(pairs: list[str]) -> list[dict]:
    out = []
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            sys.exit(f"--var expects KEY=VALUE, got {pair!r}")
        if "KEY" in key.upper() or "TOKEN" in key.upper():
            sys.exit(f"{key} looks like a secret; add it in the Space settings, not with --var")
        out.append({"key": key, "value": value})
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--space", required=True, help="owner/name of the Space")
    p.add_argument("--var", action="append", default=[], help="non-secret Space variable K=V")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="stage and list files, upload nothing")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    variables = parse_vars(args.var)

    with tempfile.TemporaryDirectory() as tmp:
        staged = stage(Path(tmp))
        size = sum(f.stat().st_size for f in staged)
        print(f"{len(staged)} files, {size / 1e6:.1f} MB")
        if args.dry_run:
            for f in sorted(staged):
                print(f"  {f.relative_to(tmp).as_posix()}")
            return 0

        from huggingface_hub import HfApi

        api = HfApi()
        url = api.create_repo(
            args.space,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
        )
        for v in variables:
            api.add_space_variable(args.space, v["key"], v["value"])
        # delete_patterns="*": the Space mirrors this upload, so removed files don't linger.
        api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=tmp,
            delete_patterns="*",
            commit_message="Deploy from scripts/deploy_space.py",
        )
    print(f"uploaded to {url}. The Space now builds the Docker image (~10 min the first time).")
    print("Add your LLM API key under Settings -> Variables and secrets if you haven't yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
