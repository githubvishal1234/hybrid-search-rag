"""BM25 keyword index over chunks, with a tokenizer built for legal/financial text.

BM25 scores a chunk by how often the query's words appear in it, weighted by how rare each word
is across the corpus (IDF) and normalised for chunk length. Rare exact terms like "148A" or a
circular number therefore dominate, which is exactly where embeddings are weak: bge's own
tokenizer splits "26AS" into "26" + "##as" and blurs it with every other number.

Tokenizer rules:
- Compound identifiers stay whole AND are split into parts:
  "10(23C)" -> "10(23c)", "10", "23c"; "SEBI/HO/IMD/..." -> the whole string plus "sebi", "ho"...
  The whole form matches an exact citation strongly; the parts still match a looser query.
- "50,00,000" -> "5000000" (Indian digit grouping), "per cent" -> "percent".
- Plain words are lowercased, stopwords dropped, and stemmed ("loans" -> "loan"); identifiers
  containing digits are never stemmed.
"""

import pickle
import re
from functools import cache

import numpy as np
import Stemmer
from rank_bm25 import BM25Okapi

from hybrid_rag.config import settings

# A token is a run of letters/digits, optionally joined to more runs by . / ( ) - _ &
TOKEN = re.compile(r"[a-z0-9]+(?:[./()\-_&]+[a-z0-9]+)*\)?")
PART = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was "
    "were will with shall such any which been being under said".split()
)
_stemmer = Stemmer.Stemmer("english")


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("per cent", "percent")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)  # 50,00,000 -> 5000000
    tokens = []
    for tok in TOKEN.findall(text):
        parts = PART.findall(tok)
        if len(parts) > 1:  # compound identifier: keep the whole form, then its parts
            tokens.append(tok)
        for p in parts:
            if p in STOPWORDS:
                continue
            tokens.append(p if any(c.isdigit() for c in p) else _stemmer.stemWord(p))
    return tokens


def index_text(chunk: dict) -> str:
    return f"{chunk['title']} {chunk['section_heading']} {chunk['text']}"


def path():
    return settings.processed_dir / "bm25.pkl"


def build(chunks: list[dict]) -> None:
    bm25 = BM25Okapi([tokenize(index_text(c)) for c in chunks])
    with path().open("wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)


@cache
def load() -> dict:
    with path().open("rb") as f:  # our own file, written by build(); never load untrusted pickles
        return pickle.load(f)


def search(query: str, top_k: int) -> list[dict]:
    """Chunks best first, each with its BM25 "score". Chunks sharing no term with the query
    (score 0) are left out, so a miss returns fewer results instead of random ones."""
    idx = load()
    scores = idx["bm25"].get_scores(tokenize(query))
    top = np.argsort(-scores)[:top_k]  # ponytail: scores every chunk; fine below ~1M chunks
    return [{**idx["chunks"][i], "score": float(scores[i])} for i in top if scores[i] > 0]
