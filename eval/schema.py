"""Schema for eval/golden_set.jsonl. Every row must validate before an eval run counts.

Drafts (golden_set.draft.jsonl) don't follow it: they get `difficulty` and the review fields
only when a person keeps them in scripts/review_golden_set.py.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

QuestionType = Literal["exact_term", "paraphrase", "numeric", "multi_hop", "unanswerable"]
TYPES: tuple[str, ...] = QuestionType.__args__
# easy: the question uses the page's own words and one passage answers it.
# medium: reworded, or the figure sits in a table, or the question must pick one case of several.
# hard: needs 2+ passages, a calculation or a condition, or (unanswerable) a near-miss trap.
Difficulty = Literal["easy", "medium", "hard"]


class Relevant(BaseModel):
    """One requirement: a chunk from this doc overlapping these pages must be retrieved."""

    model_config = ConfigDict(extra="forbid")
    doc: str
    pages: list[int] = Field(min_length=1)


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelt field fails loudly

    id: str = Field(min_length=1)
    question: str = Field(min_length=5)
    type: QuestionType
    expected_answer: str
    relevant: list[Relevant]
    answerable: bool
    difficulty: Difficulty
    # Stored, not recomputed, so adding questions never moves old ones between halves.
    # dev: tune on it. test: the holdout, looked at only when reporting a finished phase.
    split: Literal["dev", "test"]
    smoke: bool = False  # in the 20-question CI subset (always from dev)
    origin: Literal["llm_draft", "claude_written", "hand"]
    reviewed: bool
    edited: bool
    reviewer: str | None = None  # absent = a person reviewed it
    review_note: str | None = None

    @model_validator(mode="after")
    def _consistent(self):
        if self.answerable != (self.type != "unanswerable"):
            raise ValueError("answerable must be False exactly when type is unanswerable")
        if self.answerable and not (self.relevant and self.expected_answer.strip()):
            raise ValueError("answerable questions need relevant pages and an expected_answer")
        if not self.answerable and self.relevant:
            raise ValueError("unanswerable questions have no relevant pages")
        if self.smoke and self.split != "dev":
            raise ValueError("smoke questions come from dev, never the test holdout")
        return self


def pick_split(qtype: str, existing: list[dict]) -> str:
    """For a new question: the half with fewer questions of its type (tie -> dev)."""
    n = {s: sum(q["type"] == qtype and q["split"] == s for q in existing) for s in ("dev", "test")}
    return "dev" if n["dev"] <= n["test"] else "test"


def load_golden(path: Path) -> list[GoldenItem]:
    """Parse and validate every row; raises on the first bad row or a duplicate id."""
    items, seen = [], set()
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = GoldenItem.model_validate_json(line)
            except ValueError as e:
                raise ValueError(f"{path.name} line {n}: {e}") from e
            if item.id in seen:
                raise ValueError(f"{path.name} line {n}: duplicate id {item.id}")
            seen.add(item.id)
            items.append(item)
    return items
