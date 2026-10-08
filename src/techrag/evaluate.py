"""Evaluate TechRAG retrieval against the labelled questions in ``eval/questions.jsonl``."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from techrag.retrieve import Result

QUESTIONS_PATH = Path("eval/questions.jsonl")


@dataclass(frozen=True)
class Question:
    """An eval question: the query, the document that answers it, and the answering text."""

    query: str
    source: Path  # relative to the data dir, like ``techrag.retrieve.Result.source``
    evidence: str  # text a retrieved chunk must contain to count as answering the query


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    """Load the questions in the JSON Lines file at ``path``, in file order.

    Blank lines are skipped. Each other line must be a JSON object with non-empty string
    ``query``, ``source`` and ``evidence`` fields; extra fields are ignored.

    Raises:
        FileNotFoundError: If ``path`` is not an existing file.
        ValueError: If a line is not valid JSON or lacks a valid field. The message
            names the offending line number.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Questions file not found: {path}")
    questions = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {e.msg}") from e
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{lineno}: expected a JSON object")
        for field in ("query", "source", "evidence"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{path}:{lineno}: '{field}' must be a non-empty string")
        questions.append(
            Question(record["query"], Path(record["source"]), record["evidence"])
        )
    return questions


def _normalize(text: str) -> str:
    """Lowercase ``text`` and collapse whitespace, so line wrapping does not affect matching."""
    return " ".join(text.lower().split())


def first_relevant_rank(question: Question, results: Sequence[Result]) -> int | None:
    """Return the 1-based rank of the first result that answers ``question``, or ``None``.

    A result answers the question if it comes from the question's source and contains its
    evidence, ignoring case and differences in whitespace.
    """
    evidence = _normalize(question.evidence)
    for rank, r in enumerate(results, start=1):
        if r.source == question.source and evidence in _normalize(r.text):
            return rank
    return None
