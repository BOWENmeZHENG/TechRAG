"""Evaluate TechRAG retrieval against the labelled questions in ``eval/questions.jsonl``."""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from techrag.ingest import EMBEDDING_MODEL, INDEX_DIR
from techrag.metrics import hit_at_k, mean_reciprocal_rank
from techrag.retrieve import Result, retrieve

QUESTIONS_PATH = Path("eval/questions.jsonl")
RESULTS_PATH = Path("eval/results.json")


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


def evaluate(
    questions: Sequence[Question],
    k: int = 5,
    retriever: Callable[[str, int], Sequence[Result]] = retrieve,
) -> list[int | None]:
    """Return, for each question, the rank of its first relevant result among the top ``k``.

    The ranks are in question order and feed ``techrag.metrics``; ``None`` marks a question
    with no relevant result in the top ``k``. ``retriever`` takes a query and ``k`` and returns
    results best first; it defaults to :func:`techrag.retrieve.retrieve` on the default index.

    Raises:
        ValueError: If ``retriever`` rejects ``k`` or a query (the default does).
        FileNotFoundError: If the default retriever finds no index.
    """
    return [first_relevant_rank(q, retriever(q.query, k)) for q in questions]


def main() -> None:
    """Entry point for the ``techrag-eval`` command."""
    parser = argparse.ArgumentParser(description="Evaluate TechRAG retrieval.")
    parser.add_argument("-k", type=int, default=5, help="number of chunks to retrieve per query")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="must match the ingest model")
    parser.add_argument(
        "--output", type=Path, default=RESULTS_PATH, help="JSON file to write the results to"
    )
    args = parser.parse_args()

    try:
        questions = load_questions(args.questions)
        retrieve_from_index = partial(retrieve, index_dir=args.index_dir, model_name=args.model)
        retrieved: list[Sequence[Result]] = []  # per question, for the report

        def retriever(query: str, k: int) -> Sequence[Result]:
            results = retrieve_from_index(query, k)
            retrieved.append(results)
            return results

        ranks = evaluate(questions, args.k, retriever)
        hit = hit_at_k(ranks, args.k)
        mrr = mean_reciprocal_rank(ranks)
        report = {
            "k": args.k,
            "model": args.model,
            "hit_at_k": hit,
            "mrr": mrr,
            "questions": [
                {
                    "query": q.query,
                    "source": str(q.source),
                    "rank": rank,
                    "retrieved": [
                        {"source": str(r.source), "score": r.score, "text": r.text}
                        for r in results
                    ],
                }
                for q, rank, results in zip(questions, ranks, retrieved, strict=True)
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")

    for q, rank in zip(questions, ranks, strict=True):
        print(f"{'miss' if rank is None else f'rank {rank}':>7}  {q.query}")
    print(f"\n{len(questions)} questions")
    print(f"hit@{args.k}: {hit:.3f}")
    print(f"MRR:    {mrr:.3f}")


if __name__ == "__main__":
    main()
