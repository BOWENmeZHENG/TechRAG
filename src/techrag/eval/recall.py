"""Document-level retrieval recall on the EnterpriseRAG-Bench questions of a subset.

A question is answered from its ``expected_doc_ids``. Retrieval returns chunks, so a chunk
counts toward a document when its ``source`` maps to that document's id (see
:func:`techrag.benchmarks.bench.doc_id_of`). Recall is measured at several cutoffs ``k`` over
*chunks*, because that is how many passages an answering model gets to read.

Questions with no expected documents (the unanswerable and high-level types) have nothing to
retrieve, so they are skipped.
"""

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from techrag.benchmarks.bench import SUBSET_DIR, BenchQuestion, doc_id_of, read_questions
from techrag.core.ingest import EMBEDDING_MODEL
from techrag.core.retrieve import Mode, Result, Retriever

KS = (5, 10, 20, 50, 100)
INDEX_DIR = Path(".index/dev")
RESULTS_PATH = Path("eval/recall.json")


@dataclass(frozen=True)
class QuestionRanks:
    """Where each of a question's gold documents first appeared in the retrieved chunks.

    ``ranks`` maps every expected document id to the 1-based rank of the first retrieved chunk
    from that document, or ``None`` if none of the retrieved chunks came from it.

    Attributes:
        question: The benchmark question.
        ranks: Rank of each expected document id, or ``None`` if it was not retrieved.
    """

    question: BenchQuestion
    ranks: dict[str, int | None]

    def recall(self, k: int) -> float:
        """Return the fraction of the gold documents found within the top ``k`` chunks.

        Args:
            k: Number of top-ranked chunks to consider.

        Returns:
            A value in ``[0, 1]``.
        """
        found = sum(r is not None and r <= k for r in self.ranks.values())
        return found / len(self.ranks)

    def complete(self, k: int) -> bool:
        """Return whether every gold document was found within the top ``k`` chunks.

        Args:
            k: Number of top-ranked chunks to consider.

        Returns:
            ``True`` if no gold document is missing from the top ``k`` chunks.
        """
        return all(r is not None and r <= k for r in self.ranks.values())


def gold_doc_ranks(
    expected_doc_ids: Sequence[str], results: Sequence[Result]
) -> dict[str, int | None]:
    """Return the 1-based rank of the first chunk of each expected document in ``results``.

    Args:
        expected_doc_ids: Ids of the documents that answer the question.
        results: Retrieved chunks, best first.

    Returns:
        A map from each expected document id to the rank of its first chunk in ``results``,
        or ``None`` if none of its chunks were retrieved.
    """
    ranks: dict[str, int | None] = dict.fromkeys(expected_doc_ids)
    for rank, result in enumerate(results, start=1):
        doc_id = doc_id_of(result.source)
        if doc_id in ranks and ranks[doc_id] is None:
            ranks[doc_id] = rank
    return ranks


def evaluate(
    questions: Sequence[BenchQuestion],
    retriever: Callable[[str, int], Sequence[Result]],
    k: int = max(KS),
) -> list[QuestionRanks]:
    """Retrieve ``k`` chunks for every question that has gold documents and locate those documents.

    ``retriever`` takes a query and a chunk count and returns chunks best first. Questions
    without expected documents are left out of the result.

    Args:
        questions: Benchmark questions to evaluate.
        retriever: Function taking a query and a chunk count and returning chunks, best first.
        k: Number of chunks to retrieve per question.

    Returns:
        One :class:`QuestionRanks` per question that has expected documents, in input order.
    """
    return [
        QuestionRanks(q, gold_doc_ranks(q.expected_doc_ids, retriever(q.text, k)))
        for q in questions
        if q.expected_doc_ids
    ]


def summarize(rows: Sequence[QuestionRanks], ks: Sequence[int] = KS) -> dict:
    """Return the mean recall and the share of fully answered questions at each cutoff.

    Args:
        rows: Per-question ranks from :func:`evaluate`.
        ks: Cutoffs, in chunks, to report at.

    Returns:
        A dict of the shape ``{"n": int, "recall": {k: float}, "complete": {k: float}}``.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError("rows must not be empty")
    return {
        "n": len(rows),
        "recall": {k: sum(r.recall(k) for r in rows) / len(rows) for k in ks},
        "complete": {k: sum(r.complete(k) for r in rows) / len(rows) for k in ks},
    }


def summarize_by_type(rows: Sequence[QuestionRanks], ks: Sequence[int] = KS) -> dict[str, dict]:
    """Return :func:`summarize` for each question type, ordered by type name.

    Args:
        rows: Per-question ranks from :func:`evaluate`.
        ks: Cutoffs, in chunks, to report at.

    Returns:
        A map from question type to its :func:`summarize` result.
    """
    groups: dict[str, list[QuestionRanks]] = defaultdict(list)
    for row in rows:
        groups[row.question.type].append(row)
    return {t: summarize(groups[t], ks) for t in sorted(groups)}


def format_table(overall: dict, by_type: dict[str, dict], ks: Sequence[int] = KS) -> str:
    """Render the overall and per-type summaries as a text table (mean recall, then complete).

    Args:
        overall: The :func:`summarize` result for all questions.
        by_type: The :func:`summarize_by_type` result.
        ks: Cutoffs to show as columns; each must be present in the summaries.

    Returns:
        The table as a multi-line string with one row per type and a final ``ALL`` row.
    """
    header = f"{'type':<26}{'n':>5}  " + "".join(f"{f'R@{k}':>8}" for k in ks)
    header += "  " + "".join(f"{f'all@{k}':>8}" for k in ks)
    lines = [header]
    for name, s in [*by_type.items(), ("ALL", overall)]:
        line = f"{name:<26}{s['n']:>5}  " + "".join(f"{s['recall'][k]:>8.3f}" for k in ks)
        line += "  " + "".join(f"{s['complete'][k]:>8.3f}" for k in ks)
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    """Entry point for the ``techrag-recall`` command."""
    parser = argparse.ArgumentParser(description="Measure document recall on benchmark questions.")
    parser.add_argument("--subset", type=Path, default=SUBSET_DIR, help="subset with questions")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="must match the ingest model")
    parser.add_argument("--mode", choices=get_args(Mode), default="dense", help="ranking method")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH, help="JSON report to write")
    args = parser.parse_args()

    try:
        questions = read_questions(args.subset / "questions.jsonl")
        retriever = Retriever(args.index_dir, args.model)
        retriever.warm_up()
        rows = evaluate(questions, lambda q, k: retriever.search(q, k, args.mode))
        overall = summarize(rows)
        by_type = summarize_by_type(rows)
        report = {
            "mode": args.mode,
            "model": args.model,
            "ks": list(KS),
            "skipped_no_gold": len(questions) - len(rows),
            "overall": overall,
            "by_type": by_type,
            "questions": [
                {"id": r.question.id, "type": r.question.type, "ranks": r.ranks} for r in rows
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")

    print(format_table(overall, by_type))
    print(f"\n{len(questions) - len(rows)} questions without gold documents skipped")


if __name__ == "__main__":
    main()
