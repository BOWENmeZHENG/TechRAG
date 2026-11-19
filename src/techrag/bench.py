"""Load EnterpriseRAG-Bench and build a small development subset of its corpus.

The full corpus has ~512k documents. ``techrag-subset`` writes every document that answers a
question plus a seeded random sample of the rest (the distractors) as Markdown files laid out
as ``<source_type>/<doc_id>.md``, so ``techrag-ingest`` can index them unchanged and a retrieved
chunk's ``source`` identifies its document (see :func:`doc_id_of`).
"""

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

BENCH_DIR = Path("data/external/EnterpriseRAG-Bench")
SUBSET_DIR = Path("data/subsets/dev")

# Written next to the documents so a subset records how it was built.
MANIFEST = "manifest.json"
QUESTIONS = "questions.jsonl"
# Separates a document id from the counter that disambiguates rows sharing that id.
_DUPLICATE_SEP = "~"


@dataclass(frozen=True)
class BenchQuestion:
    """A benchmark question with its reference answer and the documents that support it.

    Attributes:
        id: Unique question id.
        type: Question type, e.g. ``info_not_found``.
        text: The question as asked.
        expected_doc_ids: Ids of the documents that answer it; empty for unanswerable and
            high-level questions.
        gold_answer: The reference answer.
        answer_facts: Checklist of facts an answer must satisfy; see :mod:`techrag.judge`.
    """

    id: str
    type: str
    text: str
    expected_doc_ids: tuple[str, ...]
    gold_answer: str
    answer_facts: tuple[str, ...]


def load_questions(bench_dir: Path = BENCH_DIR) -> list[BenchQuestion]:
    """Load the benchmark questions from the dataset under ``bench_dir``, in file order.

    Args:
        bench_dir: Root of the downloaded EnterpriseRAG-Bench dataset.

    Returns:
        The questions in file order.

    Raises:
        FileNotFoundError: If the questions file is missing.
    """
    path = bench_dir / "data" / "questions" / "test.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Questions not found: {path}")
    return [
        BenchQuestion(
            id=row["question_id"],
            type=row["question_type"],
            text=row["question"],
            expected_doc_ids=tuple(row["expected_doc_ids"]),
            gold_answer=row["gold_answer"],
            answer_facts=tuple(row["answer_facts"]),
        )
        for row in pq.read_table(path).to_pylist()
    ]


def write_questions(questions: Iterable[BenchQuestion], path: Path) -> None:
    """Write ``questions`` to ``path`` as JSON Lines, one object per question.

    Args:
        questions: Questions to write.
        path: File to create or overwrite. Its directory must exist.
    """
    lines = [
        json.dumps(
            {
                "question_id": q.id,
                "question_type": q.type,
                "question": q.text,
                "expected_doc_ids": list(q.expected_doc_ids),
                "gold_answer": q.gold_answer,
                "answer_facts": list(q.answer_facts),
            },
            ensure_ascii=False,
        )
        for q in questions
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_questions(path: Path) -> list[BenchQuestion]:
    """Read the questions :func:`write_questions` wrote to the JSON Lines file at ``path``.

    Args:
        path: JSON Lines file of question records.

    Returns:
        The questions in file order. Blank lines are skipped.

    Raises:
        FileNotFoundError: If ``path`` is not an existing file.
        ValueError: If a line is not a valid question record. The message names the line.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Questions not found: {path}")
    questions = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            questions.append(
                BenchQuestion(
                    id=record["question_id"],
                    type=record["question_type"],
                    text=record["question"],
                    expected_doc_ids=tuple(record["expected_doc_ids"]),
                    gold_answer=record["gold_answer"],
                    answer_facts=tuple(record["answer_facts"]),
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"{path}:{lineno}: invalid question record: {e!r}") from e
    return questions


def doc_id_of(source: Path | str) -> str:
    """Return the benchmark document id of a subset file path such as ``jira/dsid_ab.md``.

    Args:
        source: Path of a subset document, relative to the subset directory.

    Returns:
        The file name without its extension or any duplicate-disambiguating suffix.
    """
    return Path(source).stem.split(_DUPLICATE_SEP)[0]


def sample_questions(
    questions: Sequence[BenchQuestion],
    n: int | None = None,
    per_type: int | None = None,
    seed: int = 0,
) -> list[BenchQuestion]:
    """Return a seeded sample of ``questions`` in their original order.

    With ``per_type``, up to that many questions of every type, so that small types are not lost
    in a random sample. Otherwise ``n`` questions drawn uniformly. With neither, all of them.
    Each type is shuffled on its own, so growing ``per_type`` only adds questions: results
    saved for a smaller sample stay valid for a larger one.

    Args:
        questions: Questions to sample from.
        n: Number of questions to draw uniformly. Exclusive with ``per_type``.
        per_type: Maximum number of questions to draw from each type. Exclusive with ``n``.
        seed: Seed for the random generator, so the sample is reproducible.

    Returns:
        The sampled questions in their original order.

    Raises:
        ValueError: If ``n`` and ``per_type`` are both given, or either is not positive.
    """
    if n is not None and per_type is not None:
        raise ValueError("give n or per_type, not both")
    if (n is not None and n <= 0) or (per_type is not None and per_type <= 0):
        raise ValueError("sample sizes must be positive")
    if per_type is not None:
        chosen: set[str] = set()
        for qtype in sorted({q.type for q in questions}):
            group = [q for q in questions if q.type == qtype]
            random.Random(f"{seed}:{qtype}").shuffle(group)
            chosen.update(q.id for q in group[:per_type])
        return [q for q in questions if q.id in chosen]
    if n is not None:
        picked = {q.id for q in random.Random(seed).sample(list(questions), min(n, len(questions)))}
        return [q for q in questions if q.id in picked]
    return list(questions)


def render_document(title: str, content: str) -> str:
    """Return the Markdown written for a document: its title as an H1, then its content.

    Args:
        title: Document title.
        content: Document body.

    Returns:
        The Markdown text of the document.
    """
    return f"# {title}\n\n{content}\n"


def sample_distractors(candidates: list[int], count: int, seed: int) -> set[int]:
    """Return ``count`` of ``candidates`` chosen uniformly at random (all if fewer exist).

    Args:
        candidates: Row numbers to choose from.
        count: Number to choose.
        seed: Seed for the random generator.

    Returns:
        The chosen row numbers.
    """
    return set(random.Random(seed).sample(candidates, min(count, len(candidates))))


def _batches(documents_path: Path, columns: list[str]) -> Iterable[list[dict]]:
    """Yield the rows of ``columns`` in ``documents_path`` in batches of up to 2000."""
    # A single row group holds the whole corpus, so stream it in small batches.
    for batch in pq.ParquetFile(documents_path).iter_batches(batch_size=2000, columns=columns):
        yield batch.to_pylist()


def build_subset(
    bench_dir: Path = BENCH_DIR,
    out_dir: Path = SUBSET_DIR,
    distractors: int = 20_000,
    seed: int = 0,
) -> dict:
    """Write the gold documents plus ``distractors`` random others under ``out_dir``.

    Existing files are overwritten; files from an earlier, differently sized subset are not
    removed, so use a fresh ``out_dir`` when changing ``distractors`` or ``seed``.

    Args:
        bench_dir: Root of the downloaded EnterpriseRAG-Bench dataset.
        out_dir: Directory to write the documents, ``questions.jsonl`` and ``manifest.json`` to.
        distractors: Number of non-gold documents to include.
        seed: Seed for choosing the distractors.

    Returns:
        The manifest that is also written to ``out_dir``: the seed, document counts and the
        count per source type.

    Raises:
        FileNotFoundError: If the dataset is missing.
        ValueError: If ``distractors`` is negative or a gold document is not in the corpus.
    """
    if distractors < 0:
        raise ValueError("distractors must not be negative")
    documents_path = bench_dir / "data" / "documents" / "test.parquet"
    if not documents_path.is_file():
        raise FileNotFoundError(f"Documents not found: {documents_path}")

    questions = load_questions(bench_dir)
    gold = {i for q in questions for i in q.expected_doc_ids}
    # Pass 1 reads only ids, to pick the distractors without loading document text.
    ids = [row["doc_id"] for rows in _batches(documents_path, ["doc_id"]) for row in rows]
    missing = gold - set(ids)
    if missing:
        raise ValueError(f"{len(missing)} gold documents are not in the corpus")
    picked = sample_distractors([n for n, i in enumerate(ids) if i not in gold], distractors, seed)

    # Pass 2 writes the documents. Rows sharing an id (a handful in the corpus) are different
    # documents, so each gets its own file and counts as the same id when scoring.
    written: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    n_gold = row_number = 0
    columns = ["doc_id", "source_type", "title", "content"]
    for rows in _batches(documents_path, columns):
        for row in rows:
            is_gold = row["doc_id"] in gold
            if is_gold or row_number in picked:
                name = row["doc_id"]
                if written[name]:
                    name += f"{_DUPLICATE_SEP}{written[name]}"
                written[row["doc_id"]] += 1
                path = out_dir / row["source_type"] / f"{name}.md"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(render_document(row["title"], row["content"]), encoding="utf-8")
                by_source[row["source_type"]] += 1
                n_gold += is_gold
            row_number += 1

    manifest = {
        "seed": seed,
        "documents": sum(by_source.values()),
        "gold_documents": n_gold,
        "distractors": len(picked),
        "by_source_type": dict(sorted(by_source.items())),
    }
    write_questions(questions, out_dir / QUESTIONS)
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    """Entry point for the ``techrag-subset`` command."""
    parser = argparse.ArgumentParser(
        description="Write a dev subset of EnterpriseRAG-Bench: all gold docs plus distractors."
    )
    parser.add_argument("--bench-dir", type=Path, default=BENCH_DIR)
    parser.add_argument("--out-dir", type=Path, default=SUBSET_DIR)
    parser.add_argument("--distractors", type=int, default=20_000, help="non-gold documents")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        manifest = build_subset(args.bench_dir, args.out_dir, args.distractors, args.seed)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")
    print(
        f"Wrote {manifest['documents']} documents ({manifest['gold_documents']} gold, "
        f"{manifest['distractors']} distractors) to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
