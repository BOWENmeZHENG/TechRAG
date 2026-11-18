"""Single-shot RAG baseline on the EnterpriseRAG-Bench questions of a subset.

For each question the baseline retrieves the top ``k`` chunks once, has an Ollama model answer
from them, and grades the answer with :mod:`techrag.judge`. It is the number the agentic search
loop has to beat, and the per-type breakdown shows where single-shot retrieval falls short.

Every graded question is appended to a JSON Lines file as soon as it finishes, so an interrupted
run (a 500-question run takes hours on a laptop) resumes where it stopped. Questions that failed
are retried on the next run; the latest record of a question wins.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from techrag.bench import SUBSET_DIR, BenchQuestion, doc_id_of, read_questions
from techrag.generate import OLLAMA_URL, answer_from, ollama_chat
from techrag.ingest import EMBEDDING_MODEL
from techrag.judge import JUDGE_MODEL, NUM_CTX, Chat, Verdict, judge, ollama_judge
from techrag.recall import INDEX_DIR, gold_doc_ranks
from techrag.retrieve import Mode, Result, Retriever

ANSWER_MODEL = "qwen3.5:9b"
RESULTS_PATH = Path("eval/baseline.jsonl")

# Groups of questions by how much of their evidence retrieval found, for the summary.
RETRIEVAL_GROUPS = ("all gold found", "some gold missing", "no gold documents")


@dataclass(frozen=True)
class Record:
    """A system's outcome on one question.

    ``sources`` are the document ids the system saw, in the order it saw them. ``gold_found`` is
    the share of the question's gold documents among them, or ``None`` for a question that has
    none. A question that could not be answered or graded has an ``error`` and no ``verdict``.

    The last four fields describe an agent run: how many model turns it took, the tokens it
    processed (prompt and reply, summed over turns), why it stopped, and what it did, one entry
    per tool call. The single-shot baseline leaves them empty.
    """

    question_id: str
    question_type: str
    answer: str
    sources: tuple[str, ...]
    gold_found: float | None
    answer_seconds: float
    verdict: Verdict | None = None
    error: str | None = None
    steps: int = 0
    tokens: int = 0
    stop: str = ""
    trace: tuple[dict, ...] = ()

    @property
    def score(self) -> float | None:
        """Return the judge's score, or ``None`` if the question has no verdict."""
        return self.verdict.score if self.verdict else None

    @property
    def retrieval_group(self) -> str:
        """Return which of :data:`RETRIEVAL_GROUPS` this question falls in."""
        if self.gold_found is None:
            return RETRIEVAL_GROUPS[2]
        return RETRIEVAL_GROUPS[0] if self.gold_found >= 1.0 else RETRIEVAL_GROUPS[1]

    def to_json(self) -> str:
        """Return the record as one line of JSON."""
        v = self.verdict
        verdict = (
            {
                "facts_met": list(v.facts_met),
                "prohibitions": list(v.prohibitions),
                "contradiction": v.contradiction,
                "reason": v.reason,
                "correctness": v.correctness,
                "completeness": v.completeness,
                "score": v.score,
            }
            if v
            else None
        )
        return json.dumps(
            {
                "question_id": self.question_id,
                "question_type": self.question_type,
                "answer": self.answer,
                "sources": list(self.sources),
                "gold_found": self.gold_found,
                "answer_seconds": round(self.answer_seconds, 2),
                "verdict": verdict,
                "error": self.error,
                "steps": self.steps,
                "tokens": self.tokens,
                "stop": self.stop,
                "trace": list(self.trace),
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, line: str) -> "Record":
        """Parse a line written by :meth:`to_json`.

        Raises:
            ValueError: If ``line`` is not such a record.
        """
        try:
            d = json.loads(line)
            v = d["verdict"]
            verdict = (
                Verdict(
                    tuple(v["facts_met"]),
                    v["contradiction"],
                    v["reason"],
                    tuple(v["prohibitions"]),
                )
                if v
                else None
            )
            return cls(
                d["question_id"],
                d["question_type"],
                d["answer"],
                tuple(d["sources"]),
                d["gold_found"],
                d["answer_seconds"],
                verdict,
                d["error"],
                d.get("steps", 0),
                d.get("tokens", 0),
                d.get("stop", ""),
                tuple(d.get("trace", ())),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"invalid baseline record: {e!r}: {line[:200]!r}") from e


def load_records(path: Path) -> dict[str, Record]:
    """Return the records in ``path`` by question id, the latest of each; empty if no file.

    Raises:
        ValueError: If a line is not a valid record. The message names the line.
    """
    records: dict[str, Record] = {}
    if not path.is_file():
        return records
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = Record.from_json(line)
        except ValueError as e:
            raise ValueError(f"{path}:{lineno}: {e}") from e
        records[record.question_id] = record
    return records


def gold_found(question: BenchQuestion, results: Sequence[Result]) -> float | None:
    """Return the share of ``question``'s gold documents that ``results`` come from."""
    if not question.expected_doc_ids:
        return None
    ranks = gold_doc_ranks(question.expected_doc_ids, results)
    return sum(r is not None for r in ranks.values()) / len(ranks)


def grade(
    question: BenchQuestion,
    base: dict,
    text: str,
    seconds: float,
    judge_chat: Chat,
    **extra,
) -> Record:
    """Grade the answer ``text`` to ``question`` and return it as a :class:`Record`.

    ``base`` holds the record's ``question_id``, ``question_type``, ``sources`` and
    ``gold_found``; ``extra`` any of its other fields. A judge that never gives a usable verdict,
    or a request that times out, becomes an ``error`` on the record so that one bad question does
    not end a long run.

    Raises:
        OSError: If the judge's server cannot be reached.
    """
    try:
        verdict = judge(question, text, judge_chat)
    except (TimeoutError, ValueError) as e:
        return Record(**base, answer=text, answer_seconds=seconds, error=f"judge: {e}", **extra)
    return Record(**base, answer=text, answer_seconds=seconds, verdict=verdict, **extra)


def run_question(
    question: BenchQuestion,
    retrieve: Callable[[str, int], Sequence[Result]],
    answer_chat: Callable[[Sequence[dict[str, str]]], str],
    judge_chat: Chat,
    k: int,
) -> Record:
    """Answer ``question`` from the top ``k`` chunks and grade the answer.

    An answer request that times out becomes an ``error`` on the record (see :func:`grade`).

    Raises:
        OSError: If a model server cannot be reached.
    """
    results = retrieve(question.text, k)
    base = {
        "question_id": question.id,
        "question_type": question.type,
        "sources": tuple(doc_id_of(r.source) for r in results),
        "gold_found": gold_found(question, results),
    }
    start = time.monotonic()
    try:
        text = answer_from(question.text, results, answer_chat).text
    except (TimeoutError, ValueError) as e:
        return Record(
            **base, answer="", answer_seconds=time.monotonic() - start, error=f"answer: {e}"
        )
    return grade(question, base, text, time.monotonic() - start, judge_chat)


def run_each(
    questions: Sequence[BenchQuestion],
    one: Callable[[BenchQuestion], Record],
    done: dict[str, Record],
    on_record: Callable[[Record], None] = lambda r: None,
) -> list[Record]:
    """Call ``one`` on the questions not already graded in ``done``; return the new records.

    ``on_record`` is called with each record as it finishes, e.g. to save it. A question in
    ``done`` is redone if it had an error.

    Raises:
        OSError: If ``one`` does; records already passed to ``on_record`` are kept.
    """
    new = []
    for q in questions:
        if q.id in done and done[q.id].error is None:
            continue
        record = one(q)
        on_record(record)
        new.append(record)
    return new


def run(
    questions: Sequence[BenchQuestion],
    retrieve: Callable[[str, int], Sequence[Result]],
    answer_chat: Callable[[Sequence[dict[str, str]]], str],
    judge_chat: Chat,
    k: int,
    done: dict[str, Record],
    on_record: Callable[[Record], None] = lambda r: None,
) -> list[Record]:
    """Run :func:`run_question` on the questions not already graded in ``done``.

    See :func:`run_each`.
    """
    return run_each(
        questions, lambda q: run_question(q, retrieve, answer_chat, judge_chat, k), done, on_record
    )


def run_logged(
    questions: Sequence[BenchQuestion],
    one: Callable[[BenchQuestion], Record],
    output: Path,
    fresh: bool = False,
) -> list[Record]:
    """Run ``one`` on the questions still to do, appending each record to ``output`` as it ends.

    Progress goes to stderr. Returns the latest record of every question that has one, in the
    order of ``questions``. ``fresh`` discards earlier results first.

    Raises:
        OSError: If ``one`` does, or ``output`` cannot be written.
        ValueError: If ``output`` holds an invalid record.
    """
    if fresh:
        output.unlink(missing_ok=True)
    done = load_records(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    todo = sum(q.id not in done or done[q.id].error is not None for q in questions)
    print(f"{len(questions)} questions, {todo} to run", file=sys.stderr)

    with output.open("a", encoding="utf-8") as f:

        def save(record: Record) -> None:
            f.write(record.to_json() + "\n")
            f.flush()
            outcome = record.error or f"score {record.score:.2f}"
            detail = f" ({record.steps} steps, {record.stop})" if record.steps else ""
            print(
                f"  {record.question_id} {record.question_type}: {outcome}{detail}", file=sys.stderr
            )

        run_each(questions, one, done, save)
    latest = load_records(output)
    return [latest[q.id] for q in questions if q.id in latest]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarize(records: Sequence[Record]) -> dict:
    """Return the question count, error count and mean scores of ``records``.

    Means cover the graded records only; ``errors`` counts those left out. They are NaN when
    nothing was graded.
    """
    graded = [r.verdict for r in records if r.verdict]
    return {
        "n": len(records),
        "errors": len(records) - len(graded),
        "score": _mean([v.score for v in graded]),
        "correctness": _mean([v.correctness for v in graded]),
        "completeness": _mean([v.completeness for v in graded]),
    }


def summarize_by(records: Sequence[Record], key: Callable[[Record], str]) -> dict[str, dict]:
    """Return :func:`summarize` for each group of ``records`` sharing a ``key``, by name."""
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    return {name: summarize(groups[name]) for name in sorted(groups)}


def format_table(overall: dict, by_group: dict[str, dict]) -> str:
    """Render the per-group and overall summaries as a text table."""
    lines = [f"{'':<26}{'n':>5}{'err':>5}{'score':>8}{'correct':>9}{'complete':>10}"]
    for name, s in [*by_group.items(), ("ALL", overall)]:
        lines.append(
            f"{name:<26}{s['n']:>5}{s['errors']:>5}{s['score']:>8.3f}"
            f"{s['correctness']:>9.3f}{s['completeness']:>10.3f}"
        )
    return "\n".join(lines)


def print_report(records: Sequence[Record]) -> None:
    """Print the scores of ``records`` by question type, then by how much gold retrieval found."""
    overall = summarize(records)
    print(format_table(overall, summarize_by(records, lambda r: r.question_type)))
    print()
    print(format_table(overall, summarize_by(records, lambda r: r.retrieval_group)))


def main() -> None:
    """Entry point for the ``techrag-baseline`` command."""
    parser = argparse.ArgumentParser(description="Run the single-shot RAG baseline.")
    parser.add_argument("--subset", type=Path, default=SUBSET_DIR, help="subset with questions")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL, help="must match ingest")
    parser.add_argument("--mode", choices=get_args(Mode), default="hybrid", help="ranking method")
    parser.add_argument("-k", type=int, default=10, help="chunks to give the model")
    parser.add_argument("--llm", default=ANSWER_MODEL, help="Ollama model to answer with")
    parser.add_argument("--think", action="store_true", help="let the answer model reason first")
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help="Ollama model to grade with")
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("-n", type=int, help="run on a seeded random sample of this many questions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH, help="JSON Lines results")
    parser.add_argument("--fresh", action="store_true", help="discard earlier results first")
    args = parser.parse_args()

    def answer_chat(messages: Sequence[dict[str, str]]) -> str:
        # The same context window as the judge, so Ollama keeps one model loaded for both.
        return ollama_chat(
            messages, args.llm, args.ollama_url, think=args.think, temperature=0.0, num_ctx=NUM_CTX
        )

    try:
        questions = read_questions(args.subset / "questions.jsonl")
        if args.n is not None:
            questions = random.Random(args.seed).sample(questions, min(args.n, len(questions)))
        retriever = Retriever(args.index_dir, args.embedding_model)
        retriever.warm_up()
        judge_chat = ollama_judge(args.judge_model, args.ollama_url)
        records = run_logged(
            questions,
            lambda q: run_question(
                q,
                lambda text, k: retriever.search(text, k, args.mode),
                answer_chat,
                judge_chat,
                args.k,
            ),
            args.output,
            args.fresh,
        )
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")
    print_report(records)


if __name__ == "__main__":
    main()
