"""Single-shot RAG baseline on the EnterpriseRAG-Bench questions of a subset.

For each question the baseline retrieves the top ``k`` chunks once, has an Ollama model answer
from them, and grades the answer with :mod:`techrag.eval.judge`. It is the number the agentic search
loop has to beat, and the per-type breakdown shows where single-shot retrieval falls short.

Results are logged and resumed as described in :mod:`techrag.reasoning.harness`.
"""

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import get_args

from techrag.benchmarks.bench import BenchQuestion, doc_id_of, read_questions, sample_questions
from techrag.common.config import (
    EMBEDDING_MODEL,
    INDEX_DIR,
    JUDGE_MODEL,
    NUM_CTX,
    OLLAMA_URL,
    SUBSET_DIR,
    AgentConfig,
    EvaluationConfig,
    LLMConfig,
    RetrievalConfig,
    RunConfig,
)
from techrag.common.log import setup_logging
from techrag.common.ollama import ollama_chat
from techrag.core.generate import answer_from
from techrag.core.retrieve import Mode, Result, Retriever
from techrag.eval.judge import Chat, ollama_judge
from techrag.reasoning.harness import (
    Record,
    data_config,
    gold_found,
    grade,
    print_report,
    run_each,
    run_logged,
    run_metadata,
    save_manifest,
)

ANSWER_MODEL = "qwen3.5:9b"
RESULTS_PATH = Path("eval/baseline.jsonl")


def run_question(
    question: BenchQuestion,
    retrieve: Callable[[str, int], Sequence[Result]],
    answer_chat: Callable[[Sequence[dict[str, str]]], str],
    judge_chat: Chat,
    k: int,
) -> Record:
    """Answer ``question`` from the top ``k`` chunks and grade the answer.

    An answer request that times out becomes an ``error`` on the record (see :func:`grade`).

    Args:
        question: The question to answer.
        retrieve: Function taking a query and a chunk count and returning chunks, best first.
        answer_chat: Function that sends the answer model its messages and returns its reply.
        judge_chat: Function that sends the judge its messages and schema and returns its reply.
        k: Number of chunks to give the answer model.

    Returns:
        The graded record of the question.

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

    Args:
        questions: Questions to run, in order.
        retrieve: Function taking a query and a chunk count and returning chunks, best first.
        answer_chat: Function that sends the answer model its messages and returns its reply.
        judge_chat: Function that sends the judge its messages and schema and returns its reply.
        k: Number of chunks to give the answer model.
        done: Earlier records by question id.
        on_record: Called with each new record as soon as it is produced.

    Returns:
        The records produced by this call, in question order.
    """
    return run_each(
        questions, lambda q: run_question(q, retrieve, answer_chat, judge_chat, k), done, on_record
    )


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
    parser.add_argument("--per-type", type=int, help="run on this many questions of every type")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH, help="JSON Lines results")
    parser.add_argument("--run-name", help="name for the run manifest (default: results file name)")
    parser.add_argument("--fresh", action="store_true", help="discard earlier results first")
    args = parser.parse_args()
    setup_logging()

    def answer_chat(messages: Sequence[dict[str, str]]) -> str:
        # The same context window as the judge, so Ollama keeps one model loaded for both.
        return ollama_chat(
            messages, args.llm, args.ollama_url, think=args.think, temperature=0.0, num_ctx=NUM_CTX
        )

    try:
        config = RunConfig(
            metadata=run_metadata(args.run_name or args.output.stem),
            llm=LLMConfig(model_name=args.llm, num_ctx=NUM_CTX),
            retrieval=RetrievalConfig(
                mode=args.mode, k=args.k, embedding_model=args.embedding_model
            ),
            agent=AgentConfig(max_steps=1, think=args.think),  # one model turn, no search loop
            evaluation=EvaluationConfig(judge_model=args.judge_model),
            data=data_config(args.subset),
        )
        save_manifest(config, args.output)
        questions = read_questions(args.subset / "questions.jsonl")
        questions = sample_questions(questions, args.n, args.per_type, args.seed)
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
