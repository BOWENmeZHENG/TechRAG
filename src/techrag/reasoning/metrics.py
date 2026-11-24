"""What to log to MLflow from the records of a baseline or agent run.

Turns :class:`~techrag.reasoning.harness.Record` s into metrics and an agent trace, and reads a
run's results file back to log them. The run itself is in :mod:`techrag.common.tracking`.
"""

import logging
import math
import re
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

from techrag.benchmarks.bench import BenchQuestion
from techrag.common.tracking import Tracker
from techrag.reasoning.harness import Record, load_records, summarize, summarize_by

logger = logging.getLogger(__name__)


def _slug(text: str) -> str:
    """Return ``text`` as a lower-case name of letters, digits and underscores."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def flatten_metrics(records: Sequence[Record]) -> dict[str, float]:
    """Return the scores of ``records`` as MLflow metrics.

    * ``score``, ``correctness``, ``completeness`` (means over graded questions), ``n_questions``
      and ``errors``;
    * the same three means and the count for each question type and retrieval group, as
      ``by_type/<type>/score`` and ``by_retrieval/<group>/score``;
    * mean ``answer_seconds`` and ``gold_found``;
    * for agent runs, mean ``steps`` and ``tokens``, and ``stop/<reason>``, the number of
      questions that ended for each reason.

    A mean over nothing is NaN and is left out.

    Args:
        records: The run's records, as returned by ``run_logged``.

    Returns:
        A map from metric name to value.
    """
    metrics: dict[str, float] = {}

    def add(prefix: str, summary: dict) -> None:
        for key in ("score", "correctness", "completeness"):
            if not math.isnan(summary[key]):
                metrics[f"{prefix}{key}"] = summary[key]

    overall = summarize(records)
    add("", overall)
    metrics["n_questions"] = overall["n"]
    metrics["errors"] = overall["errors"]
    for name, summary in summarize_by(records, lambda r: r.question_type).items():
        add(f"by_type/{_slug(name)}/", summary)
        metrics[f"by_type/{_slug(name)}/n"] = summary["n"]
    for name, summary in summarize_by(records, lambda r: r.retrieval_group).items():
        add(f"by_retrieval/{_slug(name)}/", summary)
        metrics[f"by_retrieval/{_slug(name)}/n"] = summary["n"]

    if records:
        metrics["answer_seconds"] = fmean(r.answer_seconds for r in records)
    found = [r.gold_found for r in records if r.gold_found is not None]
    if found:
        metrics["gold_found"] = fmean(found)
    agent = [r for r in records if r.steps]
    if agent:
        metrics["steps"] = fmean(r.steps for r in agent)
        metrics["tokens"] = fmean(r.tokens for r in agent)
        for r in agent:
            key = f"stop/{_slug(r.stop) or 'unknown'}"
            metrics[key] = metrics.get(key, 0) + 1
    return metrics


def agent_trace(records: Sequence[Record]) -> dict[str, dict]:
    """Return what the agent did on each question, by question id.

    Each entry has the question's type, score, ``steps``, ``tokens``, ``stop`` reason and
    ``error``, and ``calls``: one item per tool call with its step, the model's ``thought``
    alongside it, the tool, its arguments, the documents it returned and the reply's size.

    Args:
        records: The records of an agent run.

    Returns:
        A map from question id to its trace, in record order.
    """
    return {
        r.question_id: {
            "question_type": r.question_type,
            "score": r.score,
            "steps": r.steps,
            "tokens": r.tokens,
            "stop": r.stop,
            "error": r.error,
            "calls": list(r.trace),
        }
        for r in records
    }


def log_results(
    tracker: Tracker, questions: Sequence[BenchQuestion], output: Path, agent: bool = False
) -> None:
    """Log the outcome of a run from its results file: metrics, the file, and the agent's trace.

    It reads ``output`` instead of taking records, so it also records what an interrupted or
    failed run got through. Only the records of ``questions`` count, not earlier runs' leftovers.
    Nothing is logged if tracking is off.

    Args:
        tracker: The run's tracker.
        questions: The questions the run was asked to answer.
        output: The JSON Lines results file.
        agent: Whether this is an agent run, which also logs ``agent_trace.json``.
    """
    if not tracker.enabled:
        return
    try:
        latest = load_records(output)
    except (OSError, ValueError) as e:
        logger.warning("mlflow: could not read %s: %s", output, e)
        return
    records = [latest[q.id] for q in questions if q.id in latest]
    tracker.log_metrics(flatten_metrics(records))
    tracker.log_artifact(output)
    if agent:
        tracker.log_dict(agent_trace(records), "agent_trace.json")
