"""Compare two systems' logged results (baseline vs agent) on the questions both graded.

Scores are paired: a question counts only if both systems have a verdict for it, and the
difference is taken question by question. The confidence interval on the overall difference is
a seeded bootstrap over those questions. With a few questions per type, per-type differences are
mostly noise; the overall interval says whether the gap is bigger than that.
"""

import argparse
import random
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from techrag.baseline import Record, load_records

Pair = tuple[Record, Record]


def paired(a: dict[str, Record], b: dict[str, Record]) -> list[Pair]:
    """Return the records of the questions graded in both ``a`` and ``b``, in ``a``'s order."""
    return [(a[q], b[q]) for q in a if q in b and a[q].verdict and b[q].verdict]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def bootstrap_ci(
    diffs: Sequence[float], resamples: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Return the bootstrap confidence interval of the mean of ``diffs``.

    Raises:
        ValueError: If ``diffs`` is empty.
    """
    if not diffs:
        raise ValueError("diffs must not be empty")
    rng = random.Random(seed)
    means = sorted(_mean([rng.choice(diffs) for _ in diffs]) for _ in range(resamples))
    tail = (1 - level) / 2
    return means[int(tail * resamples)], means[min(resamples - 1, int((1 - tail) * resamples))]


def summarize_pairs(pairs: Sequence[Pair]) -> dict[str, float]:
    """Return the question count, mean score of each system and mean difference (b minus a)."""
    a = [x.score for x, _ in pairs if x.score is not None]
    b = [y.score for _, y in pairs if y.score is not None]
    return {"n": len(pairs), "a": _mean(a), "b": _mean(b), "delta": _mean(b) - _mean(a)}


def summarize_by_type(pairs: Sequence[Pair]) -> dict[str, dict[str, float]]:
    """Return :func:`summarize_pairs` for each question type, by name."""
    groups: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs:
        groups[pair[0].question_type].append(pair)
    return {t: summarize_pairs(groups[t]) for t in sorted(groups)}


def cost(records: Sequence[Record]) -> dict[str, float]:
    """Return the mean seconds, model turns and tokens per question of ``records``."""
    return {
        "seconds": _mean([r.answer_seconds for r in records]),
        "steps": _mean([r.steps for r in records]),
        "tokens": _mean([r.tokens for r in records]),
    }


def format_report(pairs: Sequence[Pair], name_a: str, name_b: str) -> str:
    """Render the per-type and overall comparison, then the cost of each system, as text."""
    lines = [f"{'':<26}{'n':>5}{name_a:>11}{name_b:>11}{'delta':>9}"]
    overall = summarize_pairs(pairs)
    for name, s in [*summarize_by_type(pairs).items(), ("ALL", overall)]:
        lines.append(f"{name:<26}{s['n']:>5}{s['a']:>11.3f}{s['b']:>11.3f}{s['delta']:>+9.3f}")
    lo, hi = bootstrap_ci([y.score - x.score for x, y in pairs])  # type: ignore[operator]
    lines.append(f"\noverall delta {overall['delta']:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")
    lines.append(f"\n{'cost per question':<26}{'seconds':>9}{'steps':>8}{'tokens':>9}")
    for name, records in ((name_a, [x for x, _ in pairs]), (name_b, [y for _, y in pairs])):
        c = cost(records)
        lines.append(f"{name:<26}{c['seconds']:>9.1f}{c['steps']:>8.1f}{c['tokens']:>9.0f}")
    return "\n".join(lines)


def main() -> None:
    """Entry point for the ``techrag-compare`` command."""
    parser = argparse.ArgumentParser(description="Compare two systems' logged results.")
    parser.add_argument("a", type=Path, help="results of the first system (e.g. the baseline)")
    parser.add_argument("b", type=Path, help="results of the second system (e.g. the agent)")
    parser.add_argument("--names", nargs=2, default=("baseline", "agent"), metavar=("A", "B"))
    args = parser.parse_args()

    try:
        a, b = load_records(args.a), load_records(args.b)
    except ValueError as e:
        sys.exit(f"error: {e}")
    pairs = paired(a, b)
    if not pairs:
        sys.exit("error: no question is graded in both files")
    print(format_report(pairs, *args.names))
    print(f"\n{len(pairs)} questions graded by both ({len(a)} in {args.a}, {len(b)} in {args.b})")


if __name__ == "__main__":
    main()
