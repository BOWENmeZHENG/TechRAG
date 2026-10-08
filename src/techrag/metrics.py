"""Retrieval metrics computed from the rank at which each query's first relevant result appeared.

A rank is 1-based; ``None`` means no relevant result was retrieved.
"""

from collections.abc import Sequence


def _check_ranks(ranks: Sequence[int | None]) -> None:
    if not ranks:
        raise ValueError("ranks must not be empty")
    if any(r is not None and r < 1 for r in ranks):
        raise ValueError("ranks must be 1-based positive integers or None")


def hit_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Return the fraction of queries whose first relevant result is within the top ``k``.

    Raises:
        ValueError: If ``ranks`` is empty or holds a rank below 1, or ``k`` is not positive.
    """
    _check_ranks(ranks)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return sum(r is not None and r <= k for r in ranks) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    """Return the mean of ``1 / rank`` over all queries, counting a miss as 0.

    Raises:
        ValueError: If ``ranks`` is empty or holds a rank below 1.
    """
    _check_ranks(ranks)
    return sum(1 / r for r in ranks if r is not None) / len(ranks)
