import pytest

from techrag.metrics import hit_at_k, mean_reciprocal_rank

RANKS = [1, 2, None, 4]


@pytest.mark.parametrize(("k", "expected"), [(1, 0.25), (2, 0.5), (3, 0.5), (4, 0.75), (10, 0.75)])
def test_hit_at_k_counts_ranks_within_k(k: int, expected: float) -> None:
    assert hit_at_k(RANKS, k) == pytest.approx(expected)


def test_mean_reciprocal_rank_counts_a_miss_as_zero() -> None:
    assert mean_reciprocal_rank(RANKS) == pytest.approx((1 + 1 / 2 + 0 + 1 / 4) / 4)


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_raises(k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        hit_at_k(RANKS, k)


@pytest.mark.parametrize("metric", [lambda r: hit_at_k(r, 1), mean_reciprocal_rank])
def test_empty_ranks_raise(metric) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        metric([])


@pytest.mark.parametrize("metric", [lambda r: hit_at_k(r, 1), mean_reciprocal_rank])
def test_rank_below_one_raises(metric) -> None:
    with pytest.raises(ValueError, match="1-based"):
        metric([1, 0])
