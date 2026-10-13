import math

import pytest

from techrag.bm25 import BM25, tokenize


def test_tokenize_keeps_underscores_but_splits_dots():
    assert tokenize("requires_grad") == ["requires_grad"]
    assert tokenize("torch.nn.Module") == ["torch", "nn", "module"]


def test_scores_one_per_document_in_order():
    docs = ["cat sat", "dog ran", "cat dog"]
    scores = BM25(docs).scores("cat")
    assert len(scores) == 3
    assert scores[0] > 0
    assert scores[1] == 0
    assert scores[2] > 0


@pytest.mark.parametrize("query", ["zeta", ""])  # term absent from the corpus, empty query
def test_query_without_known_terms_scores_zero(query: str):
    assert BM25(["alpha beta", "gamma"]).scores(query) == [0.0, 0.0]


def test_empty_corpus():
    assert BM25([]).scores("anything") == []


def test_all_empty_documents_do_not_divide_by_zero():
    assert BM25(["", ""]).scores("anything") == [0.0, 0.0]


def test_scoring_is_case_insensitive():
    bm = BM25(["Gradient Descent", "other"])
    assert bm.scores("GRADIENT") == bm.scores("gradient")


def test_rare_term_outweighs_common_term():
    docs = ["the cat", "the dog", "the bird", "the tensor"]
    bm = BM25(docs)
    assert bm.scores("tensor")[3] > bm.scores("the")[3]


def test_term_in_every_document_still_non_negative():
    scores = BM25(["common a", "common b", "common c"]).scores("common")
    assert all(s > 0 for s in scores)


def test_more_occurrences_score_higher_but_saturate():
    # Equal lengths so only tf differs; "x" pads with a non-query filler token.
    docs = ["w x x x x x x x", "w w x x x x x x", "w w w x x x x x", "x x x x x x x x"]
    s = BM25(docs).scores("w")
    assert s[2] > s[1] > s[0] > 0
    assert s[1] - s[0] > s[2] - s[1] > 0  # per-occurrence gain shrinks (the last jump is 5 more)


def test_shorter_document_scores_higher_for_same_tf():
    docs = ["gradient", "gradient " + "filler " * 20]
    s = BM25(docs).scores("gradient")
    assert s[0] > s[1]


def test_b_zero_ignores_length():
    docs = ["gradient", "gradient " + "filler " * 20]
    s = BM25(docs, b=0).scores("gradient")
    assert s[0] == pytest.approx(s[1])


def test_duplicate_query_terms_count_once():
    bm = BM25(["gradient descent", "other"])
    assert bm.scores("gradient gradient gradient") == bm.scores("gradient")


def test_multiple_query_terms_sum():
    bm = BM25(["alpha beta", "alpha gamma", "delta"])
    both = bm.scores("alpha beta")[0]
    assert both == pytest.approx(bm.scores("alpha")[0] + bm.scores("beta")[0])


def test_matches_hand_computed_value():
    # Single doc, single term: N=1, df=1, tf=1, length=avg => norm=1.
    k1 = 1.5
    idf = math.log(1 + (1 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 1 * (k1 + 1) / (1 + k1)
    assert BM25(["hello"], k1=k1).scores("hello") == [pytest.approx(expected)]
