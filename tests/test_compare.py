import math

import pytest

from techrag.benchmarks.bench import BenchQuestion, sample_questions
from techrag.eval.judge import Verdict
from techrag.reasoning.baseline import Record
from techrag.reasoning.compare import bootstrap_ci, cost, format_report, paired, summarize_by_type


def _rec(qid: str, met: bool | None, qtype: str = "basic", seconds: float = 1.0, steps: int = 0):
    verdict = Verdict((met,), "", "r", (False,)) if met is not None else None
    error = None if verdict else "x"
    return Record(qid, qtype, "a", (), None, seconds, verdict, error, steps)


def test_paired_keeps_questions_graded_in_both_in_first_order():
    a = {"q1": _rec("q1", True), "q2": _rec("q2", None), "q3": _rec("q3", True)}
    b = {
        "q3": _rec("q3", False),
        "q1": _rec("q1", True),
        "q2": _rec("q2", True),
        "q4": _rec("q4", True),
    }
    assert [x.question_id for x, _ in paired(a, b)] == ["q1", "q3"]


def test_summarize_by_type_reports_delta_as_b_minus_a():
    pairs = [
        (_rec("q1", False, "t1"), _rec("q1", True, "t1")),
        (_rec("q2", True, "t2"), _rec("q2", True, "t2")),
    ]
    s = summarize_by_type(pairs)
    assert s["t1"] == {"n": 1, "a": 0.0, "b": 1.0, "delta": 1.0}
    assert s["t2"]["delta"] == 0.0


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    diffs = [1.0, 0.0, 0.5, -0.5, 1.0, 0.0, 0.0, 0.5]
    lo, hi = bootstrap_ci(diffs)
    assert lo <= sum(diffs) / len(diffs) <= hi
    assert (lo, hi) == bootstrap_ci(diffs)
    assert bootstrap_ci([0.25] * 5) == (0.25, 0.25)
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_cost_and_report():
    pairs = [(_rec("q1", False, seconds=10), _rec("q1", True, seconds=20, steps=4))]
    assert cost([y for _, y in pairs])["steps"] == 4
    assert math.isnan(cost([])["seconds"])
    report = format_report(pairs, "base", "agent")
    assert "overall delta +1.000" in report and "base" in report and "agent" in report


def _q(qid: str, qtype: str) -> BenchQuestion:
    return BenchQuestion(qid, qtype, "?", (), "", ("f",))


def test_sample_questions_stratifies_and_keeps_file_order():
    questions = [_q(f"a{i}", "a") for i in range(10)] + [_q(f"b{i}", "b") for i in range(2)]
    sample = sample_questions(questions, per_type=3, seed=1)
    assert [q.type for q in sample].count("a") == 3 and [q.type for q in sample].count("b") == 2
    assert [q.id for q in sample] == [q.id for q in questions if q in sample]


def test_sample_questions_grows_by_adding_questions_and_is_seeded():
    questions = [_q(f"a{i}", "a") for i in range(20)] + [_q(f"b{i}", "b") for i in range(20)]
    small = {q.id for q in sample_questions(questions, per_type=3)}
    large = {q.id for q in sample_questions(questions, per_type=6)}
    assert small < large
    assert small == {q.id for q in sample_questions(questions, per_type=3)}
    assert small != {q.id for q in sample_questions(questions, per_type=3, seed=9)}


def test_sample_questions_plain_and_invalid_arguments():
    questions = [_q(f"q{i}", "a") for i in range(10)]
    assert len(sample_questions(questions, n=4)) == 4
    assert sample_questions(questions) == questions
    with pytest.raises(ValueError):
        sample_questions(questions, n=2, per_type=2)
    with pytest.raises(ValueError):
        sample_questions(questions, per_type=0)
