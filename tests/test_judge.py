import json
from collections.abc import Sequence

import pytest

from techrag.bench import BenchQuestion
from techrag.judge import (
    ABSTAIN,
    Verdict,
    build_messages,
    judge,
    make_probes,
    parse_verdict,
    selftest,
    summarize_probes,
)


def _question(qid: str, facts: tuple[str, ...], qtype: str = "basic") -> BenchQuestion:
    return BenchQuestion(qid, qtype, f"question {qid}?", ("d",), f"gold {qid}", facts)


def _reply(met: Sequence[bool], contradiction: str = "") -> str:
    return json.dumps(
        {
            "facts": [{"evidence": "e", "met": m} for m in met],
            "contradiction": contradiction,
            "reason": "because",
        }
    )


def test_verdict_scores():
    assert Verdict((True, False, True, True), "", "").completeness == 0.75
    assert Verdict((True, True), "says 20%", "").score == 0.0
    assert Verdict((True, False), "  ", "").score == 0.5  # a blank contradiction is none
    assert Verdict((True, False), "", "").score == 0.5


def test_parse_verdict():
    verdict = parse_verdict(_reply([True, False], contradiction="says 20%"), 2)
    assert verdict == Verdict((True, False), "says 20%", "because")
    assert verdict.incorrect


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"facts": []}',
        _reply([True]),  # wrong number of facts
        '{"facts": [{"met": "yes"}], "contradiction": "", "reason": ""}',
        '{"facts": [{"met": true}], "contradiction": false, "reason": ""}',
        '{"facts": [{"met": true}], "incorrect": false, "reason": ""}',  # the old schema
    ],
)
def test_parse_verdict_rejects_bad_replies(raw: str):
    with pytest.raises(ValueError):
        parse_verdict(raw, 2)


def test_build_messages_includes_question_reference_facts_and_answer():
    q = _question("q1", ("fact one", "fact two"))
    system, user = build_messages(q, "my answer")
    assert system["role"] == "system" and user["role"] == "user"
    for text in (
        "question q1?",
        "gold q1",
        "1. fact one",
        "2. fact two",
        "my answer",
        "2 required",
    ):
        assert text in user["content"]


def test_judge_grades_with_chat_reply():
    q = _question("q1", ("a", "b"))
    assert judge(q, "ans", lambda messages: _reply([True, False])).completeness == 0.5


def test_judge_skips_the_model_for_a_blank_answer():
    def chat(messages):
        raise AssertionError("must not be called")

    verdict = judge(_question("q1", ("a", "b")), "  \n", chat)
    assert verdict.facts_met == (False, False) and verdict.score == 0.0


def test_judge_retries_unusable_replies_then_raises():
    q = _question("q1", ("a",))
    replies = iter(["garbage", _reply([True])])
    assert judge(q, "ans", lambda m: next(replies)).facts_met == (True,)

    calls = []

    def always_bad(messages):
        calls.append(1)
        return "garbage"

    with pytest.raises(ValueError):
        judge(q, "ans", always_bad, retries=2)
    assert len(calls) == 3


def test_judge_rejects_question_without_facts():
    with pytest.raises(ValueError, match="no answer facts"):
        judge(_question("q1", ()), "ans", lambda m: _reply([]))


def test_make_probes_kinds_and_expected_scores():
    qa = _question("a", ("a1", "a2", "a3", "a4"))
    qb = _question("b", ("b1",))
    qn = _question("n", ("must say not found",), "info_not_found")
    probes = {
        (q.id, kind): (answer, expected) for q, kind, answer, expected in make_probes([qa, qb, qn])
    }

    assert probes[("a", "reference")] == ("gold a", 1.0)
    assert probes[("a", "abstain")] == (ABSTAIN, 0.0)
    assert probes[("a", "partial")] == ("a1 a2", 0.5)
    assert probes[("a", "mismatch")] == ("gold b", 0.0)
    assert probes[("b", "mismatch")] == ("gold a", 0.0)
    assert ("b", "partial") not in probes  # a single fact cannot be halved
    # the reference answer of an unanswerable question is a rubric, so it is not probed
    assert ("n", "reference") not in probes
    assert probes[("n", "abstain")] == (ABSTAIN, 1.0)
    assert probes[("n", "mismatch")][1] == 0.0


def test_make_probes_expect_prohibitions_to_be_met_by_any_answer():
    q = _question("p", ("s1", "s2", "The answer must not say X.", "The answer must not say Y."))
    probes = {kind: (answer, expected) for _, kind, answer, expected in make_probes([q, q])[:3]}
    assert probes["abstain"][1] == 0.5  # the two prohibitions
    assert probes["partial"] == ("s1", 0.75)  # one of two statements, plus the prohibitions


def test_selftest_and_summary_with_a_perfect_judge():
    questions = [_question(f"q{i}", ("x", "y")) for i in range(4)]

    def perfect(messages):
        user = messages[1]["content"]
        answer = user.split("Candidate answer:\n")[1].split("\n\nFor each")[0]
        if answer == "x":  # the "partial" probe: only the first fact
            return _reply([True, False])
        if answer.startswith("gold q") and f"Reference answer: {answer}" in user:
            return _reply([True, True])
        return _reply([False, False])

    probes = selftest(questions, perfect, n=4, seed=1)
    summary = summarize_probes(probes)
    assert set(summary) == {"abstain", "mismatch", "partial", "reference"}
    assert all(s["mean_abs_error"] == 0 and s["within_0.25"] == 1 for s in summary.values())
    assert summary["reference"]["n"] == 4
