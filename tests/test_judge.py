import json
from collections.abc import Sequence

import pytest

from techrag.benchmarks.bench import BenchQuestion
from techrag.eval.judge import (
    ABSTAIN,
    Verdict,
    build_messages,
    is_prohibition,
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


def test_prohibitions_count_toward_correctness_not_completeness():
    # facts: two statements, then two prohibitions; the answer states one and breaks one ban
    verdict = Verdict((True, False, True, False), "", "", (False, False, True, True))
    assert verdict.completeness == 0.5
    assert verdict.violated and verdict.incorrect
    assert verdict.score == 0.0

    kept = Verdict((True, False, True, True), "", "", (False, False, True, True))
    assert kept.completeness == 0.5 and not kept.incorrect and kept.score == 0.5

    # an answer that says nothing keeps every ban but states nothing, so it scores zero
    silent = Verdict((False, False, True, True), "", "", (False, False, True, True))
    assert silent.score == 0.0


def test_is_prohibition():
    assert is_prohibition("The answer must not claim the target is 20%.")
    assert not is_prohibition("The answer must state that the target is 30%.")
    assert not is_prohibition("The target is 30%.")


def test_parse_verdict():
    verdict = parse_verdict(_reply([True, False], contradiction="says 20%"), (False, False))
    assert verdict == Verdict((True, False), "says 20%", "because", (False, False))
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
        parse_verdict(raw, (False, False))


def test_build_messages_includes_question_reference_facts_and_answer():
    q = _question("q1", ("fact one", "The answer must not say X."))
    system, user = build_messages(q, "my answer")
    assert system["role"] == "system" and user["role"] == "user"
    for text in (
        "question q1?",
        "gold q1",
        "1. [STATE] fact one",
        "2. [FORBIDDEN] The answer must not say X.",
        "my answer",
        "2 required",
    ):
        assert text in user["content"]


def test_judge_grades_with_chat_reply():
    q = _question("q1", ("a", "b"))
    seen = []

    def chat(messages, schema):
        seen.append(schema)
        return _reply([True, False])

    assert judge(q, "ans", chat).completeness == 0.5
    # the schema pins the number of fact entries, so the model cannot skip or add one
    assert seen[0]["properties"]["facts"]["minItems"] == 2
    assert seen[0]["properties"]["facts"]["maxItems"] == 2


def test_judge_skips_the_model_for_a_blank_answer():
    def chat(messages, schema):
        raise AssertionError("must not be called")

    verdict = judge(_question("q1", ("a", "b")), "  \n", chat)
    assert verdict.facts_met == (False, False) and verdict.score == 0.0


def test_judge_retries_unusable_replies_then_raises():
    q = _question("q1", ("a",))
    replies = iter(["garbage", _reply([True])])
    assert judge(q, "ans", lambda m, s: next(replies)).facts_met == (True,)

    calls = []

    def always_bad(messages, schema):
        calls.append(1)
        return "garbage"

    with pytest.raises(ValueError):
        judge(q, "ans", always_bad, retries=2)
    assert len(calls) == 3


def test_judge_rejects_question_without_facts():
    with pytest.raises(ValueError, match="no answer facts"):
        judge(_question("q1", ()), "ans", lambda m, s: _reply([]))


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


def test_make_probes_ignore_prohibitions_when_halving_facts():
    q = _question("p", ("s1", "s2", "The answer must not say X.", "The answer must not say Y."))
    probes = {kind: (answer, expected) for _, kind, answer, expected in make_probes([q, q])[:3]}
    assert probes["abstain"][1] == 0.0  # silence earns nothing from the prohibitions
    assert probes["partial"] == ("s1", 0.5)  # one of the two statements


def test_judge_treats_a_broken_prohibition_as_incorrect():
    q = _question("q1", ("s1", "The answer must not say X."))
    assert judge(q, "ans", lambda m, s: _reply([True, True])).score == 1.0
    broken = judge(q, "ans", lambda m, s: _reply([True, False]))
    assert broken.completeness == 1.0 and broken.violated and broken.score == 0.0


def test_judge_gives_a_blank_answer_no_credit_for_prohibitions():
    q = _question("q1", ("s1", "The answer must not say X."))
    verdict = judge(q, "", lambda m, s: _reply([True, True]))
    assert verdict.completeness == 0.0 and not verdict.violated and verdict.score == 0.0


def test_selftest_records_unusable_verdicts_and_keeps_going():
    questions = [_question(f"q{i}", ("x", "y")) for i in range(3)]
    calls = []

    def flaky(messages, schema):
        calls.append(1)
        return "garbage" if len(calls) <= 2 else _reply([True, True])  # first probe fails twice

    probes, failures = selftest(questions, flaky, n=3, seed=0)
    assert len(failures) == 1 and "Judge reply is not a verdict" in failures[0]
    assert len(probes) == len(make_probes(questions)) - 1


def test_selftest_and_summary_with_a_perfect_judge():
    questions = [_question(f"q{i}", ("x", "y")) for i in range(4)]

    def perfect(messages, schema):
        user = messages[1]["content"]
        answer = user.split("Candidate answer:\n")[1].split("\n\nEach required fact")[0]
        if answer == "x":  # the "partial" probe: only the first fact
            return _reply([True, False])
        if answer.startswith("gold q") and f"Reference answer: {answer}" in user:
            return _reply([True, True])
        return _reply([False, False])

    probes, failures = selftest(questions, perfect, n=4, seed=1)
    assert failures == []
    summary = summarize_probes(probes)
    assert set(summary) == {"abstain", "mismatch", "partial", "reference"}
    assert all(s["mean_abs_error"] == 0 and s["within_0.25"] == 1 for s in summary.values())
    assert summary["reference"]["n"] == 4
