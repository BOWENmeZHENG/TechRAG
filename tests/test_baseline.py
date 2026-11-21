import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from techrag.benchmarks.bench import BenchQuestion
from techrag.core.retrieve import Result
from techrag.eval.judge import Verdict
from techrag.reasoning.baseline import run, run_question
from techrag.reasoning.harness import Record


def _question(qid: str, expected: tuple[str, ...] = ("d1",), qtype: str = "basic") -> BenchQuestion:
    return BenchQuestion(qid, qtype, f"question {qid}?", expected, f"gold {qid}", ("fact one",))


def _results(*doc_ids: str) -> list[Result]:
    return [Result(Path(f"jira/{d}.md"), f"text of {d}", 1.0) for d in doc_ids]


def _judge_reply(met: bool, contradiction: str = "") -> str:
    return json.dumps(
        {"facts": [{"evidence": "e", "met": met}], "contradiction": contradiction, "reason": "r"}
    )


def _judge(met: bool = True, contradiction: str = ""):
    def chat(messages: Sequence[dict[str, str]], schema: dict) -> str:
        return _judge_reply(met, contradiction)

    return chat


def _answer(messages: Sequence[dict[str, str]]) -> str:
    return " an answer "


def _record(qid: str, score_met: bool | None, qtype: str = "basic", gold: float | None = 1.0):
    verdict = Verdict((score_met,), "", "r", (False,)) if score_met is not None else None
    error = None if verdict else "judge: boom"
    return Record(qid, qtype, "a", ("d1",), gold, 1.0, verdict, error)


def test_run_question_answers_and_grades():
    q = _question("q")
    record = run_question(q, lambda text, k: _results("d1", "x"), _answer, _judge(True), 2)
    assert record.error is None
    assert record.answer == "an answer"
    assert record.sources == ("d1", "x")
    assert record.gold_found == 1.0
    assert record.score == 1.0


def test_run_question_passes_k_to_retriever():
    seen = []
    run_question(_question("q"), lambda text, k: seen.append(k) or [], _answer, _judge(), 7)
    assert seen == [7]


def test_run_question_records_unusable_verdict_as_error():
    def bad_judge(messages, schema):
        return "not json"

    record = run_question(_question("q"), lambda t, k: _results("d1"), _answer, bad_judge, 1)
    assert record.verdict is None
    assert record.score is None
    assert record.error is not None and record.error.startswith("judge:")
    assert record.answer == "an answer"


def test_run_question_records_timeout_as_error():
    def slow(messages):
        raise TimeoutError("timed out")

    record = run_question(_question("q"), lambda t, k: _results("d1"), slow, _judge(), 1)
    assert record.error == "answer: timed out"
    assert record.verdict is None


def test_run_question_lets_unreachable_server_abort():
    def down(messages):
        raise ConnectionRefusedError("refused")

    with pytest.raises(OSError):
        run_question(_question("q"), lambda t, k: _results("d1"), down, _judge(), 1)


def test_run_skips_done_questions_and_retries_errors():
    questions = [_question("a"), _question("b"), _question("c")]
    done = {"a": _record("a", True), "b": _record("b", None)}
    saved = []
    new = run(questions, lambda t, k: _results("d1"), _answer, _judge(), 1, done, saved.append)
    assert [r.question_id for r in new] == ["b", "c"]
    assert saved == new
