import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from techrag.benchmarks.bench import BenchQuestion
from techrag.core.retrieve import Result
from techrag.eval.judge import Verdict
from techrag.reasoning.harness import (
    Record,
    gold_found,
    load_records,
    run_each,
    summarize,
    summarize_by,
)


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


def test_gold_found_is_share_of_gold_documents_among_results():
    q = _question("q", expected=("d1", "d2"))
    assert gold_found(q, _results("d1", "x")) == 0.5
    assert gold_found(q, _results("d2", "d1")) == 1.0
    assert gold_found(q, []) == 0.0


def test_gold_found_is_none_without_gold_documents():
    assert gold_found(_question("q", expected=()), _results("d1")) is None


def test_record_json_round_trip():
    record = Record(
        "q1",
        "basic",
        "ans",
        ("d1", "d2"),
        0.5,
        1.234,
        Verdict((True, False), "c", "r", (False, True)),
    )
    assert Record.from_json(record.to_json()) == Record(
        "q1", "basic", "ans", ("d1", "d2"), 0.5, 1.23, record.verdict
    )


def test_record_json_round_trip_keeps_agent_fields_and_reads_older_records():
    trace = ({"step": 1, "tool": "search", "args": {"query": "q"}, "docs": ["d1"]},)
    record = Record(
        "q1", "basic", "ans", ("d1",), 1.0, 2.0, None, "judge: x", 3, 900, "answered", trace
    )
    assert Record.from_json(record.to_json()) == record
    old = json.loads(record.to_json())
    for key in ("steps", "tokens", "stop", "trace"):
        del old[key]
    assert Record.from_json(json.dumps(old)).steps == 0


def test_run_each_skips_graded_questions_only():
    done = {"a": _record("a", True), "b": _record("b", None)}
    new = run_each([_question("a"), _question("b")], lambda q: _record(q.id, True), done)
    assert [r.question_id for r in new] == ["b"]


def test_record_from_json_rejects_garbage():
    with pytest.raises(ValueError):
        Record.from_json('{"question_id": "q"}')


def test_load_records_missing_file_is_empty(tmp_path):
    assert load_records(tmp_path / "none.jsonl") == {}


def test_load_records_latest_record_wins_and_names_bad_line(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(_record("q", None).to_json() + "\n" + _record("q", True).to_json() + "\n")
    assert load_records(path)["q"].error is None
    path.write_text(_record("q", True).to_json() + "\nnot json\n")
    with pytest.raises(ValueError, match=r"r\.jsonl:2"):
        load_records(path)


def test_summarize_means_cover_graded_records_only():
    s = summarize([_record("a", True), _record("b", False), _record("c", None)])
    assert s["n"] == 3
    assert s["errors"] == 1
    assert s["score"] == 0.5
    assert s["correctness"] == 1.0
    assert s["completeness"] == 0.5


def test_summarize_with_nothing_graded_is_nan():
    s = summarize([_record("a", None)])
    assert s["errors"] == 1
    assert math.isnan(s["score"])


def test_summarize_by_groups_in_name_order():
    records = [_record("a", True, "z"), _record("b", False, "m"), _record("c", True, "z")]
    by_type = summarize_by(records, lambda r: r.question_type)
    assert list(by_type) == ["m", "z"]
    assert by_type["z"]["n"] == 2 and by_type["z"]["score"] == 1.0


def test_retrieval_group():
    assert _record("a", True, gold=1.0).retrieval_group == "all gold found"
    assert _record("a", True, gold=0.5).retrieval_group == "some gold missing"
    assert _record("a", True, gold=None).retrieval_group == "no gold documents"
