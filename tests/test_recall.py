from pathlib import Path

import pytest

from techrag.bench import BenchQuestion
from techrag.recall import (
    QuestionRanks,
    evaluate,
    format_table,
    gold_doc_ranks,
    summarize,
    summarize_by_type,
)
from techrag.retrieve import Result


def _question(qid: str, expected: tuple[str, ...], qtype: str = "basic") -> BenchQuestion:
    return BenchQuestion(qid, qtype, f"question {qid}", expected, "gold", ())


def _result(source: str) -> Result:
    return Result(Path(source), "text", 1.0)


def test_gold_doc_ranks_uses_first_chunk_of_each_doc_and_maps_source_to_doc_id():
    results = [
        _result("slack/dsid_x.md"),
        _result("jira/dsid_a.md"),
        _result("jira/dsid_a.md"),  # a later chunk of a doc already found
        _result("gmail/dsid_b~1.md"),  # duplicate-id suffix still maps to dsid_b
    ]
    assert gold_doc_ranks(["dsid_a", "dsid_b", "dsid_c"], results) == {
        "dsid_a": 2,
        "dsid_b": 4,
        "dsid_c": None,
    }


def test_recall_and_complete_respect_the_cutoff():
    row = QuestionRanks(_question("q", ("a", "b", "c")), {"a": 1, "b": 7, "c": None})
    assert row.recall(5) == pytest.approx(1 / 3)
    assert row.recall(7) == pytest.approx(2 / 3)
    assert not row.complete(100)
    assert QuestionRanks(_question("q", ("a", "b")), {"a": 3, "b": 10}).complete(10)


def test_evaluate_skips_questions_without_gold_docs():
    questions = [_question("q1", ("a",)), _question("q2", (), "info_not_found")]
    calls = []

    def retriever(query: str, k: int) -> list[Result]:
        calls.append((query, k))
        return [_result("jira/a.md")]

    rows = evaluate(questions, retriever, k=7)
    assert [r.question.id for r in rows] == ["q1"]
    assert rows[0].ranks == {"a": 1}
    assert calls == [("question q1", 7)]


def test_summarize_and_by_type():
    rows = [
        QuestionRanks(_question("q1", ("a",), "basic"), {"a": 1}),
        QuestionRanks(_question("q2", ("b",), "basic"), {"b": None}),
        QuestionRanks(_question("q3", ("c", "d"), "completeness"), {"c": 2, "d": 9}),
    ]
    overall = summarize(rows, ks=(5, 10))
    assert overall["n"] == 3
    assert overall["recall"][5] == pytest.approx((1 + 0 + 0.5) / 3)
    assert overall["complete"][10] == pytest.approx(2 / 3)

    by_type = summarize_by_type(rows, ks=(5, 10))
    assert list(by_type) == ["basic", "completeness"]
    assert by_type["basic"]["recall"][5] == 0.5
    assert by_type["completeness"]["complete"][5] == 0.0
    assert by_type["completeness"]["complete"][10] == 1.0

    table = format_table(overall, by_type, ks=(5, 10))
    assert "basic" in table and "ALL" in table


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([])
