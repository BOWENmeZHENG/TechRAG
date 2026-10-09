from pathlib import Path

import pytest

from techrag import evaluate as evaluate_module
from techrag.evaluate import Question, evaluate, first_relevant_rank, load_questions, main
from techrag.retrieve import Result

QUESTION = Question("q", Path("a.md"), "chain rule")


def test_load_questions_skips_blank_lines_and_extra_fields(tmp_path: Path):
    f = tmp_path / "q.jsonl"
    f.write_text(
        '{"query": "q1", "source": "a.md", "evidence": "e1", "extra": 1}\n\n'
        '{"query": "q2", "source": "d/b.md", "evidence": "e2"}\n',
        encoding="utf-8",
    )
    assert load_questions(f) == [
        Question("q1", Path("a.md"), "e1"),
        Question("q2", Path("d/b.md"), "e2"),
    ]


def test_load_questions_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_questions(tmp_path / "nope.jsonl")


@pytest.mark.parametrize("line", ["not json", '{"query": "q", "source": "a.md"}'])
def test_load_questions_rejects_bad_line_with_line_number(tmp_path: Path, line: str):
    f = tmp_path / "q.jsonl"
    f.write_text(f'{{"query": "q", "source": "a.md", "evidence": "e"}}\n{line}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        load_questions(f)


def _result(source: str, text: str) -> Result:
    # 1.0 is dummy score, won't be used by first_relevant_rank
    return Result(source=Path(source), text=text, score=1.0) 


def test_first_relevant_rank_needs_right_source_and_evidence():
    results = [
        _result("b.md", "chain rule"),  # right text, wrong source
        _result("a.md", "something else"),  # right source, no evidence
        _result("a.md", "the chain rule"),
    ]
    assert first_relevant_rank(QUESTION, results) == 3
    assert first_relevant_rank(QUESTION, results[:2]) is None


def test_first_relevant_rank_ignores_case_and_whitespace():
    assert first_relevant_rank(QUESTION, [_result("a.md", "The Chain\n  RULE applies")]) == 1


def test_evaluate_ranks_each_question_in_order_using_k():
    by_query = {
        "hit": [_result("a.md", "chain rule")],
        "miss": [_result("b.md", "chain rule")],
    }
    calls = []

    def retriever(query: str, k: int) -> list[Result]:
        calls.append((query, k))
        return by_query[query]

    questions = [Question(q, Path("a.md"), "chain rule") for q in ("hit", "miss")]
    assert evaluate(questions, k=3, retriever=retriever) == [1, None]
    assert calls == [("hit", 3), ("miss", 3)]


def _write_questions(path: Path) -> Path:
    path.write_text(
        '{"query": "hit", "source": "a.md", "evidence": "chain rule"}\n'
        '{"query": "miss", "source": "a.md", "evidence": "nope"}\n',
        encoding="utf-8",
    )
    return path


def test_main_prints_per_question_ranks_and_metrics(tmp_path: Path, monkeypatch, capsys):
    def fake_retrieve(query: str, k: int, index_dir: Path, model_name: str) -> list[Result]:
        return [_result("a.md", "chain rule")]

    monkeypatch.setattr(evaluate_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        "sys.argv", ["techrag-eval", "-k", "3", "--questions", str(_write_questions(tmp_path / "q"))]
    )
    main()
    out = capsys.readouterr().out
    assert "rank 1  hit" in out
    assert "miss  miss" in out
    assert "hit@3: 0.500" in out
    assert "MRR:    0.500" in out
