from pathlib import Path

import pytest

from techrag.evaluate import Question, load_questions


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
