import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from techrag.bench import (
    build_subset,
    doc_id_of,
    load_questions,
    read_questions,
    render_document,
    sample_distractors,
    write_questions,
)


def _write_bench(root: Path) -> Path:
    """Write a 6-document benchmark: d1 and d2 are gold, d2 appears twice (a duplicated id)."""
    (root / "data" / "questions").mkdir(parents=True)
    (root / "data" / "documents").mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "question_id": ["q1", "q2"],
                "question_type": ["basic", "info_not_found"],
                "question": ["What is d1?", "What is nothing?"],
                "expected_doc_ids": [["d1", "d2"], []],
                "gold_answer": ["d1 is one.", "Not found."],
                "answer_facts": [["says one"], []],
            }
        ),
        root / "data" / "questions" / "test.parquet",
    )
    rows = [
        ("d1", "slack", "One", "first"),
        ("d2", "jira", "Two", "second"),
        ("d2", "jira", "Two again", "second, other content"),
        ("d3", "slack", "Three", "third"),
        ("d4", "gmail", "Four", "fourth"),
        ("d5", "gmail", "Five", "fifth"),
    ]
    pq.write_table(
        pa.table(
            {
                "doc_id": [r[0] for r in rows],
                "source_type": [r[1] for r in rows],
                "title": [r[2] for r in rows],
                "content": [r[3] for r in rows],
            }
        ),
        root / "data" / "documents" / "test.parquet",
    )
    return root


def test_load_questions(tmp_path: Path) -> None:
    questions = load_questions(_write_bench(tmp_path))

    assert [q.id for q in questions] == ["q1", "q2"]
    assert questions[0].expected_doc_ids == ("d1", "d2")
    assert questions[0].answer_facts == ("says one",)
    assert questions[1].type == "info_not_found"
    assert questions[1].expected_doc_ids == ()


def test_load_questions_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Questions not found"):
        load_questions(tmp_path)


def test_doc_id_of_strips_directory_extension_and_duplicate_suffix() -> None:
    assert doc_id_of("jira/dsid_ab.md") == "dsid_ab"
    assert doc_id_of(Path("jira/dsid_ab~1.md")) == "dsid_ab"


def test_render_document_puts_title_first() -> None:
    assert render_document("Title", "Body") == "# Title\n\nBody\n"


def test_sample_distractors_is_seeded_and_capped() -> None:
    assert sample_distractors(list(range(100)), 10, seed=1) == sample_distractors(
        list(range(100)), 10, seed=1
    )
    assert sample_distractors(list(range(5)), 10, seed=1) == set(range(5))


def test_build_subset_writes_gold_and_distractors(tmp_path: Path) -> None:
    bench = _write_bench(tmp_path / "bench")
    out = tmp_path / "out"

    manifest = build_subset(bench, out, distractors=2, seed=0)

    assert manifest["gold_documents"] == 3  # d1 and both d2 rows
    assert manifest["distractors"] == 2
    assert manifest["documents"] == 5
    assert (out / "slack" / "d1.md").read_text(encoding="utf-8") == "# One\n\nfirst\n"
    # The duplicated id keeps both documents, in separate files.
    assert (out / "jira" / "d2.md").read_text(encoding="utf-8").startswith("# Two\n")
    assert (out / "jira" / "d2~1.md").read_text(encoding="utf-8").startswith("# Two again\n")
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == manifest
    distractor_files = {p.stem for p in out.rglob("*.md")} - {"d1", "d2", "d2~1"}
    assert len(distractor_files) == 2 and distractor_files <= {"d3", "d4", "d5"}


def test_build_subset_with_zero_distractors_writes_only_gold(tmp_path: Path) -> None:
    manifest = build_subset(_write_bench(tmp_path / "bench"), tmp_path / "out", distractors=0)

    assert manifest["documents"] == manifest["gold_documents"] == 3


def test_build_subset_rejects_negative_distractors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="negative"):
        build_subset(_write_bench(tmp_path / "bench"), tmp_path / "out", distractors=-1)


def test_build_subset_missing_gold_document_raises(tmp_path: Path) -> None:
    bench = _write_bench(tmp_path / "bench")
    pq.write_table(
        pa.table({"doc_id": ["d3"], "source_type": ["slack"], "title": ["T"], "content": ["c"]}),
        bench / "data" / "documents" / "test.parquet",
    )

    with pytest.raises(ValueError, match="gold documents are not in the corpus"):
        build_subset(bench, tmp_path / "out")


def test_read_questions_round_trips_write_questions(tmp_path: Path) -> None:
    questions = load_questions(_write_bench(tmp_path / "bench"))
    path = tmp_path / "q.jsonl"
    write_questions(questions, path)
    assert read_questions(path) == questions


def test_read_questions_rejects_bad_line_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text('{"question_id": "q1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":1:"):
        read_questions(path)
    with pytest.raises(FileNotFoundError):
        read_questions(tmp_path / "nope.jsonl")


def test_build_subset_writes_questions_jsonl(tmp_path: Path) -> None:
    bench = _write_bench(tmp_path / "bench")
    out = tmp_path / "out"

    build_subset(bench, out, distractors=0)

    lines = (out / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {
            "question_id": "q1",
            "question_type": "basic",
            "question": "What is d1?",
            "expected_doc_ids": ["d1", "d2"],
            "gold_answer": "d1 is one.",
            "answer_facts": ["says one"],
        },
        {
            "question_id": "q2",
            "question_type": "info_not_found",
            "question": "What is nothing?",
            "expected_doc_ids": [],
            "gold_answer": "Not found.",
            "answer_facts": [],
        },
    ]


def test_write_questions_keeps_non_ascii_readable(tmp_path: Path) -> None:
    question = load_questions(_write_bench(tmp_path / "bench"))[0]
    question = type(question)(question.id, question.type, "Café — d1?", (), "a", ())

    write_questions([question], tmp_path / "q.jsonl")

    assert "Café — d1?" in (tmp_path / "q.jsonl").read_text(encoding="utf-8")
