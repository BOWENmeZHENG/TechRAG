from pathlib import Path
import pytest

from techrag.ingest import Document, load_documents, chunk, embed_and_store

def test_loads_markdown_recursively_in_sorted_order(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "two.md").write_text("two", encoding="utf-8")
    (tmp_path / "a.md").write_text("one", encoding="utf-8")

    docs = load_documents(tmp_path)

    assert docs == [
        Document(source=Path("a.md"), text="one"),
        Document(source=Path("b/two.md"), text="two"),
    ]


def test_missing_directory_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError, match="not found"):
        load_documents(missing)


def test_file_path_instead_of_directory_raises_file_not_found(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.md"
    file_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="not found"):
        load_documents(file_path)


def test_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    assert load_documents(tmp_path) == []


def test_chunk_short_text_returns_single_chunk() -> None:
    assert chunk("hello world", size=100, overlap=10) == ["hello world"]


def test_chunk_empty_text_returns_no_chunks() -> None:
    assert chunk("") == []


def test_chunk_does_not_exceed_size() -> None:
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 30 for i in range(10))

    chunks = chunk(text, size=200, overlap=20)

    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_overlaps_adjacent_chunks() -> None:
    text = " ".join(f"w{i}" for i in range(200))

    chunks = chunk(text, size=100, overlap=30)

    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.split()[-1] in nxt.split()


def test_chunk_preserves_all_content() -> None:
    words = [f"w{i}" for i in range(200)]

    chunks = chunk(" ".join(words), size=100, overlap=30)

    assert set(words) <= {w for c in chunks for w in c.split()}