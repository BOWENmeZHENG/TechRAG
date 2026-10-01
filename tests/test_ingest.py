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