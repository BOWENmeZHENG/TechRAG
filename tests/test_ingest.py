import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from techrag import ingest
from techrag.ingest import Document, _embed, chunk, embed_and_store, load_documents


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


class FakeSentenceTransformer:
    """Stand-in for SentenceTransformer: one-hot-ish float64 vectors, no model download."""

    instances: list["FakeSentenceTransformer"] = []
    dim = 4

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encode_kwargs: dict = {}
        FakeSentenceTransformer.instances.append(self)

    def encode(self, chunks: list[str], **kwargs) -> np.ndarray:
        self.encode_kwargs = kwargs
        vectors = np.zeros((len(chunks), self.dim), dtype=np.float64)
        for i in range(len(chunks)):
            vectors[i, i % self.dim] = 3.0
        if kwargs.get("normalize_embeddings"):
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> type[FakeSentenceTransformer]:
    FakeSentenceTransformer.instances = []
    monkeypatch.setattr(ingest, "SentenceTransformer", FakeSentenceTransformer)
    return FakeSentenceTransformer


def test_embed_returns_one_float32_row_per_chunk(fake_model) -> None:
    vectors = _embed(["a", "b", "c"], "any-model")

    assert vectors.shape == (3, fake_model.dim)
    assert vectors.dtype == np.float32


def test_embed_rows_are_l2_normalized(fake_model) -> None:
    vectors = _embed(["a", "b"], "any-model")

    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)


def test_embed_loads_requested_model_and_asks_for_numpy_normalized(fake_model) -> None:
    _embed(["a"], "my-model")

    (model,) = fake_model.instances
    assert model.model_name == "my-model"
    assert model.encode_kwargs == {"normalize_embeddings": True, "convert_to_numpy": True}


def test_embed_and_store_writes_index_and_chunks(tmp_path: Path, fake_model) -> None:
    chunks = ["alpha", "beta", "gamma"]

    embed_and_store(chunks, index_dir=tmp_path)

    index = faiss.read_index(str(tmp_path / "index.faiss"))
    assert index.ntotal == len(chunks)
    assert index.d == fake_model.dim
    assert json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8")) == chunks


def test_embed_and_store_index_row_i_matches_chunk_i(tmp_path: Path, fake_model) -> None:
    chunks = ["alpha", "beta", "gamma"]
    embed_and_store(chunks, index_dir=tmp_path)
    index = faiss.read_index(str(tmp_path / "index.faiss"))

    for i in range(len(chunks)):
        query = np.zeros((1, fake_model.dim), dtype=np.float32)
        query[0, i] = 1.0
        scores, ids = index.search(query, 1)
        assert ids[0][0] == i
        assert scores[0][0] == pytest.approx(1.0)


def test_embed_and_store_creates_missing_nested_index_dir(tmp_path: Path, fake_model) -> None:
    index_dir = tmp_path / "nested" / ".index"

    embed_and_store(["alpha"], index_dir=index_dir)

    assert (index_dir / "index.faiss").is_file()
    assert (index_dir / "chunks.json").is_file()


def test_embed_and_store_passes_model_name_through(tmp_path: Path, fake_model) -> None:
    embed_and_store(["alpha"], index_dir=tmp_path, model_name="custom-model")

    assert [m.model_name for m in fake_model.instances] == ["custom-model"]


def test_embed_and_store_replaces_existing_index(tmp_path: Path, fake_model) -> None:
    embed_and_store(["old1", "old2", "old3"], index_dir=tmp_path)

    embed_and_store(["new"], index_dir=tmp_path)

    index = faiss.read_index(str(tmp_path / "index.faiss"))
    assert index.ntotal == 1
    assert json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8")) == ["new"]


def test_embed_and_store_round_trips_non_ascii_chunks(tmp_path: Path, fake_model) -> None:
    chunks = ["naïve café — 数据"]

    embed_and_store(chunks, index_dir=tmp_path)

    assert json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8")) == chunks


def test_embed_and_store_empty_chunks_raises_and_writes_nothing(
    tmp_path: Path, fake_model
) -> None:
    index_dir = tmp_path / ".index"

    with pytest.raises(ValueError, match="No chunks"):
        embed_and_store([], index_dir=index_dir)

    assert not index_dir.exists()
    assert fake_model.instances == []
