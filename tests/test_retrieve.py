import json
from pathlib import Path

import numpy as np
import pytest

from techrag import ingest
from techrag.ingest import Chunk, embed_and_store
from techrag.retrieve import QUERY_PREFIX, retrieve


class FakeSentenceTransformer:
    """Stand-in for SentenceTransformer: maps known texts to fixed unit-ish vectors."""

    instances: list["FakeSentenceTransformer"] = []
    # Passage "alpha" -> axis 0, "beta" -> axis 1, "gamma" -> axis 2.
    # The query "find beta" (with prefix) lands mostly on axis 1, a bit on axis 2.
    vectors = {
        "alpha": [1.0, 0.0, 0.0],
        "beta": [0.0, 1.0, 0.0],
        "gamma": [0.0, 0.0, 1.0],
        QUERY_PREFIX + "find beta": [0.0, 3.0, 1.0],
    }

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encoded: list[str] = []
        FakeSentenceTransformer.instances.append(self)

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        self.encoded.extend(texts)
        out = np.array([self.vectors[t] for t in texts], dtype=np.float64)
        if kwargs.get("normalize_embeddings"):
            out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> type[FakeSentenceTransformer]:
    FakeSentenceTransformer.instances = []
    monkeypatch.setattr(ingest, "SentenceTransformer", FakeSentenceTransformer)
    return FakeSentenceTransformer


@pytest.fixture
def index_dir(tmp_path: Path, fake_model) -> Path:
    chunks = [
        Chunk(Path("a.md"), "alpha"),
        Chunk(Path("sub/b.md"), "beta"),
        Chunk(Path("sub/b.md"), "gamma"),
    ]
    embed_and_store(chunks, index_dir=tmp_path)
    fake_model.instances.clear()
    return tmp_path


def test_returns_best_match_first_with_source_and_score(index_dir: Path) -> None:
    results = retrieve("find beta", k=2, index_dir=index_dir)

    assert [r.text for r in results] == ["beta", "gamma"]
    assert results[0].source == Path("sub/b.md")
    assert results[0].score == pytest.approx(3 / np.sqrt(10))
    assert results[0].score > results[1].score


def test_k_larger_than_index_returns_every_chunk(index_dir: Path) -> None:
    results = retrieve("find beta", k=10, index_dir=index_dir)

    assert sorted(r.text for r in results) == ["alpha", "beta", "gamma"]


def test_query_is_prefixed_and_model_name_passed_through(index_dir: Path, fake_model) -> None:
    retrieve("find beta", index_dir=index_dir, model_name="custom-model")

    (model,) = fake_model.instances
    assert model.model_name == "custom-model"
    assert model.encoded == [QUERY_PREFIX + "find beta"]


@pytest.mark.parametrize("query", ["", "   \n"])
def test_blank_query_raises(index_dir: Path, query: str) -> None:
    with pytest.raises(ValueError, match="Query"):
        retrieve(query, index_dir=index_dir)


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_raises(index_dir: Path, k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        retrieve("find beta", k=k, index_dir=index_dir)


def test_missing_index_raises_file_not_found(tmp_path: Path, fake_model) -> None:
    with pytest.raises(FileNotFoundError, match="techrag-ingest"):
        retrieve("find beta", index_dir=tmp_path)

    assert fake_model.instances == []


def test_index_and_chunks_out_of_sync_raises(index_dir: Path) -> None:
    (index_dir / "chunks.json").write_text(
        json.dumps([{"source": "a.md", "text": "alpha"}]), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="re-run techrag-ingest"):
        retrieve("find beta", index_dir=index_dir)
