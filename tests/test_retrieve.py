import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from techrag import ingest
from techrag.ingest import Chunk, embed_and_store
from techrag.retrieve import QUERY_PREFIX, _rrf, retrieve


class FakeSentenceTransformer:
    """Stand-in for SentenceTransformer: maps known texts to fixed unit-ish vectors."""

    # Passage "alpha" -> axis 0, "beta" -> axis 1, "gamma" -> axis 2.
    # The query "find beta" (with prefix) lands mostly on axis 1, a bit on axis 2.
    vectors = {
        "alpha": [1.0, 0.0, 0.0],
        "beta": [0.0, 1.0, 0.0],
        "gamma": [0.0, 0.0, 1.0],
        QUERY_PREFIX + "find beta": [0.0, 3.0, 1.0],
    }

    def __init__(self, model_name: str) -> None:
        pass

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        out = np.array([self.vectors[t] for t in texts], dtype=np.float64)
        if kwargs.get("normalize_embeddings"):
            out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(ingest, "SentenceTransformer", FakeSentenceTransformer)
    ingest._load_model.cache_clear()  # don't reuse a model cached by another test
    yield
    ingest._load_model.cache_clear()  # don't leak the fake into other tests


@pytest.fixture
def index_dir(tmp_path: Path, fake_model) -> Path:
    chunks = [
        Chunk(Path("a.md"), "alpha"),
        Chunk(Path("sub/b.md"), "beta"),
        Chunk(Path("sub/b.md"), "gamma"),
    ]
    embed_and_store(chunks, index_dir=tmp_path)
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


def test_bm25_mode_ranks_by_keyword_match_and_skips_non_matching_chunks(index_dir: Path) -> None:
    results = retrieve("gamma", k=3, index_dir=index_dir, mode="bm25")
    assert [r.text for r in results] == ["gamma"]
    assert results[0].source == Path("sub/b.md")
    assert results[0].score > 0


def test_hybrid_mode_ranks_chunk_found_by_both_methods_first(index_dir: Path) -> None:
    # Dense ranks beta, gamma, alpha; BM25 matches only "beta".
    results = retrieve("find beta", k=3, index_dir=index_dir, mode="hybrid")
    print(results)

    assert [r.text for r in results] == ["beta", "gamma", "alpha"]
    assert results[0].source == Path("sub/b.md")
    assert results[0].score == pytest.approx(2 / 61)
    assert results[1].score == pytest.approx(1 / 62)
    assert results[2].score == pytest.approx(1 / 63)


def test_hybrid_mode_fuses_ranks_from_both_rankings() -> None:
    dense = [(0, 0.9), (1, 0.5), (2, 0.1)]
    bm25 = [(2, 7.0), (1, 3.0)]

    fused = _rrf(dense, bm25)

    # id 1 is rank 2 and 2; id 2 is rank 3 and 1; id 0 is rank 1 in one ranking only.
    assert [i for i, _ in fused] == [1, 2, 0]
    assert dict(fused)[1] == pytest.approx(2 / 62)
    assert dict(fused)[2] == pytest.approx(1 / 63 + 1 / 61)
    assert dict(fused)[0] == pytest.approx(1 / 61)


def test_hybrid_mode_respects_k(index_dir: Path) -> None:
    assert len(retrieve("find beta", k=2, index_dir=index_dir, mode="hybrid")) == 2


def test_unknown_mode_raises(index_dir: Path) -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        retrieve("find beta", index_dir=index_dir, mode="fuzzy")  # type: ignore[arg-type]


@pytest.mark.parametrize("query", ["", "   \n"])
def test_blank_query_raises(index_dir: Path, query: str) -> None:
    with pytest.raises(ValueError, match="Query"):
        retrieve(query, index_dir=index_dir)


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_raises(index_dir: Path, k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        retrieve("find beta", k=k, index_dir=index_dir)


def test_missing_index_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="techrag-ingest"):
        retrieve("find beta", index_dir=tmp_path)


def test_index_and_chunks_out_of_sync_raises(index_dir: Path) -> None:
    (index_dir / "chunks.json").write_text(
        json.dumps([{"source": "a.md", "text": "alpha"}]), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="re-run techrag-ingest"):
        retrieve("find beta", index_dir=index_dir)
