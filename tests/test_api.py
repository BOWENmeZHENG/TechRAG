from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from techrag import ingest
from techrag.api import create_app
from techrag.ingest import Chunk, embed_and_store
from techrag.retrieve import QUERY_PREFIX, Retriever


class FakeSentenceTransformer:
    """Stand-in for SentenceTransformer: maps known texts to fixed vectors."""

    vectors = {
        "alpha": [1.0, 0.0, 0.0],
        "beta": [0.0, 1.0, 0.0],
        "gamma": [0.0, 0.0, 1.0],
        QUERY_PREFIX + "find beta": [0.0, 3.0, 1.0],
    }
    loads = 0

    def __init__(self, model_name: str) -> None:
        type(self).loads += 1

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        out = np.array([self.vectors[t] for t in texts], dtype=np.float64)
        if kwargs.get("normalize_embeddings"):
            out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(ingest, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(FakeSentenceTransformer, "loads", 0)
    ingest._load_model.cache_clear()  # don't reuse a model cached by another test
    yield
    ingest._load_model.cache_clear()  # don't leak the fake into other tests


@pytest.fixture
def retriever(tmp_path: Path, fake_model) -> Retriever:
    chunks = [
        Chunk(Path("a.md"), "alpha"),
        Chunk(Path("sub/b.md"), "beta"),
        Chunk(Path("sub/b.md"), "gamma"),
    ]
    embed_and_store(chunks, index_dir=tmp_path)
    retriever = Retriever(tmp_path)
    retriever.warm_up()
    return retriever


def fake_chat(messages: Sequence[dict[str, str]]) -> str:
    return f"  answered from {len(messages)} messages [1]  "


@pytest.fixture
def client(retriever: Retriever) -> TestClient:
    return TestClient(create_app(retriever, fake_chat))


def test_healthz_reports_chunk_count(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "chunks": 3}


def test_search_returns_best_match_first(client: TestClient) -> None:
    response = client.post("/search", json={"query": "find beta", "k": 2})

    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["text"] for r in results] == ["beta", "gamma"]
    assert results[0]["source"] == "sub/b.md"
    assert results[0]["score"] == pytest.approx(3 / np.sqrt(10))


def test_search_defaults_to_dense_with_k_5(client: TestClient) -> None:
    response = client.post("/search", json={"query": "find beta"})

    assert [r["text"] for r in response.json()["results"]] == ["beta", "gamma", "alpha"]


def test_search_honours_mode(client: TestClient) -> None:
    response = client.post("/search", json={"query": "gamma", "mode": "bm25"})

    assert [r["text"] for r in response.json()["results"]] == ["gamma"]


def test_ask_returns_stripped_answer_and_sources(client: TestClient) -> None:
    response = client.post("/ask", json={"query": "find beta", "k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "answered from 2 messages [1]"
    assert [s["text"] for s in body["sources"]] == ["beta", "gamma"]
    assert body["sources"][0]["source"] == "sub/b.md"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "find beta", "k": 0},
        {"query": "find beta", "mode": "nope"},
    ],
)
@pytest.mark.parametrize("path", ["/search", "/ask"])
def test_invalid_request_is_rejected(client: TestClient, path: str, body: dict) -> None:
    assert client.post(path, json=body).status_code == 422


@pytest.mark.parametrize("error", [ConnectionRefusedError("refused"), ValueError("bad reply")])
def test_ask_returns_502_when_the_model_fails(retriever: Retriever, error: Exception) -> None:
    def failing_chat(messages: Sequence[dict[str, str]]) -> str:
        raise error

    client = TestClient(create_app(retriever, failing_chat))

    response = client.post("/ask", json={"query": "find beta"})

    assert response.status_code == 502
    assert str(error) in response.json()["detail"]


def test_model_is_loaded_once_across_requests(client: TestClient) -> None:
    # The retriever fixture already warmed the model up: later requests must reuse it.
    assert FakeSentenceTransformer.loads == 1

    for _ in range(3):
        client.post("/search", json={"query": "find beta"})

    assert FakeSentenceTransformer.loads == 1
