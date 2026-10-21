"""Retrieve the chunks of technical documentation most relevant to a query."""

import argparse
import json
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, get_args

import faiss

from techrag.bm25 import BM25
from techrag.ingest import EMBEDDING_MODEL, INDEX_DIR, _embed, _load_model

# BGE models are trained to match short queries against longer passages when the query
# carries this instruction. Passages (see ingest) are embedded without it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

Mode = Literal["dense", "bm25", "hybrid"]

# Reciprocal rank fusion damping constant; 60 is the value from the original RRF paper.
RRF_K = 60


@dataclass(frozen=True)
class Result:
    """A retrieved chunk: its source path, its text, and its score for the query.

    The score is the cosine similarity in ``dense`` mode, the BM25 score in ``bm25`` mode and the
    reciprocal rank fusion score in ``hybrid`` mode, so scores are only comparable between
    results of the same query and mode.
    """

    source: Path
    text: str
    score: float


class Retriever:
    """A loaded index that answers many queries without re-reading it from disk.

    Reads the FAISS index and ``chunks.json`` once on construction. The BM25 statistics are built
    on first use and the embedding model is loaded on the first dense or hybrid query;
    :meth:`warm_up` does both up front, e.g. at server startup. Searches only read this state,
    so one instance can serve concurrent callers.

    Raises:
        FileNotFoundError: If ``index_dir`` does not contain an index built by ingest.
        ValueError: If the index and ``chunks.json`` disagree on the number of chunks.
    """

    def __init__(self, index_dir: Path = INDEX_DIR, model_name: str = EMBEDDING_MODEL) -> None:
        index_path = index_dir / "index.faiss"
        chunks_path = index_dir / "chunks.json"
        if not index_path.is_file() or not chunks_path.is_file():
            raise FileNotFoundError(f"No index found in {index_dir}; run techrag-ingest first")
        self._model_name = model_name
        self._index = faiss.read_index(str(index_path))
        self._records = json.loads(chunks_path.read_text(encoding="utf-8"))
        if self._index.ntotal != len(self._records):
            raise ValueError(
                f"Index has {self._index.ntotal} vectors but chunks.json has "
                f"{len(self._records)} chunks; re-run techrag-ingest"
            )

    def __len__(self) -> int:
        """Return the number of chunks in the index."""
        return len(self._records)

    @cached_property
    def _bm25(self) -> BM25:
        return BM25([r["text"] for r in self._records])

    def warm_up(self) -> None:
        """Load the embedding model and build the BM25 statistics now rather than on first use."""
        _load_model(self._model_name)
        _ = self._bm25

    def search(self, query: str, k: int = 5, mode: Mode = "dense") -> list[Result]:
        """Return the ``k`` chunks most relevant to ``query``, best first.

        See :func:`retrieve` for what the modes mean and what is returned.

        Raises:
            ValueError: If ``query`` is blank, ``k`` is not positive or ``mode`` is unknown.
        """
        if not query.strip():
            raise ValueError("Query must not be empty")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if mode not in get_args(Mode):
            raise ValueError(f"mode must be one of {', '.join(get_args(Mode))}, got {mode!r}")

        if mode == "bm25":
            ranked = self._bm25_ranking(query)
        elif mode == "dense":
            ranked = self._dense_ranking(query, k)
        else:
            ranked = _rrf(
                self._dense_ranking(query, self._index.ntotal), self._bm25_ranking(query)
            )
        return [
            Result(
                source=Path(self._records[i]["source"]), text=self._records[i]["text"], score=score
            )
            for i, score in ranked[:k]
        ]

    def _dense_ranking(self, query: str, n: int) -> list[tuple[int, float]]:
        """Return the ``n`` chunk ids most similar to ``query`` with their cosine similarity."""
        vectors = _embed([QUERY_PREFIX + query], self._model_name)
        scores, ids = self._index.search(vectors, min(n, self._index.ntotal))
        return [(int(i), float(s)) for s, i in zip(scores[0], ids[0], strict=True)]

    def _bm25_ranking(self, query: str) -> list[tuple[int, float]]:
        """Return the chunks sharing a term with ``query`` as (id, BM25 score), best first."""
        scores = self._bm25.scores(query)
        ranked = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], i))
        return [(i, scores[i]) for i in ranked]


def retrieve(
    query: str,
    k: int = 5,
    index_dir: Path = INDEX_DIR,
    model_name: str = EMBEDDING_MODEL,
    mode: Mode = "dense",
) -> list[Result]:
    """Return the ``k`` chunks most relevant to ``query``, best first.

    ``mode`` is ``dense`` (embedding similarity), ``bm25`` (keyword match) or ``hybrid`` (the
    dense and BM25 rankings merged by reciprocal rank fusion). Fewer than ``k`` results are
    returned if the index holds fewer than ``k`` chunks, and ``bm25`` returns only chunks
    sharing a term with the query. ``model_name`` must match the model the index was built
    with; ``bm25`` does not use it.

    This loads the index on every call; to answer many queries, build a :class:`Retriever` once
    and call its :meth:`~Retriever.search`.

    Raises:
        ValueError: If ``query`` is blank, ``k`` is not positive, ``mode`` is unknown, or the
            index and ``chunks.json`` in ``index_dir`` disagree on the number of chunks.
        FileNotFoundError: If ``index_dir`` does not contain an index built by ingest.
    """
    return Retriever(index_dir, model_name).search(query, k, mode)


def _rrf(*rankings: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Merge best-first rankings by reciprocal rank fusion, best first.

    A chunk's score is the sum of ``1 / (RRF_K + rank)`` over the rankings it appears in. Ties
    are broken by chunk id so the order is deterministic.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (i, _) in enumerate(ranking, start=1):
            fused[i] = fused.get(i, 0.0) + 1 / (RRF_K + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def main() -> None:
    """Entry point for the ``techrag-retrieve`` command."""
    parser = argparse.ArgumentParser(description="Search the TechRAG index.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5, help="number of chunks to return")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="must match the ingest model")
    parser.add_argument("--mode", choices=get_args(Mode), default="dense", help="ranking method")
    args = parser.parse_args()

    try:
        results = retrieve(args.query, args.k, args.index_dir, args.model, args.mode)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")
    for rank, r in enumerate(results, start=1):
        print(f"[{rank}] {r.source}  (score {r.score:.3f})\n{r.text}\n")


if __name__ == "__main__":
    main()
