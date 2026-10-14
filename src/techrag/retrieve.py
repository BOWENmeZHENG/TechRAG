"""Retrieve the chunks of technical documentation most relevant to a query."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import faiss

from techrag.bm25 import BM25
from techrag.ingest import EMBEDDING_MODEL, INDEX_DIR, _embed

# BGE models are trained to match short queries against longer passages when the query
# carries this instruction. Passages (see ingest) are embedded without it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

Mode = Literal["dense", "bm25"]


@dataclass(frozen=True)
class Result:
    """A retrieved chunk: its source path, its text, and its score for the query.

    The score is the cosine similarity in ``dense`` mode and the BM25 score in ``bm25`` mode,
    so scores are only comparable between results of the same query and mode.
    """

    source: Path
    text: str
    score: float


def retrieve(
    query: str,
    k: int = 5,
    index_dir: Path = INDEX_DIR,
    model_name: str = EMBEDDING_MODEL,
    mode: Mode = "dense",
) -> list[Result]:
    """Return the ``k`` chunks most relevant to ``query``, best first.

    ``mode`` is ``dense`` (embedding similarity) or ``bm25`` (keyword match). Fewer than ``k``
    results are returned if the index holds fewer than ``k`` chunks, and ``bm25`` returns only
    chunks sharing a term with the query. ``model_name`` must match the model the index was
    built with; ``bm25`` does not use it.

    Raises:
        ValueError: If ``query`` is blank, ``k`` is not positive, ``mode`` is unknown, or the
            index and ``chunks.json`` in ``index_dir`` disagree on the number of chunks.
        FileNotFoundError: If ``index_dir`` does not contain an index built by ingest.
    """
    if not query.strip():
        raise ValueError("Query must not be empty")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if mode not in get_args(Mode):
        raise ValueError(f"mode must be one of {', '.join(get_args(Mode))}, got {mode!r}")
    index_path = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.json"
    if not index_path.is_file() or not chunks_path.is_file():
        raise FileNotFoundError(f"No index found in {index_dir}; run techrag-ingest first")

    index = faiss.read_index(str(index_path))
    records = json.loads(chunks_path.read_text(encoding="utf-8"))
    if index.ntotal != len(records):
        raise ValueError(
            f"Index has {index.ntotal} vectors but chunks.json has {len(records)} chunks; "
            "re-run techrag-ingest"
        )

    if mode == "bm25":
        scores = BM25([r["text"] for r in records]).scores(query)
        ranked = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], i))
        return [
            Result(source=Path(records[i]["source"]), text=records[i]["text"], score=scores[i])
            for i in ranked[:k]
        ]

    vectors = _embed([QUERY_PREFIX + query], model_name)
    scores, ids = index.search(vectors, min(k, index.ntotal))
    return [
        Result(source=Path(records[i]["source"]), text=records[i]["text"], score=float(s))
        for s, i in zip(scores[0], ids[0], strict=True)
    ]


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
