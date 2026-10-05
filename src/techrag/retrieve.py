"""Retrieve the chunks of technical documentation most relevant to a query."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import faiss

from techrag.ingest import EMBEDDING_MODEL, INDEX_DIR, _embed

# BGE models are trained to match short queries against longer passages when the query
# carries this instruction. Passages (see ingest) are embedded without it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class Result:
    """A retrieved chunk: its source path, its text, and its cosine similarity to the query."""

    source: Path
    text: str
    score: float


def retrieve(
    query: str,
    k: int = 5,
    index_dir: Path = INDEX_DIR,
    model_name: str = EMBEDDING_MODEL,
) -> list[Result]:
    """Return the ``k`` chunks most similar to ``query``, best first.

    Fewer than ``k`` results are returned if the index holds fewer than ``k`` chunks.
    ``model_name`` must match the model the index was built with.

    Raises:
        ValueError: If ``query`` is blank, ``k`` is not positive, or the index and
            ``chunks.json`` in ``index_dir`` disagree on the number of chunks.
        FileNotFoundError: If ``index_dir`` does not contain an index built by ingest.
    """
    if not query.strip():
        raise ValueError("Query must not be empty")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
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
    args = parser.parse_args()

    try:
        results = retrieve(args.query, args.k, args.index_dir, args.model)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")
    for rank, r in enumerate(results, start=1):
        print(f"[{rank}] {r.source}  (score {r.score:.3f})\n{r.text}\n")


if __name__ == "__main__":
    main()
