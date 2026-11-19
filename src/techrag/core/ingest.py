"""Ingest documentation into the TechRAG index."""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import faiss
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from techrag.common.config import EMBEDDING_MODEL, INDEX_DIR, SUBSET_DIR
from techrag.common.log import setup_logging
from techrag.core.embed import embed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    """A source document: its path relative to the data dir and its raw text."""

    source: Path
    text: str


@dataclass(frozen=True)
class Chunk:
    """A piece of a document: the text and the path (relative to the data dir) it came from."""

    source: Path
    text: str


def load_documents(path: Path) -> list[Document]:
    """Load every Markdown file under ``path``, in sorted order.

    Args:
        path: Directory to search recursively for ``*.md`` files.

    Returns:
        One :class:`Document` per file, each with its path relative to ``path`` and its
        UTF-8 text.

    Raises:
        FileNotFoundError: If ``path`` is not an existing directory.
    """
    if not path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {path}")
    return [
        Document(source=f.relative_to(path), text=f.read_text(encoding="utf-8"))
        for f in sorted(path.rglob("*.md"))
    ]


def chunk(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Split Markdown ``text`` into chunks of at most ``size`` characters.

    Splits prefer heading, code-fence and paragraph boundaries, falling back to finer
    separators only when a section is too large. Adjacent chunks may share up to
    ``overlap`` characters of context.

    Args:
        text: Markdown source to split.
        size: Maximum characters per chunk. Must be positive.
        overlap: Maximum characters shared by adjacent chunks. Must be non-negative and
            smaller than ``size``.

    Returns:
        The chunks in document order, empty if ``text`` is blank.

    Raises:
        ValueError: If ``size`` is not positive, or ``overlap`` is negative or not
            smaller than ``size``.
    """
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN, chunk_size=size, chunk_overlap=overlap
    )
    return splitter.split_text(text)


def embed_and_store(
    chunks: list[Chunk], index_dir: Path = INDEX_DIR, model_name: str = EMBEDDING_MODEL
) -> None:
    """Embed ``chunks`` and write them to a FAISS index under ``index_dir``.

    Writes ``index.faiss`` (inner-product index over normalized vectors, i.e. cosine
    similarity) and ``chunks.json`` (a list of ``{"source", "text"}`` objects, where
    row ``i`` of the index corresponds to ``chunks[i]``). Any existing index in
    ``index_dir`` is replaced.

    Args:
        chunks: Chunks to index. Must not be empty.
        index_dir: Directory to write the index into; created if missing.
        model_name: sentence-transformers model to embed with. Queries must later use the
            same model.

    Raises:
        ValueError: If ``chunks`` is empty.
    """
    if not chunks:
        raise ValueError("No chunks to embed")
    vectors = embed([c.text for c in chunks], model_name)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    records = [{"source": c.source.as_posix(), "text": c.text} for c in chunks]
    (index_dir / "chunks.json").write_text(json.dumps(records), encoding="utf-8")


def main() -> None:
    """Entry point for the ``techrag-ingest`` command."""
    parser = argparse.ArgumentParser(description="Build the TechRAG index from Markdown docs.")
    parser.add_argument("data_dir", nargs="?", type=Path, default=SUBSET_DIR)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--size", type=int, default=500, help="max characters per chunk")
    parser.add_argument("--overlap", type=int, default=50, help="characters shared by chunks")
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="sentence-transformers model")
    args = parser.parse_args()
    setup_logging()

    try:
        documents = load_documents(args.data_dir)
        chunks = [
            Chunk(d.source, text)
            for d in documents
            for text in chunk(d.text, args.size, args.overlap)
        ]
        embed_and_store(chunks, args.index_dir, args.model)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")
    logger.info(
        "Indexed %d chunks from %d documents into %s", len(chunks), len(documents), args.index_dir
    )


if __name__ == "__main__":
    main()
