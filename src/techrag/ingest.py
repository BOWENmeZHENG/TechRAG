"""Ingest technical documentation into the TechRAG index."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
INDEX_DIR = Path(".index")


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

    Raises:
        ValueError: If ``size`` is not positive, or ``overlap`` is negative or not
            smaller than ``size``.
    """
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN, chunk_size=size, chunk_overlap=overlap
    )
    return splitter.split_text(text)


def _embed(chunks: list[str], model_name: str) -> np.ndarray:
    """Return L2-normalized float32 embeddings for ``chunks``, one row per chunk."""
    model = SentenceTransformer(model_name)
    return model.encode(chunks, normalize_embeddings=True, convert_to_numpy=True).astype(
        np.float32
    )


def embed_and_store(
    chunks: list[Chunk], index_dir: Path = INDEX_DIR, model_name: str = EMBEDDING_MODEL
) -> None:
    """Embed ``chunks`` and write them to a FAISS index under ``index_dir``.

    Writes ``index.faiss`` (inner-product index over normalized vectors, i.e. cosine
    similarity) and ``chunks.json`` (a list of ``{"source", "text"}`` objects, where
    row ``i`` of the index corresponds to ``chunks[i]``). Any existing index in
    ``index_dir`` is replaced.

    Raises:
        ValueError: If ``chunks`` is empty.
    """
    if not chunks:
        raise ValueError("No chunks to embed")
    vectors = _embed([c.text for c in chunks], model_name)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    records = [{"source": c.source.as_posix(), "text": c.text} for c in chunks]
    (index_dir / "chunks.json").write_text(json.dumps(records), encoding="utf-8")


def main() -> None:
    """Entry point for the ``techrag-ingest`` command."""
    parser = argparse.ArgumentParser(description="Build the TechRAG index from Markdown docs.")
    parser.add_argument("data_dir", nargs="?", type=Path, default=Path("data"))
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--size", type=int, default=500, help="max characters per chunk")
    parser.add_argument("--overlap", type=int, default=50, help="characters shared by chunks")
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="sentence-transformers model")
    args = parser.parse_args()

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
    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents into {args.index_dir}")


if __name__ == "__main__":
    main()
