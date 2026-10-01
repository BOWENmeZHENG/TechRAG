"""Ingest technical documentation into the TechRAG index."""

import json
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
    chunks: list[str], index_dir: Path = INDEX_DIR, model_name: str = EMBEDDING_MODEL
) -> None:
    """Embed ``chunks`` and write them to a FAISS index under ``index_dir``.

    Writes ``index.faiss`` (inner-product index over normalized vectors, i.e. cosine
    similarity) and ``chunks.json`` (the chunk texts, where row ``i`` of the index
    corresponds to ``chunks[i]``). Any existing index in ``index_dir`` is replaced.

    Raises:
        ValueError: If ``chunks`` is empty.
    """
    if not chunks:
        raise ValueError("No chunks to embed")
    vectors = _embed(chunks, model_name)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")


def main() -> None:
    """Entry point for the ``techrag-ingest`` command."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
