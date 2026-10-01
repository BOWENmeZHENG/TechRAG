"""Ingest technical documentation into the TechRAG index."""

from dataclasses import dataclass
from pathlib import Path


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
    """Split ``text`` into chunks of ``size`` characters overlapping by ``overlap``."""
    raise NotImplementedError


def embed_and_store(chunks: list[str]) -> None:
    """Embed ``chunks`` and write them to the index."""
    raise NotImplementedError


def main() -> None:
    """Entry point for the ``techrag-ingest`` command."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
