"""Ingest technical documentation into the TechRAG index."""

from pathlib import Path


def load_documents(path: Path) -> list[str]:
    """Load the raw text of each document under ``path``."""
    raise NotImplementedError


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
