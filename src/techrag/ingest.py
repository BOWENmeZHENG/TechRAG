"""Ingest technical documentation into the TechRAG index."""

from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

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


def embed_and_store(chunks: list[str]) -> None:
    """Embed ``chunks`` and write them to the index."""
    raise NotImplementedError


def main() -> None:
    """Entry point for the ``techrag-ingest`` command."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
