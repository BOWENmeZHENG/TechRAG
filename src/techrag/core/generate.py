"""Answer questions about the indexed documentation with a local Ollama model."""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from techrag.common.config import EMBEDDING_MODEL, INDEX_DIR, LLM_MODEL, OLLAMA_URL
from techrag.common.log import setup_logging
from techrag.common.ollama import ollama_chat
from techrag.core.retrieve import Mode, Result, retrieve

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You answer questions about a company's internal documents: Slack threads, emails, Jira and "
    "Linear tickets, Confluence pages, GitHub issues and pull requests, Google Drive files, "
    "HubSpot records and meeting transcripts. Use only the numbered context passages provided; "
    "each is an excerpt of one document and is labelled with its source.\n\n"
    "Give every specific the question asks for (names, numbers, versions, dates, steps), "
    "combining facts from several passages when the question needs them, and nothing the "
    "passages do not support; do not pad the answer with unrelated detail. Cite the passages you "
    "rely on like [1] or [2][3]. If passages disagree, say so and cite both rather than picking "
    "one silently.\n\n"
    "If the passages do not contain the answer, or only part of it, say plainly that you cannot "
    "fully answer from the available documentation, and state what is missing and what you did "
    "find. Do not guess."
)


@dataclass(frozen=True)
class Answer:
    """A generated answer and the retrieved chunks it was generated from, best first."""

    text: str
    sources: list[Result]


def build_messages(query: str, results: Sequence[Result]) -> list[dict[str, str]]:
    """Return the chat messages asking the model to answer ``query`` from ``results``.

    Passages are numbered from 1 in the order given, so a citation like ``[2]`` refers to
    ``results[1]``.

    Args:
        query: The user's question.
        results: Retrieved chunks, best first, to use as numbered context passages.

    Returns:
        A system message with the instructions followed by a user message with the passages
        and the question, in the format Ollama's chat API expects.
    """
    context = "\n\n".join(
        f"[{i}] (source: {r.source})\n{r.text}" for i, r in enumerate(results, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {query}"},
    ]


def answer(
    query: str,
    k: int = 5,
    chat: Callable[[Sequence[dict[str, str]]], str] = ollama_chat,
    index_dir: Path = INDEX_DIR,
    embedding_model: str = EMBEDDING_MODEL,
    mode: Mode = "dense",
) -> Answer:
    """Retrieve the ``k`` chunks most relevant to ``query`` and answer from them.

    ``chat`` takes the chat messages and returns the model's reply; it defaults to
    :func:`ollama_chat` with the default model and server. ``mode`` selects how chunks are
    retrieved (``dense``, ``bm25`` or ``hybrid``); with ``bm25`` the model gets no passages if
    none share a term with ``query``.

    Args:
        query: The user's question.
        k: Number of chunks to retrieve and give the model.
        chat: Function that takes the chat messages and returns the model's reply.
        index_dir: Directory holding the index written by ingest.
        embedding_model: Model used to embed the query. Must match the one the index was
            built with.
        mode: Retrieval method: ``"dense"``, ``"bm25"`` or ``"hybrid"``.

    Returns:
        The model's reply and the chunks it was given.

    Raises:
        ValueError, FileNotFoundError: As for :func:`techrag.core.retrieve.retrieve`.
        OSError, ValueError: If ``chat`` fails (the default does on an unreachable server).
    """
    results = retrieve(query, k, index_dir, embedding_model, mode)
    return answer_from(query, results, chat)


def answer_from(
    query: str,
    results: Sequence[Result],
    chat: Callable[[Sequence[dict[str, str]]], str] = ollama_chat,
) -> Answer:
    """Answer ``query`` from already retrieved ``results``, best first.

    Args:
        query: The user's question.
        results: Retrieved chunks, best first.
        chat: Function that takes the chat messages and returns the model's reply.

    Returns:
        The model's reply, stripped of surrounding whitespace, and ``results``.

    Raises:
        OSError, ValueError: If ``chat`` fails (the default does on an unreachable server).
    """
    return Answer(text=chat(build_messages(query, results)).strip(), sources=list(results))


def main() -> None:
    """Entry point for the ``techrag-ask`` command."""
    parser = argparse.ArgumentParser(description="Ask a question about the indexed docs.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5, help="number of chunks to give the model")
    parser.add_argument("--llm", default=LLM_MODEL, help="Ollama model to answer with")
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument(
        "--embedding-model", default=EMBEDDING_MODEL, help="must match the ingest model"
    )
    parser.add_argument("--mode", choices=get_args(Mode), default="dense", help="ranking method")
    args = parser.parse_args()
    setup_logging()

    def chat(messages: Sequence[dict[str, str]]) -> str:
        return ollama_chat(messages, args.llm, args.ollama_url)

    try:
        result = answer(args.query, args.k, chat, args.index_dir, args.embedding_model, args.mode)
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")
    logger.info(result.text)
    logger.info("\nSources:")
    for i, r in enumerate(result.sources, start=1):
        logger.info("[%d] %s  (score %.3f)", i, r.source, r.score)


if __name__ == "__main__":
    main()
