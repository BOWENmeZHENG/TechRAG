"""Answer questions about the indexed documentation with a local Ollama model."""

import argparse
import json
import sys
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from techrag.common.config import EMBEDDING_MODEL, INDEX_DIR, LLM_MODEL, OLLAMA_URL
from techrag.core.retrieve import Mode, Result, retrieve

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


def ollama_chat(
    messages: Sequence[dict[str, str]],
    model: str = LLM_MODEL,
    host: str = OLLAMA_URL,
    *,
    format: dict | None = None,
    think: bool | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
) -> str:
    """Send ``messages`` to the Ollama server at ``host`` and return the model's reply.

    ``format`` is a JSON schema the reply must conform to, ``think`` turns a reasoning model's
    thinking on or off, ``temperature`` overrides the model's sampling temperature and ``num_ctx``
    its context window in tokens (prompt and reply together; a reply is cut short when they do
    not fit). Each is left to the server's default when ``None``.

    Args:
        messages: Chat messages, each with ``role`` and ``content`` keys.
        model: Name of the Ollama model to chat with.
        host: Base URL of the Ollama server.
        format: JSON schema the reply must conform to.
        think: Whether a reasoning model should think before replying.
        temperature: Sampling temperature.
        num_ctx: Context window in tokens, prompt and reply together.

    Returns:
        The text of the model's reply.

    Raises:
        OSError: If the server cannot be reached (including a connection refused because
            Ollama is not running) or the request fails.
        ValueError: If the server's response is not a chat reply, e.g. ``model`` is not
            pulled.
    """
    body: dict = {"model": model, "messages": list(messages), "stream": False}
    if format is not None:
        body["format"] = format
    if think is not None:
        body["think"] = think
    options = {
        name: value
        for name, value in (("temperature", temperature), ("num_ctx", num_ctx))
        if value is not None
    }
    if options:
        body["options"] = options
    request_body = json.dumps(body)
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=request_body.encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        response_payload = json.load(response)
    try:
        return response_payload["message"]["content"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"Unexpected response from Ollama: {response_payload!r}") from e


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

    def chat(messages: Sequence[dict[str, str]]) -> str:
        return ollama_chat(messages, args.llm, args.ollama_url)

    try:
        result = answer(args.query, args.k, chat, args.index_dir, args.embedding_model, args.mode)
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")
    print(result.text)
    print("\nSources:")
    for i, r in enumerate(result.sources, start=1):
        print(f"[{i}] {r.source}  (score {r.score:.3f})")


if __name__ == "__main__":
    main()
