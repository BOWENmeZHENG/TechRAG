"""Answer questions about the technical documentation with a local Ollama model."""

import argparse
import json
import sys
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from techrag.ingest import EMBEDDING_MODEL, INDEX_DIR
from techrag.retrieve import Result, retrieve

LLM_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434"

SYSTEM_PROMPT = (
    "You answer questions about technical documentation using only the numbered context "
    "passages provided. Cite the passages you rely on like [1] or [2][3]. If the passages do "
    "not contain the answer, say that you cannot answer from the documentation instead of "
    "guessing."
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
    """
    context = "\n\n".join(
        f"[{i}] (source: {r.source})\n{r.text}" for i, r in enumerate(results, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {query}"},
    ]


def ollama_chat(
    messages: Sequence[dict[str, str]], model: str = LLM_MODEL, host: str = OLLAMA_URL
) -> str:
    """Send ``messages`` to the Ollama server at ``host`` and return the model's reply.

    Raises:
        OSError: If the server cannot be reached (including a connection refused because
            Ollama is not running) or the request fails.
        ValueError: If the server's response is not a chat reply, e.g. ``model`` is not
            pulled.
    """
    request_body = json.dumps({"model": model, "messages": list(messages), "stream": False})
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
) -> Answer:
    """Retrieve the ``k`` chunks most relevant to ``query`` and answer from them.

    ``chat`` takes the chat messages and returns the model's reply; it defaults to
    :func:`ollama_chat` with the default model and server.

    Raises:
        ValueError, FileNotFoundError: As for :func:`techrag.retrieve.retrieve`.
        OSError, ValueError: If ``chat`` fails (the default does on an unreachable server).
    """
    results = retrieve(query, k, index_dir, embedding_model)
    return Answer(text=chat(build_messages(query, results)).strip(), sources=results)


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
    args = parser.parse_args()

    def chat(messages: Sequence[dict[str, str]]) -> str:
        return ollama_chat(messages, args.llm, args.ollama_url)

    try:
        result = answer(args.query, args.k, chat, args.index_dir, args.embedding_model)
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")
    print(result.text)
    print("\nSources:")
    for i, r in enumerate(result.sources, start=1):
        print(f"[{i}] {r.source}  (score {r.score:.3f})")


if __name__ == "__main__":
    main()
