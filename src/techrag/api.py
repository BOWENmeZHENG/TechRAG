"""Serve TechRAG search and question answering over HTTP."""

import argparse
import sys
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints

from techrag.generate import LLM_MODEL, OLLAMA_URL, answer_from, ollama_chat
from techrag.ingest import EMBEDDING_MODEL, INDEX_DIR
from techrag.retrieve import Mode, Result, Retriever

Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SearchRequest(BaseModel):
    query: Query
    k: int = Field(5, ge=1, description="number of chunks to return or give the model")
    mode: Mode = "dense"


class Source(BaseModel):
    source: str = Field(description="path of the document, relative to the data directory")
    text: str
    score: float = Field(description="comparable only within one query and mode")

    @classmethod
    def from_result(cls, result: Result) -> "Source":
        return cls(source=result.source.as_posix(), text=result.text, score=result.score)


class SearchResponse(BaseModel):
    results: list[Source]


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    chunks: int


def create_app(
    retriever: Retriever,
    chat: Callable[[Sequence[dict[str, str]]], str] = ollama_chat,
) -> FastAPI:
    """Return an app that answers from ``retriever`` and asks the model behind ``chat``.

    ``retriever`` is loaded once by the caller and shared by every request, so the index and
    embedding model are not reloaded per call. ``chat`` takes the chat messages and returns the
    model's reply, as in :func:`techrag.generate.answer_from`.
    """
    app = FastAPI(title="TechRAG")
    app.state.retriever = retriever

    # These are plain ``def`` handlers so FastAPI runs them in a worker thread; embedding and
    # waiting on Ollama would otherwise block the event loop.

    @app.get("/healthz")
    def healthz(request: Request) -> HealthResponse:
        return HealthResponse(status="ok", chunks=len(request.app.state.retriever))

    @app.post("/search")
    def search(body: SearchRequest, request: Request) -> SearchResponse:
        results = request.app.state.retriever.search(body.query, body.k, body.mode)
        return SearchResponse(results=[Source.from_result(r) for r in results])

    @app.post("/ask")
    def ask(body: SearchRequest, request: Request) -> AskResponse:
        results = request.app.state.retriever.search(body.query, body.k, body.mode)
        try:
            result = answer_from(body.query, results, chat)
        except (OSError, ValueError) as e:  # Ollama unreachable, or an unexpected reply
            raise HTTPException(status_code=502, detail=f"Language model failed: {e}") from e
        return AskResponse(
            answer=result.text, sources=[Source.from_result(r) for r in result.sources]
        )

    return app


def main() -> None:
    """Entry point for the ``techrag-serve`` command."""
    parser = argparse.ArgumentParser(description="Serve the TechRAG API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument(
        "--embedding-model", default=EMBEDDING_MODEL, help="must match the ingest model"
    )
    parser.add_argument("--llm", default=LLM_MODEL, help="Ollama model to answer with")
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    args = parser.parse_args()

    try:
        retriever = Retriever(args.index_dir, args.embedding_model)
        retriever.warm_up()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")
    chat = partial(ollama_chat, model=args.llm, host=args.ollama_url)
    uvicorn.run(create_app(retriever, chat), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
