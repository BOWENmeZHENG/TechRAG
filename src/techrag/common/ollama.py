"""Minimal client for a local Ollama server."""

import json
import urllib.request
from collections.abc import Sequence

from techrag.common.config import LLM_MODEL, OLLAMA_URL


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
