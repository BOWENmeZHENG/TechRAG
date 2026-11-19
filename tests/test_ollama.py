import io
import json
import urllib.request

import pytest

from techrag.common.ollama import ollama_chat


def json_response(body: object) -> io.BytesIO:
    return io.BytesIO(json.dumps(body).encode("utf-8"))


def test_ollama_chat_request_and_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    def urlopen(request, timeout=None):
        requests.append(request)
        return json_response({"message": {"role": "assistant", "content": "hello"}})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    messages = [{"role": "user", "content": "hi"}]

    reply = ollama_chat(messages, model="some-model", host="http://example:1234/")

    assert reply == "hello"
    (request,) = requests
    assert request.full_url == "http://example:1234/api/chat"  # trailing slash is dropped
    assert json.loads(request.data) == {
        "model": "some-model",
        "messages": messages,
        "stream": False,
    }


def test_ollama_chat_passes_format_think_and_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    def urlopen(request, timeout=None):
        requests.append(request)
        return json_response({"message": {"role": "assistant", "content": "{}"}})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    schema = {"type": "object"}

    ollama_chat(
        [{"role": "user", "content": "hi"}],
        "m",
        format=schema,
        think=False,
        temperature=0.0,
        num_ctx=8192,
    )

    body = json.loads(requests[0].data)
    assert body["format"] == schema
    assert body["think"] is False  # False must be sent, not treated as unset
    assert body["options"] == {"temperature": 0.0, "num_ctx": 8192}


def test_ollama_chat_unexpected_response_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout=None: json_response({"error": "model 'x' not found"}),
    )

    with pytest.raises(ValueError, match="Unexpected response from Ollama"):
        ollama_chat([{"role": "user", "content": "hi"}])
