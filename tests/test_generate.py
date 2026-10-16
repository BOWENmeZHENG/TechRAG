import io
import json
import urllib.request
from pathlib import Path

import pytest

from techrag import generate
from techrag.generate import SYSTEM_PROMPT, Answer, answer, build_messages, main, ollama_chat
from techrag.retrieve import Result

RESULTS = [
    Result(Path("a.md"), "alpha text", 0.9),
    Result(Path("sub/b.md"), "beta text", 0.5),
]


def json_response(body: object) -> io.BytesIO:
    return io.BytesIO(json.dumps(body).encode("utf-8"))


def test_build_messages() -> None:
    messages = build_messages("what is beta?", RESULTS)

    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1] == {
        "role": "user",
        "content": (
            "Context:\n\n"
            "[1] (source: a.md)\nalpha text\n\n"
            "[2] (source: sub/b.md)\nbeta text\n\n"
            "Question: what is beta?"
        ),
    }


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


def test_answer_chats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    calls = []

    def fake_retrieve(query, k, index_dir, embedding_model, mode):
        calls.append((query, k, index_dir, embedding_model, mode))
        return RESULTS

    def chat(messages):
        return "  per [1], beta.\n"

    monkeypatch.setattr(generate, "retrieve", fake_retrieve)

    result = answer(
        "what is beta?", k=2, chat=chat, index_dir=tmp_path, embedding_model="m", mode="bm25"
    )

    assert result == Answer(text="per [1], beta.", sources=RESULTS)
    assert calls == [("what is beta?", 2, tmp_path, "m", "bm25")]


def test_main_prints_answer_and_numbered_sources(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        generate, "answer", lambda *args: Answer(text="the answer [1]", sources=RESULTS)
    )
    monkeypatch.setattr("sys.argv", ["techrag-ask", "what is beta?"])

    main()

    assert capsys.readouterr().out == (
        "the answer [1]\n\nSources:\n[1] a.md  (score 0.900)\n[2] sub/b.md  (score 0.500)\n"
    )


@pytest.mark.parametrize("error", [OSError("connection refused"), ValueError("bad index")])
def test_main_exits_with_error_message(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fake_answer(*args):
        raise error

    monkeypatch.setattr(generate, "answer", fake_answer)
    monkeypatch.setattr("sys.argv", ["techrag-ask", "q"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == f"error: {error}"
