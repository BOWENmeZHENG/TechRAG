from pathlib import Path

import pytest

from techrag.core import generate
from techrag.core.generate import (
    SYSTEM_PROMPT,
    Answer,
    answer,
    answer_from,
    build_messages,
    main,
)
from techrag.core.retrieve import Result

RESULTS = [
    Result(Path("a.md"), "alpha text", 0.9),
    Result(Path("sub/b.md"), "beta text", 0.5),
]


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


def test_answer_chats(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:

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


def test_answer_from_uses_given_results_without_retrieving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_retrieve(*args):
        raise AssertionError("answer_from must not retrieve")

    monkeypatch.setattr(generate, "retrieve", fail_retrieve)
    sent = []

    def chat(messages):
        sent.append(messages)
        return "  per [1], beta.\n"

    result = answer_from("what is beta?", tuple(RESULTS), chat)

    assert result == Answer(text="per [1], beta.", sources=RESULTS)
    assert isinstance(result.sources, list)  # a tuple input is copied into a list
    assert sent == [build_messages("what is beta?", RESULTS)]


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
def test_main_exits_with_error_message(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def fake_answer(*args):
        raise error

    monkeypatch.setattr(generate, "answer", fake_answer)
    monkeypatch.setattr("sys.argv", ["techrag-ask", "q"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == f"error: {error}"
