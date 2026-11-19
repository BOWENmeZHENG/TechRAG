import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from techrag.agent import (
    FINALIZE_PROMPT,
    SYSTEM_PROMPT,
    DocStore,
    Reply,
    ToolCall,
    Tools,
    gold_share,
    parse_reply,
    run_agent,
    run_question,
)
from techrag.bench import BenchQuestion
from techrag.retrieve import Result


@pytest.fixture
def docs(tmp_path: Path) -> DocStore:
    (tmp_path / "jira").mkdir()
    (tmp_path / "slack").mkdir()
    (tmp_path / "jira" / "dsid_a.md").write_text("# Gateway outage\n\n" + "x" * 100)
    (tmp_path / "jira" / "dsid_b.md").write_text("# Cache bug\n\nbody b")
    (tmp_path / "slack" / "dsid_c.md").write_text("no heading here")
    (tmp_path / "slack" / "dsid_c~2.md").write_text("duplicate row")
    return DocStore(tmp_path)


def _hits(*pairs: tuple[str, str]):
    """A fake chunk search returning ``(doc_id, text)`` chunks, whatever the query."""
    results = [Result(Path(f"jira/{d}.md"), t, 1.0) for d, t in pairs]

    def search(query: str, k: int) -> list[Result]:
        return results[:k]

    return search


def _scripted(*replies: Reply):
    """A chat that plays ``replies`` in order, recording each call's messages and tools."""
    calls: list[tuple[list[dict], Sequence[dict] | None]] = []
    it = iter(replies)

    def chat(messages, tools):
        calls.append((list(messages), tools))
        return next(it)

    chat.calls = calls  # type: ignore[attr-defined]
    return chat


def _search(query: str = "gateway") -> ToolCall:
    return ToolCall("search", {"query": query})


def test_docstore_indexes_documents_by_id(docs: DocStore):
    assert len(docs) == 3  # the duplicate row of dsid_c is the same document
    assert "dsid_a" in docs and "dsid_zzz" not in docs
    assert docs.source_type("dsid_b") == "jira"
    assert docs.title("dsid_a") == "Gateway outage"
    assert docs.title("dsid_c") == "dsid_c"  # no heading: fall back to the id


def test_docstore_read_pages_and_says_where_to_continue(docs: DocStore):
    first = docs.read("dsid_a", 0, size=30)
    assert "characters 0-30 of 118" in first
    assert "[more: call read_doc with start=30]" in first
    last = docs.read("dsid_a", 100, size=30)
    assert "characters 100-118 of 118" in last and "[more" not in last
    assert "characters 118-118" in docs.read("dsid_a", 5000)  # past the end


def test_search_groups_chunks_by_document_and_limits_documents(docs: DocStore):
    search = _hits(("dsid_a", "first a"), ("dsid_b", "only b"), ("dsid_a", "second a"))
    text, shown = Tools(search, docs).run(ToolCall("search", {"query": "q", "k": 2}))
    assert shown == ["dsid_a", "dsid_b"]
    assert "[dsid_a] jira - Gateway outage\nfirst a\n...\nsecond a" in text
    assert "[dsid_b] jira - Cache bug\nonly b" in text


def test_search_truncates_long_excerpts_and_handles_no_hits(docs: DocStore):
    text, _ = Tools(_hits(("dsid_a", "y" * 5000)), docs).run(_search())
    assert len(text) < 1000
    text, shown = Tools(_hits(), docs).run(_search())
    assert shown == [] and "No documents match" in text


def test_search_clamps_k_and_survives_bad_arguments(docs: DocStore):
    seen = []

    def search(query: str, k: int):
        seen.append(k)
        return []

    tools = Tools(search, docs)
    tools.run(ToolCall("search", {"query": "q", "k": 1000}))
    assert seen[0] <= 15 * 3
    for bad in ({}, {"query": "q", "k": "many"}, {"query": None, "k": []}):
        text, shown = tools.run(ToolCall("search", bad))
        assert shown == [] and (text.startswith("Bad arguments") or "No documents" in text)


def test_read_doc_accepts_paths_and_reports_unknown_ids(docs: DocStore):
    tools = Tools(_hits(), docs)
    text, shown = tools.run(ToolCall("read_doc", {"doc_id": "jira/dsid_b.md"}))
    assert shown == ["dsid_b"] and "body b" in text
    text, shown = tools.run(ToolCall("read_doc", {"doc_id": "dsid_nope"}))
    assert shown == [] and "No document with id" in text
    text, _ = tools.run(ToolCall("delete_all", {}))
    assert text.startswith("Unknown tool")


def test_run_agent_searches_reads_then_answers(docs: DocStore):
    chat = _scripted(
        Reply("Looking for the outage.", (_search(),), tokens=10),
        Reply("Reading it.", (ToolCall("read_doc", {"doc_id": "dsid_a"}),), tokens=20),
        Reply(" The fix was X. ", tokens=5),
    )
    traj = run_agent("what fixed it?", chat, Tools(_hits(("dsid_a", "t"), ("dsid_b", "u")), docs))
    assert traj.answer == "The fix was X."
    assert traj.stop == "answered" and traj.steps == 3 and traj.tokens == 35
    assert traj.docs_seen == ("dsid_a", "dsid_b")
    assert [t["tool"] for t in traj.trace] == ["search", "read_doc"]
    assert traj.trace[0]["thought"] == "Looking for the outage."
    # the model sees its own calls and their results on the next turn
    messages, tools = chat.calls[2]
    assert tools is not None
    assert messages[-2]["role"] == "assistant" and messages[-2]["tool_calls"]
    assert messages[-1]["role"] == "tool" and "Gateway outage" in messages[-1]["content"]


def test_run_agent_forces_an_answer_when_steps_run_out(docs: DocStore):
    chat = _scripted(
        Reply("", (_search("a"),)), Reply("", (_search("b"),)), Reply("best guess", tokens=3)
    )
    traj = run_agent("q", chat, Tools(_hits(("dsid_a", "t")), docs), max_steps=2)
    assert traj.answer == "best guess" and traj.stop == "max_steps" and traj.steps == 2
    messages, tools = chat.calls[-1]
    assert tools is None  # no tools offered for the forced answer
    assert messages[-1] == {"role": "user", "content": FINALIZE_PROMPT}


def test_run_agent_forces_an_answer_when_the_prompt_grows_too_long(docs: DocStore):
    chat = _scripted(Reply("", (_search("a"),)), Reply("done"))
    traj = run_agent(
        "q", chat, Tools(_hits(("dsid_a", "t")), docs), context_chars=len(SYSTEM_PROMPT) + 30
    )
    assert traj.stop == "context" and traj.answer == "done" and traj.steps == 1


def test_run_agent_does_not_repeat_an_identical_call(docs: DocStore):
    searched = []

    def search(query: str, k: int):
        searched.append(query)
        return []

    chat = _scripted(Reply("", (_search(),)), Reply("", (_search(),)), Reply("ok"))
    traj = run_agent("q", chat, Tools(search, docs))
    assert searched == ["gateway"]
    assert "already made this exact call" in chat.calls[2][0][-1]["content"]
    assert traj.answer == "ok"


def test_run_agent_runs_at_most_a_few_calls_per_turn(docs: DocStore):
    chat = _scripted(Reply("", tuple(_search(f"q{i}") for i in range(10))), Reply("ok"))
    traj = run_agent("q", chat, Tools(_hits(), docs))
    assert len(traj.trace) == 4
    assert len(chat.calls[1][0][-5]["tool_calls"]) == 4


def test_run_agent_treats_an_empty_reply_as_a_reason_to_stop(docs: DocStore):
    traj = run_agent("q", _scripted(Reply("  "), Reply("late answer")), Tools(_hits(), docs))
    assert traj.stop == "empty" and traj.answer == "late answer"


def test_parse_reply():
    payload = {
        "message": {
            "content": "thinking out loud",
            "tool_calls": [
                {"function": {"name": "search", "arguments": {"query": "q"}}},
                {"function": {"name": "read_doc", "arguments": '{"doc_id": "d"}'}},
            ],
        },
        "prompt_eval_count": 100,
        "eval_count": 7,
    }
    reply = parse_reply(payload)
    assert reply.content == "thinking out loud" and reply.tokens == 107
    assert reply.tool_calls == (
        ToolCall("search", {"query": "q"}),
        ToolCall("read_doc", {"doc_id": "d"}),
    )
    assert parse_reply({"message": {"content": "hi"}}) == Reply("hi")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": {"tool_calls": [{"function": {"name": "x", "arguments": "not json"}}]}},
        {"message": {"tool_calls": [{"function": {"name": "x", "arguments": "[1]"}}]}},
        {"message": {"tool_calls": [{"nope": 1}]}},
    ],
)
def test_parse_reply_rejects_malformed_payloads(payload: dict):
    with pytest.raises(ValueError):
        parse_reply(payload)


def _question(expected: tuple[str, ...] = ("dsid_a",)) -> BenchQuestion:
    return BenchQuestion("q1", "basic", "what fixed it?", expected, "gold", ("fact",))


def _judge(met: bool = True):
    def chat(messages, schema):
        return json.dumps(
            {"facts": [{"evidence": "e", "met": met}], "contradiction": "", "reason": "r"}
        )

    return chat


def test_gold_share():
    assert gold_share(_question(("a", "b")), ["a", "x"]) == 0.5
    assert gold_share(_question(()), ["a"]) is None


def test_run_question_records_the_trajectory_and_grades(docs: DocStore):
    chat = _scripted(Reply("", (_search(),), tokens=4), Reply("the fix", tokens=6))
    record = run_question(
        _question(), chat, lambda: Tools(_hits(("dsid_a", "t"), ("dsid_b", "u")), docs), _judge()
    )
    assert record.error is None and record.score == 1.0
    assert record.answer == "the fix"
    assert (record.steps, record.tokens, record.stop) == (2, 10, "answered")
    assert record.sources == ("dsid_a", "dsid_b") and record.gold_found == 1.0
    assert record.trace[0]["tool"] == "search"


def test_run_question_records_timeouts_as_errors_and_lets_outages_abort(docs: DocStore):
    def slow(messages, tools):
        raise TimeoutError("timed out")

    record = run_question(_question(), slow, lambda: Tools(_hits(), docs), _judge())
    assert record.error == "agent: timed out" and record.verdict is None
    assert record.gold_found == 0.0

    def down(messages, tools):
        raise ConnectionRefusedError("refused")

    with pytest.raises(OSError):
        run_question(_question(), down, lambda: Tools(_hits(), docs), _judge())
