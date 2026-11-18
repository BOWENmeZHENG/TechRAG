"""An agentic search loop for the EnterpriseRAG-Bench questions.

Instead of retrieving once, the model decides what to look for: it calls ``search`` to find
passages and ``read_doc`` to read a whole document, as often as it needs within a step budget,
then answers (or says it cannot). Answers are graded and logged exactly like the single-shot
baseline's (see :mod:`techrag.baseline`), so the two are directly comparable.

The loop is independent of Ollama: it talks to the model through an :data:`AgentChat` function,
which :func:`ollama_agent_chat` provides.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from techrag.baseline import Record, grade, print_report, run_logged
from techrag.bench import SUBSET_DIR, BenchQuestion, doc_id_of, read_questions
from techrag.generate import OLLAMA_URL
from techrag.ingest import EMBEDDING_MODEL
from techrag.judge import JUDGE_MODEL, NUM_CTX, Chat, ollama_judge
from techrag.recall import INDEX_DIR
from techrag.retrieve import Mode, Result, Retriever

AGENT_MODEL = "qwen3.5:9b"
RESULTS_PATH = Path("eval/agent.jsonl")

MAX_STEPS = 8
# The prompt may grow to about this many characters (3 per token is cautious for mixed prose and
# identifiers) before the agent must answer: Ollama silently drops the start of a prompt that
# does not fit the context window, which would lose the question itself.
CONTEXT_CHARS = int(NUM_CTX * 0.75 * 3)
MAX_CALLS_PER_TURN = 4

SEARCH_DOCS = 8  # documents a search shows by default
SEARCH_DOCS_MAX = 15
CHUNKS_PER_DOC = 3  # chunks fetched per document shown, since several may come from one document
EXCERPT_CHARS = 700
PAGE_CHARS = 6000

SYSTEM_PROMPT = (
    "You answer questions about a company's internal documents: Slack threads, emails, Jira and "
    "Linear tickets, Confluence pages, GitHub issues and pull requests, Google Drive files, "
    "HubSpot records and meeting transcripts. You have two tools:\n"
    "- search(query): finds passages across all the documents. Each hit gives a document id, its "
    "title and an excerpt. Use specific terms from the question (names, services, error "
    "messages, ticket ids) and try different wording if a search misses.\n"
    "- read_doc(doc_id): reads a whole document, one page at a time. Excerpts are short and can "
    "mislead, so read a document before relying on it for details.\n\n"
    "Work like a careful researcher. Before each tool call, write one sentence on what you are "
    "looking for. Some questions need facts from several documents. If documents disagree, say "
    "so rather than picking one silently.\n\n"
    "When you have what you need, answer in plain prose without calling a tool. Give every "
    "specific the question asks for (names, numbers, versions, dates, steps) and nothing the "
    "documents do not support; do not pad the answer with unrelated detail. If, after searching "
    "thoroughly, the documents do not contain the answer, or only part of it, say plainly that "
    "you cannot fully answer from the available documentation, and state what is missing and "
    "what you did find."
)

FINALIZE_PROMPT = (
    "You can not search any more. Answer the question now using only what you have found. If it "
    "does not contain the answer, say that you cannot fully answer from the available "
    "documentation, and state what is missing and what you did find."
)

TOOL_SCHEMAS: tuple[dict, ...] = (
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search all documents for passages relevant to a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "what to look for"},
                    "k": {
                        "type": "integer",
                        "description": f"documents to show (default {SEARCH_DOCS}, "
                        f"at most {SEARCH_DOCS_MAX})",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_doc",
            "description": "Read a document by id, one page at a time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "id from a search result"},
                    "start": {
                        "type": "integer",
                        "description": "character offset to read from (default 0)",
                    },
                },
                "required": ["doc_id"],
            },
        },
    },
)


@dataclass(frozen=True)
class ToolCall:
    """A tool the model asked to call."""

    name: str
    arguments: dict


@dataclass(frozen=True)
class Reply:
    """One model turn: what it said, the tools it called, and the tokens it used."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tokens: int = 0


# Takes the conversation and the tool schemas (None to forbid tool calls) and returns the reply.
AgentChat = Callable[[Sequence[dict], Sequence[dict] | None], Reply]


class DocStore:
    """The full text of the documents of a subset, found by document id.

    Documents are the ``<source_type>/<doc_id>.md`` files under ``root``. Only the paths are
    indexed up front; text is read when asked for.
    """

    def __init__(self, root: Path) -> None:
        self._paths: dict[str, Path] = {}
        for path in sorted(root.glob("*/*.md")):
            self._paths.setdefault(doc_id_of(path), path)
        self._titles: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._paths)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._paths

    def source_type(self, doc_id: str) -> str:
        """Return the source type (``slack``, ``jira``, ...) of a document."""
        return self._paths[doc_id].parent.name

    def title(self, doc_id: str) -> str:
        """Return the title of a document: its first Markdown heading, or its id."""
        if doc_id not in self._titles:
            title = doc_id
            with self._paths[doc_id].open(encoding="utf-8") as f:
                first = f.readline().strip()
            if first.startswith("# "):
                title = first[2:].strip()
            self._titles[doc_id] = title
        return self._titles[doc_id]

    def read(self, doc_id: str, start: int = 0, size: int = PAGE_CHARS) -> str:
        """Return ``size`` characters of a document from offset ``start``, labelled.

        The page names the document and the span it covers, and says where to continue when the
        document goes on.

        Raises:
            KeyError: If there is no such document.
        """
        text = self._paths[doc_id].read_text(encoding="utf-8")
        start = max(0, min(start, len(text)))
        end = min(len(text), start + size)
        header = (
            f"[{doc_id}] {self.source_type(doc_id)} - {self.title(doc_id)} "
            f"(characters {start}-{end} of {len(text)})"
        )
        footer = f"\n\n[more: call read_doc with start={end}]" if end < len(text) else ""
        return f"{header}\n\n{text[start:end]}{footer}"


class Tools:
    """The agent's tools, over a chunk search and a :class:`DocStore`.

    ``search`` takes a query and a chunk count and returns chunks best first.
    """

    def __init__(self, search: Callable[[str, int], Sequence[Result]], docs: DocStore) -> None:
        self._search = search
        self._docs = docs

    def run(self, call: ToolCall) -> tuple[str, list[str]]:
        """Run ``call``; return its result text and the ids of the documents it showed.

        A call that cannot run (unknown tool, bad arguments, unknown document) returns an
        explanation as its text, so the model can correct itself.
        """
        args = call.arguments
        try:
            if call.name == "search":
                k = max(1, min(int(args.get("k", SEARCH_DOCS)), SEARCH_DOCS_MAX))
                return self._do_search(str(args["query"]), k)
            if call.name == "read_doc":
                return self._do_read(doc_id_of(str(args["doc_id"])), int(args.get("start", 0)))
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            return f"Bad arguments for {call.name}: {e!r}", []
        return f"Unknown tool {call.name!r}. Use search or read_doc.", []

    def _do_search(self, query: str, k: int) -> tuple[str, list[str]]:
        results = self._search(query, k * CHUNKS_PER_DOC)
        excerpts: dict[str, list[str]] = {}
        for r in results:
            excerpts.setdefault(doc_id_of(r.source), []).append(r.text.strip())
        shown = list(excerpts)[:k]
        if not shown:
            return f"No documents match {query!r}. Try different words.", []
        blocks = [
            f"[{d}] {self._docs.source_type(d)} - {self._docs.title(d)}\n"
            + "\n...\n".join(excerpts[d])[:EXCERPT_CHARS]
            for d in shown
            if d in self._docs
        ]
        return f"{len(blocks)} documents for {query!r}:\n\n" + "\n\n".join(blocks), shown

    def _do_read(self, doc_id: str, start: int) -> tuple[str, list[str]]:
        if doc_id not in self._docs:
            return f"No document with id {doc_id!r}. Use an id from a search result.", []
        return self._docs.read(doc_id, start), [doc_id]


@dataclass(frozen=True)
class Trajectory:
    """What an agent run produced and how it got there.

    ``stop`` is ``answered`` when the model chose to answer, otherwise why it was made to:
    ``max_steps`` (step budget used up), ``context`` (prompt grew too long) or ``empty`` (a reply
    with neither an answer nor a tool call). ``docs_seen`` lists the documents it was shown, in
    order of first appearance; ``trace`` has one entry per tool call.
    """

    answer: str
    steps: int
    tokens: int
    stop: str
    docs_seen: tuple[str, ...]
    trace: tuple[dict, ...]


def _size(messages: Sequence[dict]) -> int:
    return sum(len(m.get("content") or "") for m in messages)


def run_agent(
    question: str,
    chat: AgentChat,
    tools: Tools,
    max_steps: int = MAX_STEPS,
    context_chars: int = CONTEXT_CHARS,
) -> Trajectory:
    """Let the model research ``question`` with ``tools`` and return its answer.

    Each model turn is a step. The model answers by replying without a tool call. When it uses
    up ``max_steps`` turns, or the prompt passes ``context_chars`` characters, it is told to
    answer with what it has. A call identical to an earlier one is not run again.

    Raises:
        OSError, ValueError: If ``chat`` fails.
    """
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    seen_calls: set[str] = set()
    seen_docs: dict[str, None] = {}
    trace: list[dict] = []
    tokens = steps = 0
    stop = "max_steps"
    for step in range(1, max_steps + 1):
        if _size(messages) > context_chars:
            stop = "context"
            break
        reply = chat(messages, TOOL_SCHEMAS)
        steps, tokens = step, tokens + reply.tokens
        if not reply.tool_calls:
            if reply.content.strip():
                return Trajectory(
                    reply.content.strip(), steps, tokens, "answered", tuple(seen_docs), tuple(trace)
                )
            stop = "empty"
            break
        calls = reply.tool_calls[:MAX_CALLS_PER_TURN]
        messages.append(
            {
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [
                    {"function": {"name": c.name, "arguments": c.arguments}} for c in calls
                ],
            }
        )
        for call in calls:
            key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
            if key in seen_calls:
                text, docs = "You already made this exact call. Try a different query.", []
            else:
                seen_calls.add(key)
                text, docs = tools.run(call)
            seen_docs.update(dict.fromkeys(docs))
            messages.append({"role": "tool", "tool_name": call.name, "content": text})
            trace.append(
                {
                    "step": step,
                    "thought": reply.content.strip(),
                    "tool": call.name,
                    "args": call.arguments,
                    "docs": docs,
                    "chars": len(text),
                }
            )

    messages.append({"role": "user", "content": FINALIZE_PROMPT})
    reply = chat(messages, None)
    return Trajectory(
        reply.content.strip(), steps, tokens + reply.tokens, stop, tuple(seen_docs), tuple(trace)
    )


def parse_reply(payload: dict) -> Reply:
    """Parse an Ollama ``/api/chat`` response into a :class:`Reply`.

    Raises:
        ValueError: If ``payload`` is not a chat response or a tool call is malformed.
    """
    try:
        message = payload["message"]
        calls = []
        for c in message.get("tool_calls") or []:
            arguments = c["function"]["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise TypeError(f"arguments are not an object: {arguments!r}")
            calls.append(ToolCall(c["function"]["name"], arguments))
        tokens = int(payload.get("prompt_eval_count") or 0) + int(payload.get("eval_count") or 0)
        return Reply(message.get("content") or "", tuple(calls), tokens)
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as e:
        raise ValueError(f"Unexpected response from Ollama: {e!r}: {str(payload)[:300]}") from e


def ollama_agent_chat(
    model: str = AGENT_MODEL,
    host: str = OLLAMA_URL,
    think: bool = False,
    num_ctx: int = NUM_CTX,
    timeout: float = 600,
) -> AgentChat:
    """Return an :data:`AgentChat` that talks to the Ollama ``model`` at ``host``.

    Sampling is greedy for repeatability. Thinking is off by default: the agent already writes a
    line of reasoning before each tool call, and thinking makes every step several times slower.
    The context window matches the judge's so that Ollama keeps one model loaded for both.

    The returned function raises ``OSError`` if the server cannot be reached or rejects the
    request (e.g. a model that does not support tools), and ``ValueError`` for an unusable reply.
    """

    def chat(messages: Sequence[dict], tools: Sequence[dict] | None) -> Reply:
        body: dict = {
            "model": model,
            "messages": list(messages),
            "stream": False,
            "think": think,
            "options": {"temperature": 0.0, "num_ctx": num_ctx},
        }
        if tools:
            body["tools"] = list(tools)
        request = urllib.request.Request(
            f"{host.rstrip('/')}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return parse_reply(json.load(response))
        except urllib.error.HTTPError as e:
            raise OSError(f"Ollama returned {e.code}: {e.read().decode('utf-8', 'replace')}") from e

    return chat


def gold_share(question: BenchQuestion, doc_ids: Sequence[str]) -> float | None:
    """Return the share of ``question``'s gold documents among ``doc_ids``, None if it has none."""
    if not question.expected_doc_ids:
        return None
    seen = set(doc_ids)
    return sum(d in seen for d in question.expected_doc_ids) / len(question.expected_doc_ids)


def run_question(
    question: BenchQuestion,
    chat: AgentChat,
    make_tools: Callable[[], Tools],
    judge_chat: Chat,
    max_steps: int = MAX_STEPS,
) -> Record:
    """Run the agent on ``question`` and grade its answer.

    A request that times out, or a reply that cannot be parsed, becomes an ``error`` on the
    record so that one bad question does not end a long run.

    Raises:
        OSError: If a model server cannot be reached.
    """
    start = time.monotonic()
    try:
        traj = run_agent(question.text, chat, make_tools(), max_steps)
    except (TimeoutError, ValueError) as e:
        return Record(
            question.id,
            question.type,
            "",
            (),
            gold_share(question, ()),
            time.monotonic() - start,
            error=f"agent: {e}",
        )
    base = {
        "question_id": question.id,
        "question_type": question.type,
        "sources": traj.docs_seen,
        "gold_found": gold_share(question, traj.docs_seen),
    }
    return grade(
        question,
        base,
        traj.answer,
        time.monotonic() - start,
        judge_chat,
        steps=traj.steps,
        tokens=traj.tokens,
        stop=traj.stop,
        trace=traj.trace,
    )


def main() -> None:
    """Entry point for the ``techrag-agent`` command."""
    parser = argparse.ArgumentParser(description="Run the agentic search loop on the benchmark.")
    parser.add_argument("--subset", type=Path, default=SUBSET_DIR, help="subset to search")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL, help="must match ingest")
    parser.add_argument("--mode", choices=get_args(Mode), default="hybrid", help="ranking method")
    parser.add_argument("--llm", default=AGENT_MODEL, help="Ollama model that does the research")
    parser.add_argument("--think", action="store_true", help="let the model reason before replying")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="model turns per question")
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help="Ollama model to grade with")
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("-n", type=int, help="run on a seeded random sample of this many questions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ids", help="comma-separated question ids to run instead of all")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH, help="JSON Lines results")
    parser.add_argument("--fresh", action="store_true", help="discard earlier results first")
    args = parser.parse_args()

    try:
        questions = read_questions(args.subset / "questions.jsonl")
        if args.ids:
            wanted = set(args.ids.split(","))
            questions = [q for q in questions if q.id in wanted]
            if len(questions) != len(wanted):
                missing = wanted - {q.id for q in questions}
                raise ValueError(f"unknown question ids: {', '.join(sorted(missing))}")
        if args.n is not None:
            questions = random.Random(args.seed).sample(questions, min(args.n, len(questions)))
        docs = DocStore(args.subset)
        retriever = Retriever(args.index_dir, args.embedding_model)
        retriever.warm_up()

        def make_tools() -> Tools:
            return Tools(lambda q, k: retriever.search(q, k, args.mode), docs)

        chat = ollama_agent_chat(args.llm, args.ollama_url, args.think)
        judge_chat = ollama_judge(args.judge_model, args.ollama_url)
        records = run_logged(
            questions,
            lambda q: run_question(q, chat, make_tools, judge_chat, args.max_steps),
            args.output,
            args.fresh,
        )
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")
    print_report(records)


if __name__ == "__main__":
    main()
