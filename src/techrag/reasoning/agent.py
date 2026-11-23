"""An agentic search loop for the EnterpriseRAG-Bench questions.

Instead of retrieving once, the model decides what to look for: it calls ``search`` to find
passages and ``read_doc`` to read a whole document, as often as it needs within a step budget,
then answers (or says it cannot). Answers are graded and logged exactly like the single-shot
baseline's (see :mod:`techrag.reasoning.harness`), so the two are directly comparable.

The loop is independent of Ollama: it talks to the model through an :data:`AgentChat` function,
which :func:`ollama_agent_chat` provides.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from techrag.benchmarks.bench import BenchQuestion, doc_id_of, read_questions, sample_questions
from techrag.common.config import (
    EMBEDDING_MODEL,
    INDEX_DIR,
    JUDGE_MODEL,
    MAX_STEPS,
    NUM_CTX,
    OLLAMA_URL,
    SUBSET_DIR,
    AgentConfig,
    EvaluationConfig,
    LLMConfig,
    RetrievalConfig,
    RunConfig,
)
from techrag.common.log import setup_logging
from techrag.core.retrieve import Mode, Result, Retriever
from techrag.eval.judge import Chat, ollama_judge
from techrag.reasoning.harness import (
    Record,
    data_config,
    grade,
    print_report,
    run_logged,
    run_metadata,
    save_manifest,
)

AGENT_MODEL = "qwen3.5:9b"
RESULTS_PATH = Path("eval/agent.jsonl")

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
    """A tool the model asked to call.

    Attributes:
        name: Name of the tool, e.g. ``search``.
        arguments: The arguments the model passed.
    """

    name: str
    arguments: dict


@dataclass(frozen=True)
class Reply:
    """One model turn: what it said, the tools it called, and the tokens it used.

    Attributes:
        content: The text of the reply.
        tool_calls: The tools the model asked to call, if any.
        tokens: Prompt and reply tokens processed for this turn.
    """

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
        """Index the documents under ``root`` by id.

        Args:
            root: Subset directory holding ``<source_type>/<doc_id>.md`` files.
        """
        self._paths: dict[str, Path] = {}
        for path in sorted(root.glob("*/*.md")):
            self._paths.setdefault(doc_id_of(path), path)
        self._titles: dict[str, str] = {}

    def __len__(self) -> int:
        """Return the number of documents."""
        return len(self._paths)

    def __contains__(self, doc_id: str) -> bool:
        """Return whether there is a document with id ``doc_id``."""
        return doc_id in self._paths

    def source_type(self, doc_id: str) -> str:
        """Return the source type (``slack``, ``jira``, ...) of a document.

        Args:
            doc_id: Id of a document in the store.

        Returns:
            The name of the directory the document is in.

        Raises:
            KeyError: If there is no such document.
        """
        return self._paths[doc_id].parent.name

    def title(self, doc_id: str) -> str:
        """Return the title of a document: its first Markdown heading, or its id.

        Args:
            doc_id: Id of a document in the store.

        Returns:
            The title, cached after the first call.

        Raises:
            KeyError: If there is no such document.
        """
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

        Args:
            doc_id: Id of a document in the store.
            start: Character offset to read from; clamped to the document.
            size: Number of characters to read.

        Returns:
            The page, with a header naming the document and span and, if the document
            continues, a footer giving the next ``start``.

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
        """Create the tools.

        Args:
            search: Function taking a query and a chunk count and returning chunks, best first.
            docs: Store the ``read_doc`` tool reads from.
        """
        self._search = search
        self._docs = docs

    def run(self, call: ToolCall) -> tuple[str, list[str]]:
        """Run ``call``; return its result text and the ids of the documents it showed.

        A call that cannot run (unknown tool, bad arguments, unknown document) returns an
        explanation as its text, so the model can correct itself.

        Args:
            call: The tool call requested by the model.

        Returns:
            The text to show the model, and the ids of the documents it showed.
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
        """Search for ``query`` and return excerpts of up to ``k`` documents and their ids."""
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
        """Return a page of ``doc_id`` from offset ``start`` and its id, or an explanation."""
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

    Attributes:
        answer: The final answer.
        steps: Model turns taken, not counting the forced final answer.
        tokens: Tokens processed, prompt and reply summed over turns.
        stop: Why the run ended.
        docs_seen: Ids of the documents shown to the model, in order of first appearance.
        trace: One entry per tool call.
    """

    answer: str
    steps: int
    tokens: int
    stop: str
    docs_seen: tuple[str, ...]
    trace: tuple[dict, ...]


def _size(messages: Sequence[dict]) -> int:
    """Return the total characters of the ``content`` of ``messages``."""
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

    Args:
        question: The question to research.
        chat: Function that sends the conversation and tool schemas to the model.
        tools: The tools the model may call.
        max_steps: Maximum number of model turns before it must answer.
        context_chars: Prompt size, in characters, past which the model must answer.

    Returns:
        The answer and how the run went.

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

    Args:
        payload: The decoded JSON response.

    Returns:
        The reply's text, tool calls and token count.

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

    Args:
        model: Name of the Ollama model.
        host: Base URL of the Ollama server.
        think: Whether the model should think before replying.
        num_ctx: Context window in tokens.
        timeout: Seconds to wait for each reply.

    Returns:
        A function taking the conversation and tool schemas (``None`` to forbid tool calls)
        and returning the model's :class:`Reply`.
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
    """Return the share of ``question``'s gold documents among ``doc_ids``, None if it has none.

    Args:
        question: The question whose gold documents to look for.
        doc_ids: Ids of the documents the agent was shown.

    Returns:
        A value in ``[0, 1]``, or ``None`` if the question has no gold documents.
    """
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

    Args:
        question: The question to answer.
        chat: Function that sends the conversation and tool schemas to the model.
        make_tools: Factory for the tools, called once per question.
        judge_chat: Function that sends the judge its messages and schema and returns its reply.
        max_steps: Maximum number of model turns before the agent must answer.

    Returns:
        The graded record of the question.

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
    parser.add_argument("--per-type", type=int, help="run on this many questions of every type")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ids", help="comma-separated question ids to run instead of all")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH, help="JSON Lines results")
    parser.add_argument("--run-name", help="name for the run manifest (default: results file name)")
    parser.add_argument("--fresh", action="store_true", help="discard earlier results first")
    args = parser.parse_args()
    setup_logging()

    try:
        config = RunConfig(
            metadata=run_metadata(args.run_name or args.output.stem),
            llm=LLMConfig(model_name=args.llm, num_ctx=NUM_CTX),
            retrieval=RetrievalConfig(
                mode=args.mode, k=SEARCH_DOCS, embedding_model=args.embedding_model
            ),
            agent=AgentConfig(max_steps=args.max_steps, think=args.think),
            evaluation=EvaluationConfig(judge_model=args.judge_model),
            data=data_config(args.subset),
        )
        save_manifest(config, args.output)
        questions = read_questions(args.subset / "questions.jsonl")
        if args.ids:
            wanted = set(args.ids.split(","))
            questions = [q for q in questions if q.id in wanted]
            if len(questions) != len(wanted):
                missing = wanted - {q.id for q in questions}
                raise ValueError(f"unknown question ids: {', '.join(sorted(missing))}")
        questions = sample_questions(questions, args.n, args.per_type, args.seed)
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
