"""Grade answers to EnterpriseRAG-Bench questions with an LLM judge.

Every question carries ``answer_facts``, a checklist an answer must satisfy. For the
unanswerable question types the facts are rubrics ("must state that the query is not fully
answerable"), so one method grades every type: the judge decides, fact by fact, whether the
answer meets it.

* **completeness** is the fraction of the facts that state something which the answer meets;
* **correctness** is 1 unless the answer contradicts the reference answer or makes a claim that
  a fact forbids ("The answer must not ..."), in which case it is 0;
* the **score** is their product.

Forbidding facts feed correctness only. Counting them toward completeness would give an answer
that says nothing credit for every "must not", so a refusal would score well on questions that
have many of them.

Because a judge can be wrong in ways that quietly distort every later number, :func:`selftest`
probes it with answers whose right score is known (the reference answer, a refusal, half of
the facts, another question's answer) and measures how often it agrees.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from techrag.benchmarks.bench import BenchQuestion, read_questions
from techrag.common.config import JUDGE_MODEL, NUM_CTX, OLLAMA_URL, SUBSET_DIR
from techrag.core.generate import ollama_chat

SELFTEST_PATH = Path("eval/judge_selftest.json")
ABSTAIN = "I cannot answer this question from the available documentation."
INFO_NOT_FOUND = "info_not_found"

SYSTEM_PROMPT = (
    "You are a strict, literal grader of answers to questions about a company's internal "
    "documents. You are given the question, the reference answer, a numbered list of required "
    "facts, and a candidate answer. You judge only the candidate answer; you never answer the "
    "question yourself."
)

INSTRUCTIONS = (
    "Each required fact is labelled STATE or FORBIDDEN.\n"
    "- STATE: the candidate has to say it. `met` is true only if the candidate states the fact, "
    "in any wording. If the candidate leaves it out, or refuses to answer, `met` is false.\n"
    "- FORBIDDEN: the candidate must not say it. `met` is true if the candidate does not make "
    "the claim, and false only if the candidate asserts it.\n"
    "For each fact, in order, quote the part of the candidate answer that bears on it (at most "
    "15 words, empty if none), then give `met`.\n"
    "Then fill in `contradiction`: if the candidate states something that is inconsistent with "
    "the reference answer, quote that statement; otherwise leave it empty. A candidate that "
    "merely leaves information out, or adds detail that does not conflict with the reference, "
    "has no contradiction. Finish with a one-sentence `reason`."
)

# Takes the chat messages and the JSON schema the reply must follow; returns the raw reply.
Chat = Callable[[Sequence[dict[str, str]], dict], str]


def verdict_schema(n_facts: int) -> dict:
    """Return the JSON schema of a verdict on ``n_facts`` facts, with exactly one entry for each.

    Args:
        n_facts: Number of answer facts being graded.

    Returns:
        A JSON schema for Ollama's ``format`` parameter.
    """
    return {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "minItems": n_facts,
                "maxItems": n_facts,
                "items": {
                    "type": "object",
                    "properties": {"evidence": {"type": "string"}, "met": {"type": "boolean"}},
                    "required": ["evidence", "met"],
                },
            },
            "contradiction": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["facts", "contradiction", "reason"],
    }


@dataclass(frozen=True)
class Verdict:
    """The judge's grading of one answer.

    ``contradiction`` is the statement in the answer that conflicts with the reference answer,
    empty if there is none. ``prohibitions`` marks the facts that forbid a claim instead of
    stating one (see :func:`is_prohibition`); for those, ``facts_met`` is true unless the answer
    makes the forbidden claim. It defaults to no prohibitions.

    Attributes:
        facts_met: Per fact, whether the answer meets it.
        contradiction: Statement conflicting with the reference answer, or empty.
        reason: The judge's one-sentence explanation.
        prohibitions: Per fact, whether it forbids a claim.
    """

    facts_met: tuple[bool, ...]
    contradiction: str
    reason: str
    prohibitions: tuple[bool, ...] = ()

    @property
    def _statements_met(self) -> list[bool]:
        mask = self.prohibitions or (False,) * len(self.facts_met)
        return [m for m, forbids in zip(self.facts_met, mask, strict=True) if not forbids]

    @property
    def violated(self) -> bool:
        """Return whether the answer makes a claim that a fact forbids."""
        mask = self.prohibitions or (False,) * len(self.facts_met)
        return any(forbids and not m for m, forbids in zip(self.facts_met, mask, strict=True))

    @property
    def incorrect(self) -> bool:
        """Return whether the answer contradicts the reference answer or makes a forbidden claim."""
        return bool(self.contradiction.strip()) or self.violated

    @property
    def completeness(self) -> float:
        """Return the fraction of the facts that state something which the answer meets.

        A question whose facts all forbid claims has nothing to be complete about, so it is 1.
        """
        statements = self._statements_met
        return sum(statements) / len(statements) if statements else 1.0

    @property
    def correctness(self) -> float:
        """Return 0 if the answer contradicts the reference or makes a forbidden claim, else 1."""
        return 0.0 if self.incorrect else 1.0

    @property
    def score(self) -> float:
        """Return correctness times completeness."""
        return self.correctness * self.completeness


def build_messages(question: BenchQuestion, answer: str) -> list[dict[str, str]]:
    """Return the chat messages asking the judge to grade ``answer`` to ``question``.

    Args:
        question: The question, with its reference answer and answer facts.
        answer: The candidate answer to grade.

    Returns:
        A system message and a user message listing the facts as ``STATE`` or ``FORBIDDEN``.
    """
    facts = "\n".join(
        f"{i}. [{'FORBIDDEN' if is_prohibition(f) else 'STATE'}] {f}"
        for i, f in enumerate(question.answer_facts, start=1)
    )
    user = (
        f"Question: {question.text}\n\n"
        f"Reference answer: {question.gold_answer}\n\n"
        f"Required facts:\n{facts}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        f"{INSTRUCTIONS}\nReturn one entry in `facts` for each of the {len(question.answer_facts)} "
        "required facts."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def is_prohibition(fact: str) -> bool:
    """Return whether ``fact`` forbids a claim ("The answer must not ...") instead of making one.

    Args:
        fact: One of a question's answer facts.

    Returns:
        ``True`` if the fact starts with "The answer must not", ignoring case.
    """
    return fact.lower().startswith("the answer must not")


def parse_verdict(raw: str, prohibitions: Sequence[bool]) -> Verdict:
    """Parse the judge's JSON reply into a :class:`Verdict`.

    Args:
        raw: The judge's reply, expected to be JSON matching :func:`verdict_schema`.
        prohibitions: One entry per fact, saying whether that fact forbids a claim.

    Returns:
        The verdict, with ``prohibitions`` attached.

    Raises:
        ValueError: If ``raw`` is not JSON of the expected shape or has no entry for each fact.
    """
    try:
        data = json.loads(raw)
        facts = data["facts"]
        met = tuple(f["met"] for f in facts)
        contradiction = data["contradiction"]
        reason = data.get("reason", "")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"Judge reply is not a verdict: {e!r}: {raw[:200]!r}") from e
    if len(met) != len(prohibitions):
        raise ValueError(f"Judge graded {len(met)} facts, expected {len(prohibitions)}")
    if not all(isinstance(m, bool) for m in met) or not isinstance(contradiction, str):
        raise ValueError(f"Judge verdict has fields of the wrong type: {raw[:200]!r}")
    return Verdict(met, contradiction, str(reason), tuple(prohibitions))


def judge(
    question: BenchQuestion,
    answer: str,
    chat: Chat,
    retries: int = 1,
) -> Verdict:
    """Grade ``answer`` to ``question`` with the judge model behind ``chat``.

    ``chat`` takes the chat messages and :func:`verdict_schema` for the question, and returns the
    judge's raw reply. A blank answer meets nothing and is graded without calling it. An unusable
    reply is retried up to ``retries`` times.

    Args:
        question: The question, with its reference answer and answer facts.
        answer: The candidate answer to grade.
        chat: Function taking the chat messages and a schema and returning the raw reply.
        retries: Number of extra attempts after an unusable reply.

    Returns:
        The judge's verdict on ``answer``.

    Raises:
        ValueError: If ``question`` has no answer facts, or the judge never gives a usable verdict.
        OSError: If ``chat`` fails.
    """
    if not question.answer_facts:
        raise ValueError(f"Question {question.id} has no answer facts to grade against")
    prohibitions = tuple(is_prohibition(f) for f in question.answer_facts)
    if not answer.strip():  # says nothing, so it makes no forbidden claim
        return Verdict(prohibitions, "", "empty answer", prohibitions)
    messages = build_messages(question, answer)
    schema = verdict_schema(len(prohibitions))
    for attempt in range(retries + 1):
        try:
            return parse_verdict(chat(messages, schema), prohibitions)
        except ValueError:
            if attempt == retries:
                raise
    raise AssertionError("unreachable")


def ollama_judge(model: str = JUDGE_MODEL, host: str = OLLAMA_URL, think: bool = False) -> Chat:
    """Return a ``chat`` function that asks the Ollama ``model`` for a schema-conforming verdict.

    Sampling is greedy (temperature 0) so grading is repeatable. Reasoning is off by default,
    since the verdict already carries per-fact evidence and thinking is slow.

    Args:
        model: Name of the Ollama model to judge with.
        host: Base URL of the Ollama server.
        think: Whether a reasoning model should think before replying.

    Returns:
        A function taking the chat messages and a schema and returning the raw reply.
    """

    def chat(messages: Sequence[dict[str, str]], schema: dict) -> str:
        return ollama_chat(
            messages, model, host, format=schema, think=think, temperature=0.0, num_ctx=NUM_CTX
        )

    return chat


@dataclass(frozen=True)
class Probe:
    """An answer whose correct score is known, and what the judge gave it.

    Attributes:
        question_id: Id of the question the answer was built for.
        kind: How the probe was built: ``reference``, ``abstain``, ``partial`` or ``mismatch``.
        answer: The probe answer.
        expected: The score a correct judge would give.
        verdict: What the judge gave.
    """

    question_id: str
    kind: str
    answer: str
    expected: float
    verdict: Verdict

    @property
    def error(self) -> float:
        """Return how far the judge's score is from the expected one."""
        return abs(self.verdict.score - self.expected)


def make_probes(questions: Sequence[BenchQuestion]) -> list[tuple[BenchQuestion, str, str, float]]:
    """Return ``(question, kind, answer, expected score)`` probes built from ``questions``.

    * ``reference``: the reference answer, expected 1 (not for ``info_not_found``, whose
      reference answer is itself a rubric);
    * ``abstain``: a refusal, expected 1 for ``info_not_found`` questions and 0 otherwise;
    * ``partial``: the first half of the facts that state something (for questions with at least
      two), expected to score that share, since it states nothing forbidden;
    * ``mismatch``: the reference answer of the next answerable question, expected 0.

    Args:
        questions: Questions to build probes for.

    Returns:
        The probes, in question order.
    """
    answerable = [q for q in questions if q.type != INFO_NOT_FOUND]
    probes = []
    for i, q in enumerate(questions):
        unanswerable = q.type == INFO_NOT_FOUND
        probes.append((q, "abstain", ABSTAIN, 1.0 if unanswerable else 0.0))
        if not unanswerable:
            probes.append((q, "reference", q.gold_answer, 1.0))
            statements = [f for f in q.answer_facts if not is_prohibition(f)]
            if len(statements) >= 2:
                head = statements[: len(statements) // 2]
                probes.append((q, "partial", " ".join(head), len(head) / len(statements)))
        # The nearest answerable question after this one (wrapping round) other than itself.
        other = next(
            (
                o
                for j in range(1, len(questions))
                if (o := questions[(i + j) % len(questions)]) in answerable and o is not q
            ),
            None,
        )
        if other is not None:
            probes.append((q, "mismatch", other.gold_answer, 0.0))
    return probes


def selftest(
    questions: Sequence[BenchQuestion],
    chat: Chat,
    n: int = 20,
    seed: int = 0,
) -> tuple[list[Probe], list[str]]:
    """Grade the probes built from ``n`` questions sampled (seeded) from those with facts.

    Returns the graded probes and a description of each probe the judge never gave a usable
    verdict for, so one bad reply does not lose the rest of a long run.

    Args:
        questions: Questions to sample from; those without answer facts are ignored.
        chat: Function taking the chat messages and a schema and returning the raw reply.
        n: Number of questions to build probes from.
        seed: Seed for the sample.

    Returns:
        The graded probes and a description of each probe that failed.

    Raises:
        OSError: If ``chat`` fails.
    """
    pool = [q for q in questions if q.answer_facts]
    sample = random.Random(seed).sample(pool, min(n, len(pool)))
    probes, failures = [], []
    for q, kind, answer, expected in make_probes(sample):
        try:
            probes.append(Probe(q.id, kind, answer, expected, judge(q, answer, chat)))
        except ValueError as e:
            failures.append(f"{q.id} {kind}: {e}")
    return probes, failures


def summarize_probes(probes: Sequence[Probe]) -> dict[str, dict[str, float]]:
    """Return, per probe kind, the count, mean expected and observed score, and mean error.

    Args:
        probes: Graded probes from :func:`selftest`.

    Returns:
        A map from probe kind to ``n``, ``expected``, ``observed``, ``mean_abs_error`` and
        ``within_0.25`` (the share of probes within 0.25 of the expected score).
    """
    groups: dict[str, list[Probe]] = defaultdict(list)
    for p in probes:
        groups[p.kind].append(p)
    return {
        kind: {
            "n": len(ps),
            "expected": sum(p.expected for p in ps) / len(ps),
            "observed": sum(p.verdict.score for p in ps) / len(ps),
            "mean_abs_error": sum(p.error for p in ps) / len(ps),
            "within_0.25": sum(p.error <= 0.25 for p in ps) / len(ps),
        }
        for kind, ps in sorted(groups.items())
    }


def main() -> None:
    """Entry point for the ``techrag-judge`` command."""
    parser = argparse.ArgumentParser(description="Check the answer judge against known answers.")
    parser.add_argument("--subset", type=Path, default=SUBSET_DIR, help="subset with questions")
    parser.add_argument("-n", type=int, default=20, help="questions to build probes from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=JUDGE_MODEL, help="Ollama model to judge with")
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--think", action="store_true", help="let the judge model reason first")
    parser.add_argument("--output", type=Path, default=SELFTEST_PATH, help="JSON report to write")
    args = parser.parse_args()

    try:
        questions = read_questions(args.subset / "questions.jsonl")
        chat = ollama_judge(args.model, args.ollama_url, args.think)
        probes, failures = selftest(questions, chat, args.n, args.seed)
        summary = summarize_probes(probes)
        report = {
            "model": args.model,
            "think": args.think,
            "seed": args.seed,
            "summary": summary,
            "failures": failures,
            "probes": [
                {
                    "question_id": p.question_id,
                    "kind": p.kind,
                    "expected": p.expected,
                    "score": p.verdict.score,
                    "facts_met": list(p.verdict.facts_met),
                    "contradiction": p.verdict.contradiction,
                    "reason": p.verdict.reason,
                    "answer": p.answer,
                }
                for p in probes
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")

    print(
        f"{'probe':<12}{'n':>4}{'expected':>10}{'observed':>10}{'mean|err|':>11}{'within .25':>12}"
    )
    for kind, s in summary.items():
        print(
            f"{kind:<12}{s['n']:>4}{s['expected']:>10.2f}{s['observed']:>10.2f}"
            f"{s['mean_abs_error']:>11.2f}{s['within_0.25']:>12.2f}"
        )
    if failures:
        print(f"\n{len(failures)} probes got no usable verdict and were left out:")
        for failure in failures:
            print(f"  {failure}")
    bad = [p for p in probes if p.error > 0.25]
    print(f"\n{len(bad)} of {len(probes)} graded probes off by more than 0.25")
    for p in bad:
        print(
            f"  {p.question_id} {p.kind}: expected {p.expected:.2f}, got {p.verdict.score:.2f}"
            f" ({p.verdict.reason})"
        )


if __name__ == "__main__":
    main()
