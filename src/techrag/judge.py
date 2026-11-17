"""Grade answers to EnterpriseRAG-Bench questions with an LLM judge.

Every question carries ``answer_facts``, a checklist an answer must satisfy. For the
unanswerable question types the facts are rubrics ("must state that the query is not fully
answerable"), so one method grades every type: the judge decides, fact by fact, whether the
answer meets it.

* **completeness** is the fraction of facts met;
* **correctness** is 1 unless the answer asserts something that contradicts the reference
  answer, in which case it is 0;
* the **score** is their product.

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

from techrag.bench import SUBSET_DIR, BenchQuestion, read_questions
from techrag.generate import OLLAMA_URL, ollama_chat

JUDGE_MODEL = "qwen3.5:9b"
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
    "For each required fact, in order, quote the part of the candidate answer that bears on it "
    "(empty if none) and decide whether the fact is met. A fact is met only if the candidate "
    "states it, in any wording, and does not contradict it. Facts phrased as requirements "
    '("must state ...", "must not claim ...") are met only if the candidate satisfies them.\n'
    "Then fill in `contradiction`: if the candidate states something that is inconsistent with "
    "the reference answer, quote that statement; otherwise leave it empty. A candidate that "
    "merely leaves information out, or adds detail that does not conflict with the reference, "
    "has no contradiction. Finish with a one-sentence `reason`."
)

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
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
    empty if there is none.
    """

    facts_met: tuple[bool, ...]
    contradiction: str
    reason: str

    @property
    def incorrect(self) -> bool:
        """Return whether the answer contradicts the reference answer."""
        return bool(self.contradiction.strip())

    @property
    def completeness(self) -> float:
        """Return the fraction of the required facts the answer meets."""
        return sum(self.facts_met) / len(self.facts_met)

    @property
    def correctness(self) -> float:
        """Return 0 if the answer contradicts the reference answer, else 1."""
        return 0.0 if self.incorrect else 1.0

    @property
    def score(self) -> float:
        """Return correctness times completeness."""
        return self.correctness * self.completeness


def build_messages(question: BenchQuestion, answer: str) -> list[dict[str, str]]:
    """Return the chat messages asking the judge to grade ``answer`` to ``question``."""
    facts = "\n".join(f"{i}. {f}" for i, f in enumerate(question.answer_facts, start=1))
    user = (
        f"Question: {question.text}\n\n"
        f"Reference answer: {question.gold_answer}\n\n"
        f"Required facts:\n{facts}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        f"{INSTRUCTIONS}\nReturn one entry in `facts` for each of the {len(question.answer_facts)} "
        "required facts."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_verdict(raw: str, n_facts: int) -> Verdict:
    """Parse the judge's JSON reply into a :class:`Verdict`.

    Raises:
        ValueError: If ``raw`` is not JSON of the expected shape or has no entry for each of the
            ``n_facts`` facts.
    """
    try:
        data = json.loads(raw)
        facts = data["facts"]
        met = tuple(f["met"] for f in facts)
        contradiction = data["contradiction"]
        reason = data.get("reason", "")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"Judge reply is not a verdict: {e!r}: {raw[:200]!r}") from e
    if len(met) != n_facts:
        raise ValueError(f"Judge graded {len(met)} facts, expected {n_facts}")
    if not all(isinstance(m, bool) for m in met) or not isinstance(contradiction, str):
        raise ValueError(f"Judge verdict has fields of the wrong type: {raw[:200]!r}")
    return Verdict(met, contradiction, str(reason))


def judge(
    question: BenchQuestion,
    answer: str,
    chat: Callable[[Sequence[dict[str, str]]], str],
    retries: int = 1,
) -> Verdict:
    """Grade ``answer`` to ``question`` with the judge model behind ``chat``.

    ``chat`` takes the chat messages and returns the judge's raw reply, which must follow
    :data:`VERDICT_SCHEMA`. A blank answer meets nothing and is graded without calling it. An
    unusable reply is retried up to ``retries`` times.

    Raises:
        ValueError: If ``question`` has no answer facts, or the judge never gives a usable verdict.
        OSError: If ``chat`` fails.
    """
    n = len(question.answer_facts)
    if n == 0:
        raise ValueError(f"Question {question.id} has no answer facts to grade against")
    if not answer.strip():
        return Verdict((False,) * n, "", "empty answer")
    messages = build_messages(question, answer)
    for attempt in range(retries + 1):
        try:
            return parse_verdict(chat(messages), n)
        except ValueError:
            if attempt == retries:
                raise
    raise AssertionError("unreachable")


def ollama_judge(
    model: str = JUDGE_MODEL, host: str = OLLAMA_URL, think: bool = False
) -> Callable[[Sequence[dict[str, str]]], str]:
    """Return a ``chat`` function that asks the Ollama ``model`` for a schema-conforming verdict.

    Sampling is greedy (temperature 0) so grading is repeatable. Reasoning is off by default,
    since the verdict already carries per-fact evidence and thinking is slow.
    """

    def chat(messages: Sequence[dict[str, str]]) -> str:
        return ollama_chat(
            messages, model, host, format=VERDICT_SCHEMA, think=think, temperature=0.0
        )

    return chat


@dataclass(frozen=True)
class Probe:
    """An answer whose correct score is known, and what the judge gave it."""

    question_id: str
    kind: str
    answer: str
    expected: float
    verdict: Verdict

    @property
    def error(self) -> float:
        """Return how far the judge's score is from the expected one."""
        return abs(self.verdict.score - self.expected)


def _is_prohibition(fact: str) -> bool:
    """Return whether ``fact`` forbids a claim ("The answer must not ...") instead of making one."""
    return fact.lower().startswith("the answer must not")


def make_probes(questions: Sequence[BenchQuestion]) -> list[tuple[BenchQuestion, str, str, float]]:
    """Return ``(question, kind, answer, expected score)`` probes built from ``questions``.

    Some facts forbid a claim ("The answer must not ..."), and any answer that does not make that
    claim meets them, so every probe except ``reference`` is expected to meet the prohibitions
    and nothing more.

    * ``reference``: the reference answer, expected 1 (not for ``info_not_found``, whose
      reference answer is itself a rubric);
    * ``abstain``: a refusal, expected 1 for ``info_not_found`` questions;
    * ``partial``: the first half of the facts that state something (for questions with at least
      two), expected to meet exactly those;
    * ``mismatch``: the reference answer of the next answerable question.
    """
    answerable = [q for q in questions if q.type != INFO_NOT_FOUND]
    probes = []
    for i, q in enumerate(questions):
        unanswerable = q.type == INFO_NOT_FOUND
        n = len(q.answer_facts)
        prohibitions = sum(_is_prohibition(f) for f in q.answer_facts)
        floor = prohibitions / n
        probes.append((q, "abstain", ABSTAIN, 1.0 if unanswerable else floor))
        if not unanswerable:
            probes.append((q, "reference", q.gold_answer, 1.0))
            statements = [f for f in q.answer_facts if not _is_prohibition(f)]
            if len(statements) >= 2:
                head = statements[: len(statements) // 2]
                probes.append((q, "partial", " ".join(head), (len(head) + prohibitions) / n))
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
            probes.append((q, "mismatch", other.gold_answer, 0.0 if unanswerable else floor))
    return probes


def selftest(
    questions: Sequence[BenchQuestion],
    chat: Callable[[Sequence[dict[str, str]]], str],
    n: int = 20,
    seed: int = 0,
) -> list[Probe]:
    """Grade the probes built from ``n`` questions sampled (seeded) from those with facts."""
    pool = [q for q in questions if q.answer_facts]
    sample = random.Random(seed).sample(pool, min(n, len(pool)))
    return [
        Probe(q.id, kind, answer, expected, judge(q, answer, chat))
        for q, kind, answer, expected in make_probes(sample)
    ]


def summarize_probes(probes: Sequence[Probe]) -> dict[str, dict[str, float]]:
    """Return, per probe kind, the count, mean expected and observed score, and mean error."""
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
        probes = selftest(questions, chat, args.n, args.seed)
        summary = summarize_probes(probes)
        report = {
            "model": args.model,
            "think": args.think,
            "seed": args.seed,
            "summary": summary,
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
    bad = [p for p in probes if p.error > 0.25]
    print(f"\n{len(bad)} of {len(probes)} probes off by more than 0.25")
    for p in bad:
        print(
            f"  {p.question_id} {p.kind}: expected {p.expected:.2f}, got {p.verdict.score:.2f}"
            f" ({p.verdict.reason})"
        )


if __name__ == "__main__":
    main()
