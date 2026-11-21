# EnterpriseRAG-Bench: data and questions

TechRAG is evaluated on [EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench),
a synthetic benchmark of internal company data. This page explains what the data looks like, what
the questions are, and how the repo uses them. The dataset lives in
`data/external/EnterpriseRAG-Bench/`; its own README is the original source for most of this.

## The corpus

The data simulates one fictional company, **Redwood Inference**, which sells AI model inference as
a service. Every document shares the same company, people, projects and jargon, so facts are
spread across documents the way they are in a real company.

The full corpus has about **512k documents** from nine sources:

| Source | Documents | Contents |
|---|---:|---|
| Slack | 285,605 | Channel and team discussions |
| Gmail | 121,390 | Email threads (management, sales, leadership, ICs) |
| Linear | 35,308 | Eng / product / design tickets |
| Google Drive | 25,108 | Shared and collaborative documents |
| HubSpot | 15,017 | Sales CRM records |
| Fireflies | 10,173 | Meeting transcripts |
| GitHub | 8,052 | Pull requests and comments |
| Jira | 6,120 | Internal and customer-facing support tickets |
| Confluence | 5,189 | Wikis, runbooks, structured docs |

Each document has four fields: `doc_id` (`dsid_...`), `source_type`, `title` and `content`.

The corpus is deliberately noisy. It includes miscellaneous off-topic files, shuffled documents,
and **near-duplicates that carry updated or conflicting facts**. It uses internal project
codenames and acronyms, so retrieval has to learn the company's vocabulary.

## The questions

There are **500 questions** in a single `test` split. Each has:

| Field | Meaning |
|---|---|
| `question_id` | Unique id, e.g. `qst_0001` |
| `question_type` | One of the 10 categories below |
| `source_types` | Which sources contain the answer |
| `question` | The question text |
| `expected_doc_ids` | Gold documents that support the answer |
| `gold_answer` | Reference answer |
| `answer_facts` | Checklist of facts a correct answer must contain (about 4.9 per question on average) |

In code these are `BenchQuestion` objects ([bench.py](../src/techrag/benchmarks/bench.py)), whose
`type` is the `question_type` column.

### Question types

The types differ in what makes the question hard.

#### Single-document lookups

| Type | Count | Gold docs | What it tests |
|---|---:|---:|---|
| `basic` | 175 | 1 | A direct question with one gold document. |
| `semantic` | 125 | 1 | Same, but phrased with little keyword overlap ("the new top end 80GB accelerator" means the H200). Keyword search struggles. |
| `intra_document_reasoning` | 40 | 1 | The answer combines two distant sections of one document, so chunking matters. |
| `miscellaneous` | 20 | 1 | The answer is in informal or off-topic material, such as a joke thread in team chat. |

#### Multi-document

| Type | Count | Gold docs | What it tests |
|---|---:|---:|---|
| `project_related` | 40 | 2–9 (avg 4.2) | Aggregating knowledge from several documents of one project. |
| `constrained` | 30 | 1–2 (avg 1.4) | Many documents look relevant, but a qualifier (date, region, customer) narrows it to the right one. |
| `conflicting_info` | 20 | 2 | Documents contradict each other, for example an old and an updated value. The answer must resolve the conflict and mention both. |
| `completeness` | 20 | 2–10 (avg 6.5) | Needs every relevant document. Missing one makes the answer incomplete. This is the recall stress test. |

#### No gold document

| Type | Count | Gold docs | What it tests |
|---|---:|---:|---|
| `high_level` | 10 | 0 | Broad questions with no single source, such as "What is the company's mission statement?" |
| `info_not_found` | 20 | 0 | The corpus does not contain the answer. A correct response says it cannot fully answer. A confident answer is wrong. |

The set is skewed: `basic` and `semantic` are 300 of the 500 questions, and `high_level` is only
10. Because of that, `sample_questions(per_type=N)` samples each type separately instead of
drawing at random, so every type is represented in a small run.

### Examples

| Type | Question | Gold answer (shortened) |
|---|---|---|
| `basic` | What are the default size limits for file uploads and total request size for the new multipart upload support on the OpenAI-compatible API endpoints? | 10 MiB per file and 50 MiB total per request. |
| `semantic` | When does booking open for the new top end 80GB accelerator on dedicated clusters in the EU Central and India South locations? | H200 reservations open Monday 2026-04-06 at 09:00 UTC. |
| `conflicting_info` | On Streamly AI's dedicated pool dp-132-usw, what % of interactive burst credits should be reserved exclusively for priority=high routes? | 30%. An earlier suggestion was 20%, but the applied target is 30%. |
| `high_level` | What is Redwood Inference's mission statement? | Make high-performance, reliable LLM inference accessible to every engineering team. |
| `info_not_found` | Which specific enterprise accounts were on the initial allowlist for the hot-route protection rollout, and what were the exact per-route-group budgets? | Must state that the question is not fully answerable from the documents. |

## How the benchmark scores answers

The upstream benchmark uses an LLM judge and reports:

- **Answer correctness**: binary per question.
- **Answer completeness**: how many of the `answer_facts` the answer covers.
- **Overall score** = mean(correct × completeness), so an incorrect answer scores 0 however
  complete it is. This is the headline number.
- **Document recall** and **invalid extra docs**: retrieval quality against `expected_doc_ids`.

Document-level metrics only apply to questions with gold documents, so `high_level` and
`info_not_found` are judged on the answer alone.

## How TechRAG uses it

The full corpus is too large to index locally, so `techrag-subset` builds a development subset
([bench.py](../src/techrag/benchmarks/bench.py)):

- **Every gold document** (so every question stays answerable), plus
- a **seeded random sample of distractors** from the rest of the corpus.

The size is set by `--distractors` (default 20,000). The subset currently on disk
(`data/subsets/dev/`, seed 0) holds 1,723 documents: 723 gold and 1,000 distractors. The results
in the main README used 20,000 distractors. Documents are written as `<source_type>/<doc_id>.md`, so a retrieved chunk's source
path identifies its document (`doc_id_of`). That is how retrieval is scored.

Because the subset has only a small share of the distractors, it is **easier than the real
benchmark**. Treat absolute scores as relative comparisons between your own systems
(baseline vs. agent, dense vs. BM25 vs. hybrid), not as leaderboard numbers.

| What | Where |
|---|---|
| Subset builder, question loading and sampling | `src/techrag/benchmarks/bench.py` |
| Document recall per question type | `src/techrag/eval/recall.py` |
| LLM answer judge using `answer_facts` | `src/techrag/eval/judge.py` |
| Results and reports | `eval/*.json`, `eval/*_report.txt` |

## Which types the agentic loop should help

This is a hypothesis to check against the per-type results in `eval/`.

- **Likely to benefit** from iterative retrieval and reasoning: `completeness`, `project_related`,
  `conflicting_info` (several documents must be found and reconciled) and `constrained`
  (follow-up searches with the qualifier).
- **Likely unchanged**: `basic`, `semantic`, `miscellaneous`. Single-shot retrieval already
  finds one document.
- **Depends on abstention**: `info_not_found` can get worse with more retrieval, because an
  agent that keeps searching may be more tempted to assemble a plausible-sounding answer.
