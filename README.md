# TechRAG

A reasoning-RAG project: it tests whether a **reasoning LLM that decides for itself what to
search for and which documents to read** answers hard enterprise questions better than
single-shot retrieval. The agent is an off-the-shelf reasoning model (`qwen3.5:9b` on local
[Ollama](https://ollama.com)) in a search loop with `search` and `read_doc` tools, not a trained
model. The model decides when to search, which documents to read and when it has enough evidence;
a small [bounded loop](#how-the-agent-works) (no agent framework) only runs the tools and enforces
step, context and duplicate-call limits, then forces an answer if the budget runs out. It is
prompted to plan each lookup, read whole documents instead of trusting short excerpts, flag
documents that disagree, and say so when the evidence is missing. It is measured on
[EnterpriseRAG-Bench](#reasoning-rag-on-enterpriserag-bench) (about 512k enterprise documents, 500
questions across ten types) against a single-shot RAG baseline, with an LLM judge for grading.
[docs/enterprise-rag-bench.md](docs/enterprise-rag-bench.md) explains the dataset and each
question type.

Underneath is a small [retrieval toolkit](#core-rag-toolkit): Markdown docs are chunked, embedded
with [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5), stored in FAISS and
searched densely, by BM25 or both, with a cited-answer command and an HTTP API.

## Workflow

```mermaid
%%{init: {"themeVariables": {"fontSize": "22px"}, "flowchart": {"nodeSpacing": 40, "rankSpacing": 50, "padding": 14}}}%%
flowchart LR
    Q["Question"]

    subgraph BASE["Baseline: single-shot RAG"]
        direction TB
        B1["Retrieve top-k chunks once"] --> B2["LLM answers<br/>from those chunks"]
    end

    subgraph AGENT["Agent: reasoning LLM in a search loop"]
        direction TB
        A1["Thought: plan the next step"] --> A2{"Call a tool<br/>(max 4 per turn,<br/>exact repeats skipped)"}
        A2 -->|"search(query)"| A3["Hits: doc id, title, excerpt"]
        A2 -->|"read_doc(doc_id)"| A4["Full document, page by page"]
        A3 --> A1
        A4 --> A1
        A1 -->|"replies without a tool call"| A5["Final answer, or<br/>'cannot fully answer'"]
        A1 -.->|"8 turns used, ~37k-char context,<br/>or empty reply"| A6["Forced answer<br/>(tools off)"]
        A6 --> A5
    end

    IDX[("Index of the dev subset<br/>723 gold docs + seeded distractors")]
    B1 -.-> IDX
    A3 -.-> IDX
    A4 -.-> IDX

    Q --> B1
    Q --> A1
    B2 --> J
    A5 -->|"answer, stop reason,<br/>docs seen, trace"| J["LLM judge<br/>checks each answer fact<br/>score = correctness x completeness"]
    J --> C["techrag-compare<br/>paired scores, bootstrap CI, cost"]
```

## Results

On all 500 questions over a 2,723-document subset (723 gold documents plus 2,000 distractors,
`data/subsets/dev2k`), the reasoning agent scored **0.745** against the baseline's **0.529**:
+0.215, 95% bootstrap CI +0.176 to +0.256. It costs about 1.6 times as much time per question
(23.4 s and 3.5 model turns on average, against 15.0 s and a single model call), and about
10,800 tokens. 

| Question type | n | Baseline | Agent | Δ |
| --- | ---: | ---: | ---: | ---: |
| constrained | 30 | 0.354 | 0.683 | +0.330 |
| project_related | 40 | 0.262 | 0.586 | +0.324 |
| intra_document_reasoning | 40 | 0.687 | 0.925 | +0.238 |
| completeness | 20 | 0.342 | 0.572 | +0.230 |
| semantic | 125 | 0.375 | 0.598 | +0.223 |
| basic | 175 | 0.628 | 0.848 | +0.220 |
| conflicting_info | 20 | 0.613 | 0.736 | +0.123 |
| miscellaneous | 20 | 0.796 | 0.898 | +0.102 |
| info_not_found | 20 | 1.000 | 1.000 | 0 |
| high_level | 10 | 0.432 | 0.408 | −0.023 |
| **All** | **500** | **0.529** | **0.745** | **+0.215** |

The agent wins on every type except `high_level` (10 questions, a difference well inside the
noise) and `info_not_found`, where both score 1.000. Even `basic` and `semantic` lookups improve
(+0.22 each), and with 175 and 125 questions those rows are reliable. The agent answered on its own for 480 of 500 questions; 13 hit the context limit and 7
the step limit. Retrieval is still the bottleneck: of the 470 questions that have gold
documents, the agent saw all of them for 395 (score 0.82) and missed some for 75 (0.31). The
baseline shows the same split (0.59 against 0.13 for 391 and 79 questions). Both runs took
several hours on a laptop: about 3.3 h for the agent and 2.1 h for the baseline.

To reproduce:

```sh
uv run techrag-subset --out-dir data/subsets/dev2k --distractors 2000
uv run techrag-ingest data/subsets/dev2k --index-dir .index/dev2k
uv run techrag-baseline --subset data/subsets/dev2k --index-dir .index/dev2k --output eval/baseline2k_q500.jsonl
uv run techrag-agent --subset data/subsets/dev2k --index-dir .index/dev2k --output eval/agent2k_q500.jsonl
uv run techrag-compare eval/baseline2k_q500.jsonl eval/agent2k_q500.jsonl
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Reasoning RAG on EnterpriseRAG-Bench

Everything runs locally on Ollama (`qwen3.5:9b` answers, researches and judges), and the commands
share defaults, so the steps run in order with no flags. Requires the
[Setup](#setup) above, the dataset under `data/external/EnterpriseRAG-Bench` (git-ignored) and
`ollama pull qwen3.5:9b`.

### The questions

The 500 questions come in ten types, which differ in what makes them hard:

| Type | Count | Gold docs | Tests |
| --- | ---: | ---: | --- |
| `basic` | 175 | 1 | A direct lookup |
| `semantic` | 125 | 1 | Phrasing with little keyword overlap |
| `intra_document_reasoning` | 40 | 1 | Combining distant sections of one document |
| `project_related` | 40 | 2–9 | Aggregating several documents of a project |
| `constrained` | 30 | 1–2 | Qualifiers (date, region, customer) pick one of several similar documents |
| `conflicting_info` | 20 | 2 | Documents disagree; the answer must reconcile them |
| `completeness` | 20 | 2–10 | Needs every relevant document |
| `miscellaneous` | 20 | 1 | Informal or off-topic documents |
| `info_not_found` | 20 | 0 | The corpus has no answer; the model should say so |
| `high_level` | 10 | 0 | Broad question with no single source |

See [docs/enterprise-rag-bench.md](docs/enterprise-rag-bench.md) for examples and scoring.

### Build a dev subset and index it

The full corpus is too big for a laptop, so `techrag-subset`
writes every document that answers a question (723 of them) plus seeded random
distractors (`--distractors`, default 20,000) to `data/subsets/dev/`, laid out as `<source_type>/<doc_id>.md`, with the questions
in `questions.jsonl`:

```sh
uv run techrag-subset                                      # --distractors, --seed, --out-dir
uv run techrag-ingest data/subsets/dev
```

### Measure retrieval

`techrag-recall` reports document-level retrieval at `k` = 5, 10, 20, 50, 100 chunks, overall and
per question type, with two metrics:

- **recall** (`R@k`): the mean fraction of a question's gold documents found in the top `k` chunks.
- **complete** (`all@k`): the share of questions whose gold documents were *all* found in the top
  `k` chunks. This is stricter, and matters for questions that need several documents (e.g.
  `completeness`), since a missing document means the answer can't be complete.

Questions with no gold documents are skipped. Takes `--mode` (default `dense`); the report is
written to `eval/recall.json`.

```sh
uv run techrag-recall --mode hybrid
```

### Grade answers

[judge.py](src/techrag/eval/judge.py) scores an answer against the question's
`answer_facts` checklist with an LLM judge. `score = correctness x completeness`: completeness is
the share of facts the answer states, and correctness drops to 0 if the answer contradicts the
reference or makes a claim a fact forbids. Because a biased judge distorts every later number,
`techrag-judge` probes it with answers whose score is known (the reference answer, a refusal,
half the facts, another question's answer) and reports how often it agrees:

```sh
uv run techrag-judge -n 20          # --seed, --model, --think; writes eval/judge_selftest.json
                                    # (--no-mlflow skips the MLflow run, see below)
```

### Run the baseline and the agent

`techrag-baseline` retrieves the top `k` chunks once (hybrid by default) and answers from them.
`techrag-agent` instead gives the reasoning model two tools, `search` (finds passages, shows
document ids, titles and excerpts) and `read_doc` (reads a whole document page by page), and lets
it use them for up to `--max-steps` turns before answering. Before each call it states what it is
looking for. `--think` also turns on the model's own reasoning mode. Both grade every answer with
the judge and print scores by question type.

```sh
uv run techrag-baseline --per-type 8
uv run techrag-agent --per-type 8
```

| Option | Default | Description |
| --- | --- | --- |
| `--subset` | `data/subsets/dev` | Subset with the questions |
| `--index-dir` | `.index/dev` | Index of the subset |
| `--mode` | `hybrid` | Retrieval method |
| `--llm` | `qwen3.5:9b` | Model that answers (baseline) or researches (agent) |
| `--think` | off | Let the model reason before replying |
| `--judge-model` | `qwen3.5:9b` | Model that grades |
| `-n` / `--per-type` | all questions | Seeded sample: `n` random questions, or this many of every type. Not both |
| `--seed` | `0` | Sampling seed. Growing `--per-type` only adds questions, so saved results stay valid |
| `--output` | `eval/baseline.jsonl`, `eval/agent.jsonl` | JSON Lines results |
| `--run-name` | name of the results file | Name recorded in the [run manifest](#run-manifest) |
| `--fresh` | off | Discard earlier results first |
| `--no-mlflow` | off | Do not record the run in [MLflow](#track-runs-in-mlflow) |
| `-k` | `10` | Chunks given to the model (baseline only) |
| `--max-steps` | `8` | Model turns per question (agent only) |
| `--ids` | all | Comma-separated question ids to run (agent only) |

Each question is appended to the results file as soon as it is graded, so an interrupted run
resumes where it stopped, and questions that errored are retried on the next run.

#### Run manifest

Both commands also save the settings of the run next to its results, as `eval/baseline.manifest.json`
or `eval/agent.manifest.json` (named after `--output`; git-ignored like the results). The schema is
`RunConfig` in [config.py](src/techrag/common/config.py), a set of frozen Pydantic models that
reject unknown fields and out-of-range values, so a bad setting such as `--max-steps 0` stops the
run before any model loads.

| Section | Records |
| --- | --- |
| `llm` | Model name, temperature, `max_tokens`, context window |
| `retrieval` | `mode`, `k`, embedding model, `index_version` (a SHA-256 of `index.faiss` and `chunks.json`), `index_dvc` (the [DVC version](#version-data-and-indexes-with-dvc) of the index directory) |
| `agent` | `max_steps` and `think`. The baseline is one model turn, so it records `max_steps=1` |
| `evaluation` | Judge model and temperature |
| `data` | Subset name, a SHA-256 of its `questions.jsonl` and `manifest.json`, `dataset_dvc` (the DVC version of the subset directory), and how questions were sampled: `seed`, `sample_size` (`-n`), `per_type`, `question_ids` (`--ids`) |
| `metadata` | Run name, git commit, UTC timestamp, author (from `git config user.name`) |

The manifest is rewritten on every start, so resuming with different options leaves it describing
only the latest settings. `RunConfig.load(path)` reads one back and validates it.

#### Track runs in MLflow

`techrag-baseline`, `techrag-agent` and `techrag-judge` each record their run in
[MLflow](https://mlflow.org), so runs can be sorted, filtered and compared in a browser. Pass
`--no-mlflow` to skip it; MLflow is then not even imported.

```sh
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://localhost:5000
```

Runs are stored in `mlflow.db` in the directory you run from, with artifacts under `mlruns/` next
to it (both git-ignored). Set `MLFLOW_TRACKING_URI` to use another store or a tracking server. Run
the commands from one directory, or they will each start a store of their own.

| Experiment | Runs | Named |
| --- | --- | --- |
| `techrag` | Baseline and agent, told apart by the `system` tag | `<system>-<model>-<mode>`, or `--run-name` |
| `techrag-judge` | Judge self-tests | `judge-<model>` |

What a baseline or agent run records:

| Kind | Contents |
| --- | --- |
| Parameters | Every setting of the [run manifest](#run-manifest) except the metadata, named `section.field` (`retrieval.k`, `agent.max_steps`, `data.dataset_hash`, ...) |
| Tags | `system`, `resumed`, `started_at`, `author`, `git_sha`, and `git_dirty` (uncommitted or untracked files, so a SHA alone does not claim to reproduce the run) |
| Metrics | `score`, `correctness`, `completeness`, `n_questions`, `errors`; the same by question type (`by_type/<type>/score`) and by retrieval group (`by_retrieval/<group>/score`); mean `answer_seconds` and `gold_found`; for the agent, mean `steps` and `tokens` and a count per stop reason (`stop/answered`, `stop/max_steps`, ...) |
| Artifacts | `manifest.json`, the results JSON Lines file, `question_ids.json` when `--ids` is used, and for the agent `agent_trace.json` |

`agent_trace.json` maps each question id to its score, steps, tokens and stop reason, and to every
tool call with its arguments and the documents it returned. A call's `thought` is the text the
model wrote alongside it, so it is empty when the model calls a tool without saying anything.

A judge run records its model, `--think`, context size, `-n` and `--seed` as parameters, and how far
the judge was from the known score: `mean_abs_error` and `within_0.25` overall and for each probe
kind (`reference/observed`, `abstain/mean_abs_error`, ...). Its artifact is the report JSON.

Notes on how runs are recorded:

- Resuming a results file starts a new MLflow run, tagged `resumed=true`, because MLflow does not
  let a run change its parameters. Its metrics cover every question of the invocation, including
  those answered earlier. Only the latest invocation's parameters are recorded, as in the manifest.
- An interrupted run (Ctrl-C) ends as `KILLED` and still logs the metrics and results of the
  questions it finished. Any other error ends it as `FAILED`.
- A bad setting stops the command before a run is created. If MLflow cannot record something once
  the run is going, it logs a warning and the evaluation continues; the results file is still
  complete.

#### How the agent works

[agent.py](src/techrag/reasoning/agent.py) is a bounded loop with no agent framework: the model
decides what to search for, what to read and when it has enough; Python only runs the tools and
enforces limits. The loop talks to the model through a plain `AgentChat` function, so it can be
tested or pointed at another backend without Ollama.

| Tool | Returns |
| --- | --- |
| `search(query, k=8)` | Up to `k` (1–15) documents as `[doc_id] source - title` plus a ~700-character excerpt, from hybrid chunk search grouped by document |
| `read_doc(doc_id, start=0)` | A 6,000-character page of the document, with a footer giving the next `start` if it continues |

Each turn the model either calls tools (at most 4 are run per turn) or replies in plain prose,
which is taken as the final answer. Tool results are appended to the conversation and the loop
repeats. A bad argument, unknown tool or unknown document id comes back to the model as text, so
it can correct itself instead of crashing the run. An exact repeat of an earlier call is not run;
the model is told to try a different query.

The run ends in one of four ways, recorded as `stop` in each result:

| `stop` | Meaning |
| --- | --- |
| `answered` | The model replied without calling a tool |
| `max_steps` | The step budget (`--max-steps`, default 8) ran out |
| `context` | The prompt passed about 37k characters (75% of the 16k-token window at 3 characters per token). Ollama silently drops the start of an overlong prompt, which would lose the question |
| `empty` | The model replied with neither text nor a tool call |

Any stop but `answered` triggers one last turn with tools disabled and an instruction to answer
from what was found, or to say what is missing. The result keeps the answer, turns, tokens, the
documents the model was shown and a per-call trace (the model's one-line thought, tool,
arguments, documents shown). The judge grades it the same way as the baseline, and the share of
gold documents the agent saw (`gold_found`) is recorded, which is how the finding above about
retrieval being the bottleneck was measured.

Sampling is greedy (`temperature=0`) for repeatability, and `think` is off by default because the
agent already writes a line of reasoning before each call and thinking makes every step several
times slower.

### Compare them

`techrag-compare` pairs the two result files question by question and reports
the per-type and overall score difference with a seeded bootstrap 95% confidence interval, plus
the cost (seconds, model turns and tokens per question). `--names A B` relabels the columns.

```sh
uv run techrag-compare eval/baseline.jsonl eval/agent.jsonl
```

### Version data and indexes with DVC

The subsets in `data/subsets/` and the indexes in `.index/` are too big for git, so
[DVC](https://dvc.org) tracks them: git holds small pointer files (`*.dvc` and `dvc.lock`) and the
data lives in an S3 remote (`.dvc/config`). DVC is in its own dependency group, so evaluating needs
only `uv sync`; for the commands below use `uv run --group data dvc ...`.

```sh
dvc pull                 # fetch the data and indexes the committed pointers name
dvc repro                # rebuild the subsets and indexes that are out of date
dvc push                 # after a rebuild: upload the new data, then commit dvc.lock
dvc add data/other       # track a directory no stage builds (and git add the .dvc file it writes)
```

`dvc.yaml` has two stages per subset (`dev1k`, `dev2k`, `dev4k`), and `dvc.lock` records the
hashes of what each one read and produced:

| Stage | Runs | Rebuilds when these change |
| --- | --- | --- |
| `subset@dev4k`, ... | `techrag-subset` with seed 0 and 1,000, 2,000 or 4,000 distractors | The benchmark files, `bench.py`, the command in `dvc.yaml` |
| `ingest@dev4k`, ... | `techrag-ingest` | The subset, `ingest.py`, `embed.py`, the chunk size, overlap or embedding model in `params.yaml` |

`dvc repro` needs the benchmark under `data/external/EnterpriseRAG-Bench`, which DVC hashes but
does not store. Without it, `dvc repro --allow-missing` skips stages whose only difference is
missing files, and `dvc repro --single-item ingest@dev4k` checks one index against the pulled
subset.

Every baseline and agent run records the DVC version of the index and of the subset it read
(`retrieval.index_dvc` and `data.dataset_dvc` in the manifest, so also MLflow parameters):

| Field | Meaning |
| --- | --- |
| `md5` | The hash `dvc add` would record for the files on disk when the run started (`….dir`). After `dvc push`, `dvc pull` restores exactly those files |
| `committed` | `true` if `md5` is the version in `dvc.lock` or the `.dvc` file, `false` if the files were changed since. A run with `false` read an index that no commit describes |

These are in addition to `index_version`, which is a hash of the files actually loaded. A directory
DVC does not track records no DVC version.

## Core RAG toolkit

The retrieval pieces the experiment above builds on, usable on their own over any Markdown docs
(the examples below use the dev subset built above).

### 1. Build the index

```sh
uv run techrag-ingest [data_dir]
```

Reads every `*.md` file under `data_dir` (default `data/subsets/dev`), chunks it, embeds the chunks, and
writes the index to `.index/dev/` (git-ignored). The embedding model is downloaded on first use.

| Option | Default | Description |
| --- | --- | --- |
| `--index-dir` | `.index/dev` | Where to write the index |
| `--size` | `500` | Max characters per chunk |
| `--overlap` | `50` | Characters shared by adjacent chunks |
| `--model` | `BAAI/bge-small-en-v1.5` | sentence-transformers model |

### 2. Search

```sh
uv run techrag-retrieve "What is our parental leave policy?" -k 3
```

Prints the `k` most similar chunks (default 5) with their source file and cosine similarity.
`--index-dir` and `--model` must match what was used for ingest. `--mode bm25` ranks by keyword
match (BM25) instead of embedding similarity and prints the BM25 score; chunks sharing no term
with the query are not returned. `--mode hybrid` merges the dense and BM25 rankings with
reciprocal rank fusion and prints the fused score (`sum of 1/(60 + rank)`, so it is only
comparable within one query).

### 3. Ask

```sh
ollama pull llama3.2:3b   # once
uv run techrag-ask "What is our parental leave policy?"
```

Retrieves the top `k` chunks for the question, sends them to a local Ollama model, and prints
the answer followed by the numbered sources. The model is instructed to answer only from the
provided passages and cite them like `[1]` or `[2][3]`; the numbers match the source list. If
the passages don't contain the answer it should say so instead of guessing.

```
Parental leave is ... [1]

Sources:
[1] hr/doc_123.md  (score 0.812)
[2] hr/doc_456.md  (score 0.640)
```

Requires an Ollama server running at `http://localhost:11434`. Errors such as an unreachable
server, a model that hasn't been pulled, or a missing index exit with an `error: ...` message.

| Option | Default | Description |
| --- | --- | --- |
| `-k` | `5` | Number of chunks given to the model |
| `--llm` | `llama3.2:3b` | Ollama model to answer with |
| `--ollama-url` | `http://localhost:11434` | Ollama server address |
| `--index-dir` | `.index/dev` | Index to search |
| `--embedding-model` | `BAAI/bge-small-en-v1.5` | Must match the ingest model |
| `--mode` | `dense` | Retrieval method: `dense` (embeddings), `bm25` (keyword match) or `hybrid` (both, fused) |

### 4. Serve

```sh
uv run techrag-serve
```

Starts an HTTP API on `http://127.0.0.1:8000`. The index is read and the embedding model is
loaded once at startup (so it can take a few seconds to come up), not per request. A missing
or inconsistent index exits with an `error: ...` message. Interactive docs are at `/docs`.

| Option | Default | Description |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Address to listen on |
| `--port` | `8000` | Port to listen on |
| `--index-dir` | `.index/dev` | Index to serve |
| `--embedding-model` | `BAAI/bge-small-en-v1.5` | Must match the ingest model |
| `--llm` | `llama3.2:3b` | Ollama model used by `/ask` |
| `--ollama-url` | `http://localhost:11434` | Ollama server address |

| Endpoint | Description |
| --- | --- |
| `GET /healthz` | Returns `{"status": "ok", "chunks": <number of indexed chunks>}` |
| `POST /search` | Returns the top `k` chunks for a query, as in [Search](#2-search) |
| `POST /ask` | Answers a query from the top `k` chunks with Ollama, as in [Ask](#3-ask) |

`/search` and `/ask` take the same JSON body; `k` (default `5`, at least 1) and `mode` (`dense`,
`bm25` or `hybrid`; default `dense`) are optional:

```sh
curl -s localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query": "What is our parental leave policy?", "k": 3, "mode": "hybrid"}'
```

```json
{"results": [{"source": "hr/doc_123.md", "text": "...", "score": 0.033}]}
```

`/ask` returns the answer and the chunks it was generated from, numbered by position, so a
citation like `[2]` refers to `sources[1]`:

```json
{"answer": "Parental leave is ... [1]", "sources": [{"source": "hr/doc_123.md", "text": "...", "score": 0.812}]}
```

A blank query, a `k` below 1 or an unknown `mode` returns `422`. If Ollama is unreachable or
replies unexpectedly, `/ask` returns `502`.

## Docker

Docker is for running the API, not for development. Build the index, search, ask and
evaluate locally with `uv run ...` as above. The image runs `techrag-ingest` once at build
time and does not include `eval/`, so to pick up changes to `data/quicktest/` or the code you rebuild
the image rather than re-ingesting inside a container.

```sh
docker compose up --build
docker compose exec ollama ollama pull llama3.2:3b   # once
```

Builds the API image and the web UI image and starts them alongside an Ollama container. The
web UI is on `http://localhost:8080` (nginx serves the built React app from [web/](web/) and
forwards `/api/*` to the API), and the API is on `http://localhost:8000` (see
[Serve](#4-serve) for the endpoints). The image runs
`techrag-ingest` on [data/quicktest/](data/quicktest/) at build time, so the index and the embedding model are
baked in and the container needs no network access to the Hugging Face Hub at runtime.
Rebuild (`docker compose up --build`) after changing the docs in `data/quicktest/`.

- `/ask` needs the Ollama model to be pulled into the `ollama` service, as above. It is stored
  in the `ollama` volume, so this is only needed once. `/search` and `/healthz` work without it.
- The container listens on `0.0.0.0` and runs as a non-root user. Extra `techrag-serve`
  options (such as `--llm`) can be added under `command:` in
  [docker-compose.yml](docker-compose.yml); the existing `--ollama-url` points at the
  `ollama` service.
- The image has a healthcheck that polls `/healthz`.

To build and run the API image on its own:

```sh
docker build -t techrag .
docker run --rm -p 8000:8000 techrag --ollama-url http://host.docker.internal:11434
```

`host.docker.internal` works on Docker Desktop; on Linux add
`--add-host host.docker.internal:host-gateway`.

## Development

```sh
uv run pytest        # tests
uv run ruff check    # lint
```

## Layout

```
.index/            Indexes built by techrag-ingest (versioned by DVC, not git)
data/subsets/      Benchmark subsets built by techrag-subset (versioned by DVC, not git)
data/external/     Downloaded datasets, e.g. EnterpriseRAG-Bench (git-ignored)
dvc.yaml           Pipeline: benchmark to subsets to indexes; dvc.lock records what it built
params.yaml        Chunk size, overlap and embedding model for the ingest stages
docs/              Notes on the EnterpriseRAG-Bench data and question types
eval/              Generated results and reports
src/techrag/
  api.py             FastAPI app serving search and ask
  common/
    config.py        Shared defaults (models, paths, Ollama URL) and the run manifest schema
    dvc_version.py   The DVC version of a data or index directory, checked against dvc.lock
    ollama.py        Minimal Ollama chat client
    tracking.py      MLflow runs: parameters, tags, metrics and artifacts, or a no-op when off
  core/
    ingest.py        Load, chunk, and store documents
    embed.py         Load the embedding model and embed texts
    retrieve.py      Query the index (dense, BM25 or hybrid)
    bm25.py          BM25 keyword ranking
    generate.py      Answer a question from retrieved chunks with Ollama
  benchmarks/
    bench.py         Load EnterpriseRAG-Bench, build the dev subset, sample questions
  eval/
    recall.py        Document-level retrieval recall on the benchmark questions
    judge.py         LLM judge (correctness x completeness) and its self-test
  reasoning/
    harness.py       Shared run, grade, log and report machinery, and manifest helpers
    baseline.py      Single-shot RAG baseline, graded and logged
    agent.py         Agentic search loop (search / read_doc tools), graded and logged
    metrics.py       Metrics and the agent trace to log to MLflow from a run's records
    compare.py       Paired comparison of two result files with a bootstrap CI
tests/
.github/workflows/ CI: ruff and pytest
Dockerfile         Image with the index and embedding model baked in
docker-compose.yml Web UI and API, plus an Ollama server
```
