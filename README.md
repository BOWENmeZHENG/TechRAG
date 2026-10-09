# TechRAG

Retrieval over technical documentation. Markdown docs are split into chunks, embedded with
[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5), and stored in a
FAISS index that you can search with natural-language queries. The repo includes a set of
labelled questions for measuring retrieval quality.

The sample corpus is a handful of PyTorch notes in [data/pytorch/](data/pytorch/).

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Usage

### 1. Build the index

```sh
uv run techrag-ingest [data_dir]
```

Reads every `*.md` file under `data_dir` (default `data`), chunks it, embeds the chunks, and
writes the index to `.index/` (git-ignored). The embedding model is downloaded on first use.

| Option | Default | Description |
| --- | --- | --- |
| `--index-dir` | `.index` | Where to write the index |
| `--size` | `500` | Max characters per chunk |
| `--overlap` | `50` | Characters shared by adjacent chunks |
| `--model` | `BAAI/bge-small-en-v1.5` | sentence-transformers model |

### 2. Search

```sh
uv run techrag-retrieve "How does broadcasting work?" -k 3
```

Prints the `k` most similar chunks (default 5) with their source file and cosine similarity.
`--index-dir` and `--model` must match what was used for ingest.

### 3. Evaluate

```sh
uv run techrag-eval -k 5
```

Runs every question in [eval/questions.jsonl](eval/questions.jsonl) against the index and
reports:

- **hit@k**: fraction of questions whose answer appears in the top `k` chunks.
- **MRR**: mean reciprocal rank of the first relevant chunk (a miss counts as 0).

A retrieved chunk counts as relevant if it comes from the question's `source` file and
contains its `evidence` text (case and whitespace are ignored). Full per-question output,
including the retrieved chunks, is written to `eval/results.json` (git-ignored; change it
with `--output`).

Each line of the questions file is a JSON object:

```json
{"query": "When are two tensors 'broadcastable'?", "source": "pytorch/broadcasting.md", "evidence": "dimension sizes must either be equal, one of them is 1, or one of them does not exist"}
```

`source` is relative to the data directory.

## Development

```sh
uv run pytest        # tests
uv run ruff check    # lint
```

## Layout

```
data/              Markdown documentation to index
eval/              Labelled questions (and generated results)
src/techrag/
  ingest.py        Load, chunk, embed, and store documents
  retrieve.py      Query the index
  evaluate.py      Run the eval questions
  metrics.py       hit@k and MRR
tests/
```
