# TechRAG

Retrieval-augmented question answering over technical documentation. Markdown docs are split
into chunks, embedded with
[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5), and stored in a
FAISS index that you can search with natural-language queries. The best-matching chunks can
be passed to a local [Ollama](https://ollama.com) model to generate a cited answer. The repo
includes a set of labelled questions for measuring retrieval quality.

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
`--index-dir` and `--model` must match what was used for ingest. `--mode bm25` ranks by keyword
match (BM25) instead of embedding similarity and prints the BM25 score; chunks sharing no term
with the query are not returned. `--mode hybrid` merges the dense and BM25 rankings with
reciprocal rank fusion and prints the fused score (`sum of 1/(60 + rank)`, so it is only
comparable within one query).

### 3. Ask

```sh
ollama pull llama3.2:3b   # once
uv run techrag-ask "How does broadcasting work?"
```

Retrieves the top `k` chunks for the question, sends them to a local Ollama model, and prints
the answer followed by the numbered sources. The model is instructed to answer only from the
provided passages and cite them like `[1]` or `[2][3]`; the numbers match the source list. If
the passages don't contain the answer it should say so instead of guessing.

```
Tensors are broadcastable when ... [1]

Sources:
[1] pytorch/broadcasting.md  (score 0.812)
[2] pytorch/tensors.md  (score 0.640)
```

Requires an Ollama server running at `http://localhost:11434`. Errors such as an unreachable
server, a model that hasn't been pulled, or a missing index exit with an `error: ...` message.

| Option | Default | Description |
| --- | --- | --- |
| `-k` | `5` | Number of chunks given to the model |
| `--llm` | `llama3.2:3b` | Ollama model to answer with |
| `--ollama-url` | `http://localhost:11434` | Ollama server address |
| `--index-dir` | `.index` | Index to search |
| `--embedding-model` | `BAAI/bge-small-en-v1.5` | Must match the ingest model |
| `--mode` | `dense` | Retrieval method: `dense` (embeddings), `bm25` (keyword match) or `hybrid` (both, fused) |

### 4. Evaluate

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
  generate.py      Answer a question from retrieved chunks with Ollama
  evaluate.py      Run the eval questions
  metrics.py       hit@k and MRR
tests/
```
