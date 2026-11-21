"""Defaults shared across the pipeline, evaluation and reasoning packages."""

from pathlib import Path

# Retrieval
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
INDEX_DIR = Path(".index/dev")

# Benchmark
SUBSET_DIR = Path("data/subsets/dev")

# Generation (Ollama)
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:3b"

# Judging
JUDGE_MODEL = "qwen3.5:9b"
# Room for the prompt and a verdict with one entry per fact: the longest question has 46 facts,
# which needs about 8k tokens, and Ollama's default window of 4096 cuts the reply off mid-JSON.
NUM_CTX = 16384
