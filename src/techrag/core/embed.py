"""Load the embedding model and embed texts with it."""

from functools import cache

import numpy as np
from sentence_transformers import SentenceTransformer


@cache
def load_model(model_name: str) -> SentenceTransformer:
    """Load the embedding model once per process, so repeated queries do not reload it.

    Args:
        model_name: sentence-transformers model name or path.

    Returns:
        The loaded model, cached per ``model_name``.
    """
    return SentenceTransformer(model_name)


def embed(texts: list[str], model_name: str) -> np.ndarray:
    """Return L2-normalized float32 embeddings for ``texts``, one row per text.

    Args:
        texts: Texts to embed.
        model_name: sentence-transformers model to embed with.

    Returns:
        An array of shape ``(len(texts), embedding_dim)``.
    """
    model = load_model(model_name)
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
