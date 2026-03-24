"""
embedder.py — Sentence embedding wrapper.

Wraps a sentence-transformers model to produce fixed-size embeddings
for arbitrary text fragments. Keeping this in its own module makes it
easy to swap the underlying model without touching downstream code.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


# Default model: all-MiniLM-L6-v2 is small (80 MB), fast, and performs well
# on semantic similarity benchmarks. Swap for a larger model if accuracy matters
# more than latency (e.g. "all-mpnet-base-v2").
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class FragmentEmbedder:
    """Encodes text fragments into dense sentence embeddings."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """
        Args:
            model_name: Any model identifier accepted by sentence-transformers.
        """
        self.model_name = model_name
        print(f"[embedder] Loading model: {model_name}")
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Encode a list of text fragments.

        Args:
            texts: Raw text strings to encode.
            show_progress: Display a tqdm progress bar (useful for large corpora).

        Returns:
            Float32 array of shape (N, embedding_dim).
        """
        embeddings: np.ndarray = self._model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalised → cosine sim == dot product
        )
        return embeddings
