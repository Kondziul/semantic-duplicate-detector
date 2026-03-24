"""
detector.py — Semantic similarity computation and duplicate detection.

Given a set of sentence embeddings, this module computes all pairwise
cosine similarities and returns candidate duplicate pairs above a
configurable threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class DuplicatePair:
    """A candidate duplicate pair with its similarity score."""

    idx_a: int
    idx_b: int
    text_a: str
    text_b: str
    similarity: float


@dataclass
class DetectionResult:
    """Full output of a duplicate-detection run."""

    fragments: List[str]
    similarity_matrix: np.ndarray
    pairs: List[DuplicatePair] = field(default_factory=list)
    threshold: float = 0.80

    @property
    def n_fragments(self) -> int:
        return len(self.fragments)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute a full pairwise cosine similarity matrix.

    Because embeddings are already L2-normalised (see FragmentEmbedder),
    cosine similarity reduces to the dot product, making this a simple
    matrix multiplication — O(N²·D) but vectorised via NumPy/BLAS.

    Args:
        embeddings: Float32 array of shape (N, D).

    Returns:
        Symmetric float32 matrix of shape (N, N) with values in [-1, 1].
    """
    return (embeddings @ embeddings.T).astype(np.float32)


def detect_duplicates(
    fragments: List[str],
    embeddings: np.ndarray,
    threshold: float = 0.80,
) -> DetectionResult:
    """
    Find all fragment pairs whose cosine similarity exceeds `threshold`.

    Only the upper triangle of the similarity matrix is examined to avoid
    reporting (A, B) and (B, A) as separate pairs.

    Args:
        fragments: Original text strings (parallel to `embeddings`).
        embeddings: Normalised embeddings, shape (N, D).
        threshold: Minimum similarity to flag a pair as a candidate duplicate.
                   Typical useful range: 0.75 – 0.95.

    Returns:
        DetectionResult containing the matrix and ranked duplicate pairs.
    """
    sim_matrix = compute_similarity_matrix(embeddings)
    n = len(fragments)

    pairs: List[DuplicatePair] = []

    # Iterate over the strict upper triangle (j > i)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i, j])
            if score >= threshold:
                pairs.append(
                    DuplicatePair(
                        idx_a=i,
                        idx_b=j,
                        text_a=fragments[i],
                        text_b=fragments[j],
                        similarity=score,
                    )
                )

    # Sort by similarity descending so the most likely duplicates appear first
    pairs.sort(key=lambda p: p.similarity, reverse=True)

    return DetectionResult(
        fragments=fragments,
        similarity_matrix=sim_matrix,
        pairs=pairs,
        threshold=threshold,
    )
