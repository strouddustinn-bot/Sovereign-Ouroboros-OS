"""Deterministic, dependency-free embedding utilities.

The README envisions Torch / Sentence-Transformers latent stores. To keep the
reference implementation runnable anywhere, this module provides a small
deterministic pseudo-embedder derived from token hashes. It is *not* a learned
model, but it yields stable, comparable vectors so the imagination and
simulation layers can reason about semantic proximity without heavy deps.
"""

from __future__ import annotations

import hashlib
import math

from sovereign_ouroboros_os.core.types import Vector

DEFAULT_DIM = 64


def embed(text: str, dim: int = DEFAULT_DIM) -> Vector:
    """Map *text* to a unit-norm latent vector of length *dim*.

    Deterministic: the same text always maps to the same vector.
    """
    acc = [0.0] * dim
    tokens = text.lower().split() or [text.lower()]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(dim):
            acc[i] += (digest[i % len(digest)] - 127.5) / 127.5
    norm = math.sqrt(sum(x * x for x in acc)) or 1.0
    return tuple(x / norm for x in acc)


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity between two vectors of equal length."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def blend(vectors: list[Vector], weights: list[float] | None = None) -> Vector:
    """Weighted-average a list of equal-length vectors into one unit vector."""
    if not vectors:
        raise ValueError("cannot blend an empty list of vectors")
    dim = len(vectors[0])
    weights = weights or [1.0] * len(vectors)
    if len(weights) != len(vectors):
        raise ValueError("weights and vectors must be the same length")
    acc = [0.0] * dim
    for vec, w in zip(vectors, weights):
        if len(vec) != dim:
            raise ValueError("all vectors must share the same dimension")
        for i in range(dim):
            acc[i] += vec[i] * w
    norm = math.sqrt(sum(x * x for x in acc)) or 1.0
    return tuple(x / norm for x in acc)
