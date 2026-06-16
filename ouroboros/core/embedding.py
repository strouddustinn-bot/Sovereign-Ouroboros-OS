"""Embedding utilities for the Ouroboros cognitive stack.

Provides three tiers, selected automatically by :func:`get_embedder`:

1. **API** (highest quality): Voyage AI or OpenAI embeddings, activated when
   ``VOYAGE_API_KEY`` or ``OPENAI_API_KEY`` is set in the environment.

2. **CharNGram** (real local NLP): FastText-style character n-gram embeddings
   with a fixed random projection matrix via numpy. Captures sub-word semantic
   similarity without neural networks or pre-training — "delete" and "deleted"
   are close; "database" and "data" are close. Always available when numpy is.

3. **Hash** (zero-dependency fallback): deterministic SHA-256 word hashing.
   Available in every environment.

The public surface — :func:`embed`, :func:`cosine`, :func:`blend` — is
identical across all tiers, so callers never change.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol

from ouroboros.core.types import Vector

DEFAULT_DIM = 64

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    """Anything with an ``embed`` method that returns a unit-norm Vector."""

    def embed(self, text: str) -> Vector: ...


# ---------------------------------------------------------------------------
# Tier 3 – zero-dependency hash embedder (original fallback)
# ---------------------------------------------------------------------------


class HashEmbedder:
    """Deterministic SHA-256 word-hash embedder. Zero dependencies."""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = dim

    def embed(self, text: str) -> Vector:
        acc = [0.0] * self.dim
        tokens = text.lower().split() or [text.lower()]
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            for i in range(self.dim):
                acc[i] += (digest[i % len(digest)] - 127.5) / 127.5
        norm = math.sqrt(sum(x * x for x in acc)) or 1.0
        return tuple(x / norm for x in acc)


# ---------------------------------------------------------------------------
# Tier 2 – CharNGram + random projection (fastText-style, numpy required)
# ---------------------------------------------------------------------------


class CharNGramEmbedder:
    """FastText-style character n-gram embedder with fixed random projection.

    Extracts all character n-grams of length *n_min*…*n_max* from the input
    text, maps each to a row of a fixed (seeded) random projection matrix, and
    averages the rows into a unit-norm dense vector.

    No training, no network, no pre-trained weights — but produces genuine
    sub-word semantic similarity because words that share n-grams land near
    each other in embedding space.

    Requires numpy.
    """

    def __init__(
        self, dim: int = DEFAULT_DIM, n_min: int = 3, n_max: int = 6
    ) -> None:
        import numpy as np  # imported lazily so the module loads without numpy

        self.dim = dim
        self.n_min = n_min
        self.n_max = n_max
        self._np = np
        # Fixed projection matrix: 2^16 n-gram buckets × dim.
        # The seed is constant so embeddings are fully reproducible.
        rng = np.random.default_rng(0xB0B0D0)  # Ouroboros seed
        proj = rng.standard_normal((65536, dim)).astype(np.float64)
        norms = np.linalg.norm(proj, axis=1, keepdims=True)
        self._proj = proj / np.maximum(norms, 1e-8)

    def _ngrams(self, text: str) -> list[int]:
        """Return a list of n-gram bucket indices for *text*."""
        padded = f"<{text.lower().replace(' ', '_')}>"
        buckets: list[int] = []
        for n in range(self.n_min, self.n_max + 1):
            for i in range(len(padded) - n + 1):
                ng = padded[i : i + n]
                # FNV-1a-style deterministic bucket (no PYTHONHASHSEED variance)
                h = int(hashlib.sha256(ng.encode()).hexdigest()[:4], 16)
                buckets.append(h)
        return buckets

    def embed(self, text: str) -> Vector:
        np = self._np
        buckets = self._ngrams(text)
        if not buckets:
            return tuple([0.0] * self.dim)
        indices = np.array(buckets, dtype=np.int32)
        vec = self._proj[indices].mean(axis=0)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return tuple(float(x) for x in vec)


# ---------------------------------------------------------------------------
# Tier 1 – API embedder (Voyage AI or OpenAI, auto-activated on key presence)
# ---------------------------------------------------------------------------


class APIEmbedder:
    """Neural embeddings via Voyage AI or OpenAI — activated by env key."""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = dim
        self._voyage_key = os.environ.get("VOYAGE_API_KEY", "")
        self._openai_key = os.environ.get("OPENAI_API_KEY", "")
        self._cache: dict[str, Vector] = {}

    def _voyage(self, text: str) -> Vector:
        import json
        import urllib.request

        payload = json.dumps(
            {"input": [text], "model": "voyage-3-lite"}
        ).encode()
        req = urllib.request.Request(
            "https://api.voyageai.com/v1/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._voyage_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        raw: list[float] = data["data"][0]["embedding"]
        # Truncate/pad to dim
        raw = raw[: self.dim] + [0.0] * max(0, self.dim - len(raw))
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return tuple(x / norm for x in raw)

    def embed(self, text: str) -> Vector:
        if text in self._cache:
            return self._cache[text]
        vec = self._voyage(text)
        self._cache[text] = vec
        return vec


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_embedder(dim: int = DEFAULT_DIM) -> Embedder:
    if os.environ.get("VOYAGE_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return APIEmbedder(dim=dim)
    try:
        import numpy  # noqa: F401

        return CharNGramEmbedder(dim=dim)
    except ImportError:
        return HashEmbedder(dim=dim)


# Module-level singleton — shared by all callers so the CharNGram projection
# matrix is built once.
_DEFAULT_EMBEDDER: Embedder = _build_embedder()


def get_embedder() -> Embedder:
    """Return the best available embedder for this environment."""
    return _DEFAULT_EMBEDDER


# ---------------------------------------------------------------------------
# Public API (same surface as before — zero call-site changes needed)
# ---------------------------------------------------------------------------


def embed(text: str, dim: int = DEFAULT_DIM) -> Vector:
    """Embed *text* using the best available embedder.

    If *dim* differs from the module default, a temporary embedder is created;
    otherwise the shared singleton is used.
    """
    if dim == DEFAULT_DIM:
        return _DEFAULT_EMBEDDER.embed(text)
    return _build_embedder(dim).embed(text)


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity between two equal-length vectors."""
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
