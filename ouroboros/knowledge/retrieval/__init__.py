"""Retrieval pipeline for the Ouroboros knowledge base.

Public API
----------
BM25Index
    Pure-Python Okapi BM25 inverted index for sparse keyword retrieval.
HybridSearch
    Dense + sparse retrieval with Reciprocal Rank Fusion (RRF).
ContextAssembler
    Parent expansion, dedup, token-budget packing, and citation building.
"""

from __future__ import annotations

from ouroboros.knowledge.retrieval.bm25 import BM25Index
from ouroboros.knowledge.retrieval.context_assembler import ContextAssembler
from ouroboros.knowledge.retrieval.hybrid_search import HybridSearch

__all__ = ["BM25Index", "HybridSearch", "ContextAssembler"]
