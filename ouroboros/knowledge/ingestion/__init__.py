"""Knowledge base ingestion subsystem.

Public exports
--------------
Chunker
    Structure-aware small-to-big document chunker.
Embedder
    Batch chunk embedder that persists vectors to the KB store.
IngestPipeline
    End-to-end orchestrator: text → SourceRecord → Document → Chunks → Vectors.
"""

from __future__ import annotations

from .chunker import Chunker
from .embedder import Embedder
from .pipeline import IngestPipeline

__all__ = ["Chunker", "Embedder", "IngestPipeline"]
