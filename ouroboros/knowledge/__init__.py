"""The Ouroboros Knowledge Base: provenance-tracked, hybrid-retrieval RAG layer.

Implements the architecture from knowledge-base-architecture.md:
- Layer-separated: schemas → storage → ingestion → retrieval → serving
- Hybrid retrieval: CharNGram dense embeddings + Okapi BM25 sparse, fused via RRF
- Small-to-big: child chunks for precision, parent chunks for context
- Versioned: supersede-not-overwrite with temporal validity fields
- Provenance-first: every chunk traces back to its source record

Quick start::

    from ouroboros.knowledge import KnowledgeBase

    kb = KnowledgeBase()
    kb.ingest("# Billing\\n## Refunds\\nRefunds take 5 business days...",
              source_uri="s3://docs/billing.md", domain="billing")

    context, citations = kb.assemble_context("how do refunds work?")
    print(context)
"""

from ouroboros.knowledge.knowledge_base import KnowledgeBase
from ouroboros.knowledge.schemas import (
    Chunk,
    ChunkMetadata,
    Document,
    KBQuery,
    RetrievedChunk,
    SourceRecord,
)

__all__ = [
    "KnowledgeBase",
    "Chunk",
    "ChunkMetadata",
    "Document",
    "KBQuery",
    "RetrievedChunk",
    "SourceRecord",
]
