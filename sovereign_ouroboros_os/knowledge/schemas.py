"""Knowledge base data schemas: Source, Document, Chunk, Query, RetrievedChunk.

Follows the architecture defined in knowledge-base-architecture.md:
- Provenance-first: every chunk traces back to a source record.
- Versioned: supersede-not-overwrite with temporal validity.
- Hybrid retrieval: chunks carry both dense vectors and BM25 keywords.
- Small-to-big: chunks carry an optional parent_id for context expansion.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from sovereign_ouroboros_os.core.types import Vector

# ---------------------------------------------------------------------------
# Controlled vocabularies (open-ended but documented)
# ---------------------------------------------------------------------------

SOURCE_TYPES = frozenset(
    {"doc", "web", "pdf", "code", "ticket", "db", "transcript"}
)
ACCESS_LEVELS = frozenset({"public", "internal", "restricted"})
EMBEDDING_MODEL = "CharNGramEmbedder-dim64"
EMBEDDING_DIM = 64


# ---------------------------------------------------------------------------
# Provenance root
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceRecord:
    """Immutable provenance record for an ingested source artifact.

    ``id`` is deterministic: ``"src_" + sha256(uri)[:16]`` so re-ingesting
    the same URI yields the same id and triggers an UPSERT, never a duplicate.
    """

    id: str
    uri: str
    source_type: str
    checksum: str          # sha256 of the raw content bytes
    ingested_at: str       # ISO-8601
    last_seen_at: str
    access_level: str
    authority: float       # 0–1; used as a retrieval tiebreaker
    status: str            # "active" | "archived"

    @staticmethod
    def make_id(uri: str) -> str:
        return "src_" + hashlib.sha256(uri.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Normalised document (source of truth)
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """Normalised full document — the source of truth for its chunks.

    ``id`` is deterministic: ``"doc_" + sha256(source_id + title)[:16]``.
    """

    id: str
    source_id: str
    title: str
    content: str
    content_hash: str      # sha256 of normalised content
    section_tree: list[dict[str, Any]]   # [{"heading": ..., "children": [...]}]
    language: str
    version: int
    created_at: str
    updated_at: str

    @staticmethod
    def make_id(source_id: str, title: str) -> str:
        return "doc_" + hashlib.sha256(
            f"{source_id}:{title}".encode()
        ).hexdigest()[:16]

    @staticmethod
    def make_content_hash(content: str) -> str:
        return "sha256:" + hashlib.sha256(content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Chunk metadata (structured, filterable)
# ---------------------------------------------------------------------------


@dataclass
class ChunkMetadata:
    """Rich, filterable metadata attached to every retrievable chunk."""

    domain: str
    language: str
    source_type: str
    source_uri: str
    version: int
    access_level: str
    authority: float
    created_at: str
    updated_at: str
    title: str | None = None
    topics: list[str] = field(default_factory=list)
    valid_from: str | None = None
    valid_until: str | None = None     # None = current / not yet expired
    superseded_by: str | None = None   # chunk id that replaces this chunk


# ---------------------------------------------------------------------------
# The retrievable unit (the critical schema)
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """The atomic unit stored, embedded, indexed, and returned by retrieval.

    ``id`` is deterministic: ``sha256(source_id:chunk_index:content_hash)[:16]``
    so re-ingesting a document UPSERTs unchanged chunks and mints new ids only
    for changed content — enabling exact change detection without full re-embed.
    """

    id: str
    document_id: str
    source_id: str
    content: str
    content_hash: str       # sha256 of normalised content
    chunk_index: int
    section_path: list[str] # e.g. ["Billing", "Refunds"]
    tokens: int
    vector: Vector          # dense embedding
    vector_model: str
    vector_dim: int
    keywords: list[str]     # for BM25 / sparse retrieval
    entities: list[str]     # named entities for graph linking / filtering
    metadata: ChunkMetadata
    parent_id: str | None = None   # larger chunk for small-to-big expansion
    tokens_estimate: int = 0  # estimated tokens of the parent

    @staticmethod
    def make_id(source_id: str, chunk_index: int, content_hash: str) -> str:
        raw = f"{source_id}:{chunk_index}:{content_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def make_content_hash(content: str) -> str:
        return "sha256:" + hashlib.sha256(content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Query contract
# ---------------------------------------------------------------------------


@dataclass
class KBQuery:
    """Runtime retrieval request, following the spec in the architecture doc."""

    text: str
    filters: dict[str, Any] = field(default_factory=dict)
    k_dense: int = 10
    k_sparse: int = 10
    k_rerank: int = 5
    expand_to_parent: bool = True

    # Convenience filter properties -----------------------------------------

    @property
    def domain(self) -> str | None:
        return self.filters.get("domain")

    @property
    def access_levels(self) -> list[str]:
        v = self.filters.get("access_level")
        if v is None:
            return list(ACCESS_LEVELS)
        return [v] if isinstance(v, str) else list(v)

    @property
    def valid_now(self) -> bool:
        return bool(self.filters.get("valid_now", True))


# ---------------------------------------------------------------------------
# Retrieval result
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A chunk returned by the retrieval pipeline, annotated with scores."""

    chunk: Chunk
    score: float           # final rerank score
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0
    expanded: bool = False  # True if swapped for its parent_id chunk
