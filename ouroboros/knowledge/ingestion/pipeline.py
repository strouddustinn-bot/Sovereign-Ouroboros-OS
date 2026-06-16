"""Ingestion orchestrator: text → SourceRecord → Document → Chunks → Vectors.

:class:`IngestPipeline` is the single public entry-point for adding content to
the Ouroboros knowledge base.  It wires together the three lower-level
components:

1. Schema construction  — builds :class:`SourceRecord` and :class:`Document`.
2. :class:`Chunker`     — splits the document into parent + child chunks.
3. :class:`Embedder`    — embeds every chunk and persists vectors.

All database writes go through the provided :class:`SQLiteKBStore` so the
pipeline is storage-agnostic and trivially testable with an in-memory store.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ouroboros.knowledge.schemas import (
    Chunk,
    ChunkMetadata,
    Document,
    SourceRecord,
)
from ouroboros.knowledge.storage.sqlite_store import SQLiteKBStore

from .chunker import Chunker, _scan_section_tree
from .embedder import Embedder

if TYPE_CHECKING:
    pass


def _sha256_hex(data: bytes) -> str:
    """Return the hex SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


class IngestPipeline:
    """End-to-end ingestion orchestrator.

    Parameters
    ----------
    store:
        Open :class:`SQLiteKBStore` that receives all upserted records.
    """

    def __init__(self, store: SQLiteKBStore) -> None:
        self._store = store
        self._chunker = Chunker()
        self._embedder = Embedder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        text: str,
        source_uri: str,
        domain: str,
        source_type: str = "doc",
        title: str = "",
        language: str = "en",
        access_level: str = "public",
        authority: float = 0.8,
        topics: list[str] | None = None,
    ) -> tuple[str, str]:
        """Ingest *text* into the knowledge base.

        Parameters
        ----------
        text:
            Raw document content.
        source_uri:
            Canonical URI identifying the source (e.g. a URL or file path).
            Re-ingesting the same URI triggers an UPSERT, never a duplicate.
        domain:
            Logical domain the document belongs to (e.g. ``"billing"``).
        source_type:
            One of the controlled ``SOURCE_TYPES`` vocabularies (``"doc"``
            by default).
        title:
            Human-readable document title (may be empty).
        language:
            BCP-47 language tag (default ``"en"``).
        access_level:
            One of ``"public"``, ``"internal"``, or ``"restricted"``.
        authority:
            Float in ``[0, 1]`` used as a retrieval tiebreaker.
        topics:
            Optional list of topic tags for the document.

        Returns
        -------
        tuple[str, str]
            ``(source_id, document_id)`` — the deterministic IDs assigned to
            the new (or updated) source record and document.
        """
        now: str = datetime.now(timezone.utc).isoformat()
        topics = topics or []

        # ------------------------------------------------------------------
        # 1. SourceRecord
        # ------------------------------------------------------------------
        source_id = SourceRecord.make_id(source_uri)
        checksum = "sha256:" + _sha256_hex(text.encode())
        source = SourceRecord(
            id=source_id,
            uri=source_uri,
            source_type=source_type,
            checksum=checksum,
            ingested_at=now,
            last_seen_at=now,
            access_level=access_level,
            authority=authority,
            status="active",
        )
        self._store.upsert_source(source)

        # ------------------------------------------------------------------
        # 2. Document
        # ------------------------------------------------------------------
        doc_id = Document.make_id(source_id, title)
        doc = Document(
            id=doc_id,
            source_id=source_id,
            title=title,
            content=text,
            content_hash=Document.make_content_hash(text),
            section_tree=_scan_section_tree(text),
            language=language,
            version=1,
            created_at=now,
            updated_at=now,
        )
        self._store.upsert_document(doc)

        # ------------------------------------------------------------------
        # 3. ChunkMetadata template
        # ------------------------------------------------------------------
        metadata = ChunkMetadata(
            domain=domain,
            language=language,
            source_type=source_type,
            source_uri=source_uri,
            version=1,
            access_level=access_level,
            authority=authority,
            valid_from=now,
            valid_until=None,
            created_at=now,
            updated_at=now,
            title=title or None,
            topics=topics,
        )

        # ------------------------------------------------------------------
        # 4. Chunk
        # ------------------------------------------------------------------
        chunks: list[Chunk] = self._chunker.chunk(doc, metadata)

        # ------------------------------------------------------------------
        # 5. Upsert all chunks
        # ------------------------------------------------------------------
        for chunk in chunks:
            self._store.upsert_chunk(chunk)

        # ------------------------------------------------------------------
        # 6. Embed (also upserts vectors)
        # ------------------------------------------------------------------
        self._embedder.embed_chunks(chunks, self._store)

        return source_id, doc_id
