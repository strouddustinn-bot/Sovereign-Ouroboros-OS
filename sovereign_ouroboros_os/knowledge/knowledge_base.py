"""Top-level KnowledgeBase: ingestion + retrieval + citations in one object.

This is the external interface callers use.  The layered internals
(SQLiteKBStore, IngestPipeline, HybridSearch, ContextAssembler) are hidden
behind this thin facade.

Design follows the build-order in knowledge-base-architecture.md:
  1. SQLite + chunk schema  →  done (storage/)
  2. Structure-aware chunker →  done (ingestion/)
  3. CharNGram embed + dense retrieval  →  done (retrieval/)
  4. Dense-only retrieval end-to-end  →  THIS FILE wires it up
  5. BM25 + RRF  →  done (retrieval/hybrid_search.py)
  6. Context assembly + citations  →  done (retrieval/context_assembler.py)

Integration point with the Ouroboros loop:
  Pass a ``KnowledgeBase`` to ``OuroborosLoop(knowledge_base=kb)`` and
  before each imagination stage the loop will recall relevant context
  and pass it to NeuroSynth as background knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sovereign_ouroboros_os.knowledge.ingestion.pipeline import IngestPipeline
from sovereign_ouroboros_os.knowledge.retrieval.context_assembler import (
    ContextAssembler,
)
from sovereign_ouroboros_os.knowledge.retrieval.hybrid_search import HybridSearch
from sovereign_ouroboros_os.knowledge.schemas import KBQuery, RetrievedChunk
from sovereign_ouroboros_os.knowledge.storage.sqlite_store import SQLiteKBStore


@dataclass
class KnowledgeBase:
    """Unified knowledge base: ingest documents, query with hybrid retrieval.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Use ``":memory:"`` for ephemeral
        in-process storage (tests).  Defaults to ``"./kb.db"``.
    max_context_tokens:
        Maximum word-count budget when assembling retrieved passages into a
        context string for the agent.
    rrf_k:
        Reciprocal Rank Fusion constant (default 60, standard value).
    """

    db_path: str = "./kb.db"
    max_context_tokens: int = 1500
    rrf_k: int = 60

    _store: SQLiteKBStore = field(init=False)
    _pipeline: IngestPipeline = field(init=False)
    _search: HybridSearch = field(init=False)
    _assembler: ContextAssembler = field(init=False)

    def __post_init__(self) -> None:
        self._store = SQLiteKBStore(self.db_path)
        self._pipeline = IngestPipeline(self._store)
        self._search = HybridSearch(self._store, rrf_k=self.rrf_k)
        self._assembler = ContextAssembler(
            self._store, max_tokens=self.max_context_tokens
        )

    # ------------------------------------------------------------------
    # Ingestion
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
        """Ingest *text* and index it for retrieval.

        Idempotent: re-ingesting the same *source_uri* UPSERTs without
        duplicating.

        Returns
        -------
        tuple[str, str]
            ``(source_id, document_id)``
        """
        ids = self._pipeline.ingest(
            text=text,
            source_uri=source_uri,
            domain=domain,
            source_type=source_type,
            title=title,
            language=language,
            access_level=access_level,
            authority=authority,
            topics=topics,
        )
        # Rebuild the BM25 index after every ingestion so it stays fresh.
        self._rebuild_index()
        return ids

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        domain: str | None = None,
        access_level: list[str] | None = None,
        valid_now: bool = True,
        k_dense: int = 10,
        k_sparse: int = 10,
        k_rerank: int = 5,
        expand_to_parent: bool = True,
    ) -> list[RetrievedChunk]:
        """Hybrid retrieval: metadata filter → dense + BM25 → RRF → top-k.

        Parameters
        ----------
        text:
            Natural language query.
        domain:
            Optional domain filter (controlled vocab from taxonomy).
        access_level:
            List of allowed access levels; defaults to all levels.
        valid_now:
            Filter out chunks whose ``valid_until`` is in the past.
        k_dense, k_sparse:
            Candidate counts for the dense and sparse search legs.
        k_rerank:
            Final number of chunks after RRF fusion.
        expand_to_parent:
            Swap small child chunks for their larger parent context chunk.

        Returns
        -------
        list[RetrievedChunk]
            Ranked results, best first.
        """
        filters: dict[str, Any] = {"valid_now": valid_now}
        if domain:
            filters["domain"] = domain
        if access_level:
            filters["access_level"] = access_level

        q = KBQuery(
            text=text,
            filters=filters,
            k_dense=k_dense,
            k_sparse=k_sparse,
            k_rerank=k_rerank,
            expand_to_parent=expand_to_parent,
        )
        return self._search.search(q)

    def assemble_context(
        self,
        text: str,
        expand_to_parent: bool = True,
        **query_kwargs: Any,
    ) -> tuple[str, list[dict]]:
        """Retrieve and assemble a context string with citations.

        Returns
        -------
        tuple[str, list[dict]]
            ``(context_text, citations)`` where *citations* is a list of
            ``{"source_uri", "title", "section_path", "version", "chunk_id"}``
            dicts for each passage included in the context.
        """
        hits = self.query(text, expand_to_parent=expand_to_parent, **query_kwargs)
        return self._assembler.assemble(hits, expand_to_parent=expand_to_parent)

    # ------------------------------------------------------------------
    # Versioning
    # ------------------------------------------------------------------

    def supersede_chunk(self, old_chunk_id: str, new_content: str, **ingest_kwargs: Any) -> str:
        """Version a chunk: create a successor and mark the old one superseded.

        Returns the new chunk's id.
        """
        old = self._store.get_chunk(old_chunk_id)
        if old is None:
            raise KeyError(f"chunk {old_chunk_id!r} not found")

        # Re-ingest produces a new chunk (different content_hash → new id).
        source_uri = old.metadata.source_uri + f"#supersedes={old_chunk_id}"
        _, doc_id = self.ingest(
            text=new_content,
            source_uri=source_uri,
            domain=old.metadata.domain,
            title=(old.metadata.title or ""),
            **ingest_kwargs,
        )
        # Find the new chunk by document and mark the old one superseded.
        new_chunks = [
            cid
            for cid in self._store.filter_chunks()
            if self._store.get_chunk(cid) is not None
            and self._store.get_chunk(cid).document_id == doc_id  # type: ignore[union-attr]
        ]
        if new_chunks:
            new_id = new_chunks[0]
            self._store.supersede_chunk(old_chunk_id, new_id)
            return new_id
        return old_chunk_id

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def chunk_count(self) -> int:
        """Number of active (non-superseded) chunks in the knowledge base."""
        return self._store.count_chunks()

    def close(self) -> None:
        """Release the underlying SQLite connection."""
        self._store.close()

    def __enter__(self) -> "KnowledgeBase":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index from all active chunk ids."""
        active_ids = self._store.filter_chunks()
        self._search.build_index(active_ids)
