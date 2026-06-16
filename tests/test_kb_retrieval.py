"""Tests for the Ouroboros knowledge-base retrieval pipeline.

Covers:
- BM25Index: basic indexing and scoring
- HybridSearch: end-to-end retrieval with metadata filtering, RRF fusion
- ContextAssembler: passage formatting, dedup, token budget, parent expansion

Chunks are created directly via SQLiteKBStore (no ingestion pipeline).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from sovereign_ouroboros_os.core.embedding import embed
from sovereign_ouroboros_os.knowledge.retrieval import (
    BM25Index,
    ContextAssembler,
    HybridSearch,
)
from sovereign_ouroboros_os.knowledge.schemas import (
    EMBEDDING_MODEL,
    Chunk,
    ChunkMetadata,
    KBQuery,
)
from sovereign_ouroboros_os.knowledge.storage.sqlite_store import SQLiteKBStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime.now(timezone.utc).isoformat()


def _make_metadata(
    domain: str = "test",
    title: str | None = None,
    access_level: str = "public",
    source_uri: str = "test://doc",
    version: int = 1,
) -> ChunkMetadata:
    return ChunkMetadata(
        domain=domain,
        language="en",
        source_type="doc",
        source_uri=source_uri,
        version=version,
        access_level=access_level,
        authority=0.9,
        created_at=NOW,
        updated_at=NOW,
        title=title,
    )


def _make_chunk(
    content: str,
    chunk_index: int = 0,
    domain: str = "test",
    section_path: list[str] | None = None,
    parent_id: str | None = None,
    access_level: str = "public",
    title: str | None = None,
    source_uri: str = "test://doc",
) -> Chunk:
    content_hash = Chunk.make_content_hash(content)
    source_id = "src_test"
    cid = Chunk.make_id(source_id, chunk_index, content_hash)
    vec = embed(content)
    md = _make_metadata(
        domain=domain,
        title=title,
        access_level=access_level,
        source_uri=source_uri,
    )
    return Chunk(
        id=cid,
        document_id="doc_test",
        source_id=source_id,
        content=content,
        content_hash=content_hash,
        chunk_index=chunk_index,
        section_path=section_path or [],
        tokens=len(content.split()),
        vector=vec,
        vector_model=EMBEDDING_MODEL,
        vector_dim=len(vec),
        keywords=content.lower().split(),
        entities=[],
        metadata=md,
        parent_id=parent_id,
    )


def _ingest(store: SQLiteKBStore, chunk: Chunk) -> None:
    """Insert a chunk and its vector into the store."""
    store.upsert_chunk(chunk)
    store.upsert_vector(chunk.id, chunk.vector)


# ---------------------------------------------------------------------------
# Shared fixture: small KB with 3+ domains
# ---------------------------------------------------------------------------

BILLING_CONTENT = (
    "Billing and invoices are processed monthly. "
    "Refunds can be requested within 30 days of the invoice date. "
    "Contact billing support for payment issues."
)

TECH_CONTENT = (
    "The API rate limit is 100 requests per minute. "
    "Use exponential backoff when you receive a 429 Too Many Requests error. "
    "Authentication uses OAuth2 bearer tokens."
)

LEGAL_CONTENT = (
    "Users must agree to the terms of service before accessing the platform. "
    "Data is stored in compliance with GDPR regulations. "
    "Privacy policy updates are communicated via email."
)

BILLING_CONTENT_2 = (
    "Invoice payment methods include credit card, wire transfer, and ACH. "
    "Late payments incur a 1.5 percent monthly fee."
)


@pytest.fixture()
def store_with_chunks() -> SQLiteKBStore:  # type: ignore[misc]
    """In-memory store pre-populated with multi-domain chunks."""
    store = SQLiteKBStore(":memory:")

    billing_chunk = _make_chunk(
        BILLING_CONTENT,
        chunk_index=0,
        domain="billing",
        section_path=["Billing", "Refunds"],
        title="Billing Overview",
        source_uri="kb://billing/overview",
    )
    billing_chunk2 = _make_chunk(
        BILLING_CONTENT_2,
        chunk_index=1,
        domain="billing",
        section_path=["Billing", "Payment Methods"],
        title="Payment Methods",
        source_uri="kb://billing/payments",
    )
    tech_chunk = _make_chunk(
        TECH_CONTENT,
        chunk_index=2,
        domain="technology",
        section_path=["API", "Rate Limits"],
        title="API Rate Limiting",
        source_uri="kb://tech/api",
    )
    legal_chunk = _make_chunk(
        LEGAL_CONTENT,
        chunk_index=3,
        domain="legal",
        section_path=["Legal", "Privacy"],
        title="Privacy Policy",
        source_uri="kb://legal/privacy",
    )

    for chunk in [billing_chunk, billing_chunk2, tech_chunk, legal_chunk]:
        _ingest(store, chunk)

    return store


# ---------------------------------------------------------------------------
# BM25Index tests
# ---------------------------------------------------------------------------


class TestBM25Index:
    def test_basic_scoring(self) -> None:
        idx = BM25Index()
        idx.index(
            ["c1", "c2", "c3"],
            [
                "refund invoice billing payment",
                "api rate limit requests authentication",
                "privacy terms service data",
            ],
        )
        results = idx.search("invoice refund billing")
        assert len(results) > 0
        top_id, top_score = results[0]
        assert top_id == "c1"
        assert top_score > 0

    def test_empty_index_returns_empty(self) -> None:
        idx = BM25Index()
        assert idx.search("anything") == []

    def test_query_no_match_returns_empty(self) -> None:
        idx = BM25Index()
        idx.index(["c1"], ["hello world this is a document"])
        results = idx.search("zzzzzznotaword")
        assert results == []

    def test_top_k_respected(self) -> None:
        idx = BM25Index()
        ids = [f"c{i}" for i in range(10)]
        texts = [f"word{i} shared common token" for i in range(10)]
        idx.index(ids, texts)
        results = idx.search("shared common", top_k=3)
        assert len(results) <= 3

    def test_candidate_filter(self) -> None:
        idx = BM25Index()
        idx.index(
            ["c1", "c2", "c3"],
            ["billing refund payment", "billing invoice", "rate limit api"],
        )
        # Restrict to only c3 — even though c1/c2 match "billing" better
        results = idx.search("billing payment", candidate_ids={"c3"})
        # c3 contains "rate limit api" so it may not score but it's the only
        # candidate — if it does appear, it must be c3
        for cid, _ in results:
            assert cid == "c3"

    def test_short_tokens_filtered(self) -> None:
        idx = BM25Index()
        # Only single-character tokens — all filtered by the >= 2 char rule
        idx.index(["c1"], ["a b c d e f"])
        # All tokens are 1 char, so nothing is indexed
        results = idx.search("a b c")
        assert results == []

    def test_scores_are_positive_for_matches(self) -> None:
        idx = BM25Index()
        idx.index(["c1", "c2"], ["database query sql select", "frontend css react"])
        results = idx.search("database query")
        assert all(score > 0 for _, score in results)

    def test_rebuild_clears_old_index(self) -> None:
        idx = BM25Index()
        idx.index(["c1"], ["old content here"])
        idx.index(["c2"], ["new content entirely different"])
        results = idx.search("old content")
        ids = [cid for cid, _ in results]
        assert "c1" not in ids  # c1 was removed on rebuild


# ---------------------------------------------------------------------------
# HybridSearch tests
# ---------------------------------------------------------------------------


class TestHybridSearch:
    def _build_hybrid(self, store: SQLiteKBStore) -> HybridSearch:
        hs = HybridSearch(store)
        all_ids = store.filter_chunks()
        hs.build_index(all_ids)
        return hs

    def test_returns_results(self, store_with_chunks: SQLiteKBStore) -> None:
        hs = self._build_hybrid(store_with_chunks)
        q = KBQuery(text="billing refund invoice payment", k_rerank=5)
        results = hs.search(q)
        assert len(results) > 0

    def test_results_are_retrieved_chunks(
        self, store_with_chunks: SQLiteKBStore
    ) -> None:
        from sovereign_ouroboros_os.knowledge.schemas import RetrievedChunk

        hs = self._build_hybrid(store_with_chunks)
        q = KBQuery(text="billing payment refund", k_rerank=3)
        results = hs.search(q)
        for rc in results:
            assert isinstance(rc, RetrievedChunk)
            assert rc.score > 0

    def test_most_relevant_chunk_ranked_first(
        self, store_with_chunks: SQLiteKBStore
    ) -> None:
        """The billing chunk should rank highest for a billing query."""
        hs = self._build_hybrid(store_with_chunks)
        q = KBQuery(text="refund invoice billing payment monthly", k_rerank=5)
        results = hs.search(q)
        assert len(results) > 0
        top_chunk = results[0].chunk
        assert top_chunk.metadata.domain == "billing"

    def test_metadata_domain_filter(self, store_with_chunks: SQLiteKBStore) -> None:
        """Filtering by domain=legal must return only legal chunks."""
        hs = self._build_hybrid(store_with_chunks)
        q = KBQuery(
            text="data privacy terms",
            filters={"domain": "legal"},
            k_rerank=10,
        )
        results = hs.search(q)
        assert len(results) > 0
        for rc in results:
            assert rc.chunk.metadata.domain == "legal"

    def test_domain_filter_excludes_other_domains(
        self, store_with_chunks: SQLiteKBStore
    ) -> None:
        hs = self._build_hybrid(store_with_chunks)
        q = KBQuery(
            text="billing invoice payment",
            filters={"domain": "technology"},
            k_rerank=10,
        )
        results = hs.search(q)
        for rc in results:
            assert rc.chunk.metadata.domain == "technology"

    def test_empty_candidates_returns_empty(self) -> None:
        store = SQLiteKBStore(":memory:")
        hs = HybridSearch(store)
        q = KBQuery(text="anything", filters={"domain": "nonexistent"})
        assert hs.search(q) == []

    def test_rrf_both_sets_score_higher(
        self, store_with_chunks: SQLiteKBStore
    ) -> None:
        """Chunks in both dense and sparse top-k should have higher RRF scores.

        We construct a query that closely matches the billing chunk both
        semantically (dense) and lexically (sparse).  The billing chunk should
        appear in both rank maps, giving it a higher RRF score than a chunk
        that only appears in one.
        """
        hs = self._build_hybrid(store_with_chunks)
        q = KBQuery(
            text="billing refund invoice payment monthly",
            k_dense=10,
            k_sparse=10,
            k_rerank=10,
        )
        results = hs.search(q)

        # Find a chunk that appears in both rank maps
        both_set = [
            rc for rc in results
            if rc.dense_rank is not None and rc.sparse_rank is not None
        ]
        one_set = [
            rc for rc in results
            if (rc.dense_rank is None) != (rc.sparse_rank is None)
        ]

        if both_set and one_set:
            max_both = max(rc.rrf_score for rc in both_set)
            max_one = max(rc.rrf_score for rc in one_set)
            assert max_both > max_one, (
                f"Chunks in both sets should score higher. "
                f"both={max_both:.4f}, one_set={max_one:.4f}"
            )

    def test_results_sorted_descending(
        self, store_with_chunks: SQLiteKBStore
    ) -> None:
        hs = self._build_hybrid(store_with_chunks)
        q = KBQuery(text="billing api privacy", k_rerank=5)
        results = hs.search(q)
        scores = [rc.score for rc in results]
        assert scores == sorted(scores, reverse=True)

    def test_access_level_filter(self) -> None:
        """Restricted chunks must not appear in public-only queries."""
        store = SQLiteKBStore(":memory:")
        public_chunk = _make_chunk(
            "public information about our services",
            chunk_index=0,
            access_level="public",
        )
        restricted_chunk = _make_chunk(
            "restricted confidential internal data",
            chunk_index=1,
            access_level="restricted",
        )
        _ingest(store, public_chunk)
        _ingest(store, restricted_chunk)

        hs = HybridSearch(store)
        hs.build_index(store.filter_chunks())

        q = KBQuery(
            text="information services",
            filters={"access_level": "public"},
            k_rerank=10,
        )
        results = hs.search(q)
        for rc in results:
            assert rc.chunk.metadata.access_level == "public"


# ---------------------------------------------------------------------------
# ContextAssembler tests
# ---------------------------------------------------------------------------


class TestContextAssembler:
    def _make_hits(
        self, store: SQLiteKBStore
    ) -> list:
        from sovereign_ouroboros_os.knowledge.schemas import RetrievedChunk

        chunks_and_vecs = store.get_all_chunks_with_vectors(store.filter_chunks())
        hits = []
        for i, (chunk, _) in enumerate(chunks_and_vecs):
            hits.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=1.0 - i * 0.1,
                    rrf_score=1.0 - i * 0.1,
                )
            )
        return hits

    def test_returns_nonempty_context(
        self, store_with_chunks: SQLiteKBStore
    ) -> None:
        assembler = ContextAssembler(store_with_chunks)
        hits = self._make_hits(store_with_chunks)
        context, citations = assembler.assemble(hits, expand_to_parent=False)
        assert len(context) > 0

    def test_returns_citations(self, store_with_chunks: SQLiteKBStore) -> None:
        assembler = ContextAssembler(store_with_chunks)
        hits = self._make_hits(store_with_chunks)
        _, citations = assembler.assemble(hits, expand_to_parent=False)
        assert len(citations) >= 1
        for cite in citations:
            assert "source_uri" in cite
            assert "title" in cite
            assert "section_path" in cite
            assert "version" in cite
            assert "chunk_id" in cite

    def test_token_budget_respected(self) -> None:
        """With max_tokens=50, fewer than ~55 words should be returned."""
        store = SQLiteKBStore(":memory:")
        # Create a short chunk and a long chunk
        short_chunk = _make_chunk(
            "Short text here only ten words or so total.",
            chunk_index=0,
        )
        long_chunk = _make_chunk(
            " ".join(["word"] * 200),  # 200 words
            chunk_index=1,
        )
        _ingest(store, short_chunk)
        _ingest(store, long_chunk)

        from sovereign_ouroboros_os.knowledge.schemas import RetrievedChunk

        # short chunk ranked first (higher score)
        hits = [
            RetrievedChunk(chunk=short_chunk, score=0.9, rrf_score=0.9),
            RetrievedChunk(chunk=long_chunk, score=0.5, rrf_score=0.5),
        ]

        assembler = ContextAssembler(store, max_tokens=50)
        context, citations = assembler.assemble(hits, expand_to_parent=False)

        word_count = len(context.split())
        # Should be well under 55 words (short chunk is ~10 words + formatting)
        assert word_count < 55, f"Expected <55 words but got {word_count}"
        # And the short chunk should be present
        assert "Short text" in context

    def test_token_budget_at_least_one_chunk(self) -> None:
        """Even with max_tokens=5, at least one chunk must be included."""
        store = SQLiteKBStore(":memory:")
        chunk = _make_chunk("This is a moderately long chunk with many words here.", 0)
        _ingest(store, chunk)

        from sovereign_ouroboros_os.knowledge.schemas import RetrievedChunk

        hits = [RetrievedChunk(chunk=chunk, score=1.0, rrf_score=1.0)]
        assembler = ContextAssembler(store, max_tokens=5)
        context, citations = assembler.assemble(hits, expand_to_parent=False)
        assert len(context) > 0
        assert len(citations) == 1

    def test_dedup_by_content_hash(self) -> None:
        """Duplicate content hashes should be included only once."""
        store = SQLiteKBStore(":memory:")
        content = "identical content for dedup test"
        chunk_a = _make_chunk(content, chunk_index=0)
        # chunk_b has same content → same content_hash
        chunk_b_content = content  # same text
        content_hash_b = Chunk.make_content_hash(chunk_b_content)
        # Build a distinct id by tweaking source_id
        cid_b = hashlib.sha256(f"src_alt:1:{content_hash_b}".encode()).hexdigest()[:16]
        md_b = _make_metadata()
        vec_b = embed(chunk_b_content)
        chunk_b = Chunk(
            id=cid_b,
            document_id="doc_test",
            source_id="src_alt",
            content=chunk_b_content,
            content_hash=content_hash_b,
            chunk_index=1,
            section_path=[],
            tokens=len(chunk_b_content.split()),
            vector=vec_b,
            vector_model=EMBEDDING_MODEL,
            vector_dim=len(vec_b),
            keywords=[],
            entities=[],
            metadata=md_b,
        )

        _ingest(store, chunk_a)
        _ingest(store, chunk_b)

        from sovereign_ouroboros_os.knowledge.schemas import RetrievedChunk

        hits = [
            RetrievedChunk(chunk=chunk_a, score=0.9, rrf_score=0.9),
            RetrievedChunk(chunk=chunk_b, score=0.8, rrf_score=0.8),
        ]
        assembler = ContextAssembler(store, max_tokens=2000)
        context, citations = assembler.assemble(hits, expand_to_parent=False)
        # Only one passage — the duplicate should be dropped
        assert citations[0]["chunk_id"] == chunk_a.id
        assert len(citations) == 1

    def test_parent_expansion(self) -> None:
        """Expanding to parent should produce a longer passage for child chunks."""
        store = SQLiteKBStore(":memory:")

        parent_content = (
            "Parent section: This is a long comprehensive parent document "
            "covering multiple topics in detail. It contains the full context "
            "needed to understand the child chunks within it."
        )
        child_content = "Child: short excerpt."

        parent_chunk = _make_chunk(parent_content, chunk_index=0, section_path=["Parent"])
        _ingest(store, parent_chunk)

        # Build child with parent_id pointing to parent
        child_content_hash = Chunk.make_content_hash(child_content)
        child_id = Chunk.make_id("src_test", 1, child_content_hash)
        child_md = _make_metadata()
        child_vec = embed(child_content)
        child_chunk = Chunk(
            id=child_id,
            document_id="doc_test",
            source_id="src_test",
            content=child_content,
            content_hash=child_content_hash,
            chunk_index=1,
            section_path=["Parent", "Child"],
            tokens=len(child_content.split()),
            vector=child_vec,
            vector_model=EMBEDDING_MODEL,
            vector_dim=len(child_vec),
            keywords=[],
            entities=[],
            metadata=child_md,
            parent_id=parent_chunk.id,
        )
        _ingest(store, child_chunk)

        from sovereign_ouroboros_os.knowledge.schemas import RetrievedChunk

        hits = [RetrievedChunk(chunk=child_chunk, score=1.0, rrf_score=1.0)]

        assembler = ContextAssembler(store, max_tokens=2000)

        # Without expansion: child content only
        context_no_expand, _ = assembler.assemble(hits, expand_to_parent=False)
        # With expansion: parent content
        context_with_expand, citations = assembler.assemble(hits, expand_to_parent=True)

        assert len(context_with_expand) > len(context_no_expand), (
            "Expanded context should be longer than child-only context"
        )
        assert parent_content in context_with_expand

    def test_passage_format_has_section_path(self) -> None:
        """Formatted passages should include section path in the header."""
        store = SQLiteKBStore(":memory:")
        chunk = _make_chunk(
            "Content about billing refunds.",
            chunk_index=0,
            section_path=["Billing", "Refunds"],
        )
        _ingest(store, chunk)

        from sovereign_ouroboros_os.knowledge.schemas import RetrievedChunk

        hits = [RetrievedChunk(chunk=chunk, score=1.0, rrf_score=1.0)]
        assembler = ContextAssembler(store, max_tokens=2000)
        context, _ = assembler.assemble(hits, expand_to_parent=False)

        assert "Billing" in context
        assert "Refunds" in context
        assert "[1]" in context

    def test_empty_hits_returns_empty(self, store_with_chunks: SQLiteKBStore) -> None:
        assembler = ContextAssembler(store_with_chunks)
        context, citations = assembler.assemble([], expand_to_parent=False)
        assert context == ""
        assert citations == []

    def test_citations_have_correct_source_uri(
        self, store_with_chunks: SQLiteKBStore
    ) -> None:
        assembler = ContextAssembler(store_with_chunks)
        hits = self._make_hits(store_with_chunks)
        _, citations = assembler.assemble(hits, expand_to_parent=False)
        for cite in citations:
            assert cite["source_uri"].startswith("kb://")

    def test_passages_separated_by_delimiter(
        self, store_with_chunks: SQLiteKBStore
    ) -> None:
        """Multiple passages must be separated by the expected delimiter."""
        assembler = ContextAssembler(store_with_chunks, max_tokens=5000)
        hits = self._make_hits(store_with_chunks)
        context, citations = assembler.assemble(hits, expand_to_parent=False)
        if len(citations) > 1:
            assert "\n\n---\n\n" in context


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_hybrid_then_assemble(self, store_with_chunks: SQLiteKBStore) -> None:
        """Run HybridSearch then ContextAssembler end-to-end."""
        all_ids = store_with_chunks.filter_chunks()
        hs = HybridSearch(store_with_chunks)
        hs.build_index(all_ids)

        q = KBQuery(text="refund billing invoice payment", k_rerank=5)
        results = hs.search(q)
        assert len(results) > 0

        assembler = ContextAssembler(store_with_chunks, max_tokens=2000)
        context, citations = assembler.assemble(results, expand_to_parent=False)

        assert len(context) > 0
        assert len(citations) > 0
        # Top result should be billing-related
        assert any("billing" in c["source_uri"].lower() for c in citations)
