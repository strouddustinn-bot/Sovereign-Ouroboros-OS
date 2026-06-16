"""Tests for the knowledge-base ingestion pipeline.

Covers:
- IngestPipeline.ingest() returns non-empty source_id and doc_id
- store.count_chunks() > 0 after ingestion
- Child chunks have parent_id set (small-to-big pattern)
- section_path is populated for Markdown-headed documents
- keywords contains expected words and excludes stop words / short words
- get_chunk_vector() returns a 64-dim tuple for an ingested chunk
- A document with multiple sections produces chunks with distinct section_paths
- Re-ingesting the same URI UPSERTs (count doesn't double)
"""

from __future__ import annotations

import pytest

from ouroboros.knowledge.ingestion import Chunker, Embedder, IngestPipeline
from ouroboros.knowledge.schemas import EMBEDDING_DIM, EMBEDDING_MODEL
from ouroboros.knowledge.storage.sqlite_store import SQLiteKBStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SIMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This sentence contains some common English words. "
    "Another sentence follows here with more interesting content. "
    "Knowledge retrieval systems rely on embeddings and indexing. "
    "The pipeline transforms raw text into structured knowledge. "
    "Each chunk carries metadata about its source and domain. "
    "Semantic search depends on high-quality vector representations. "
    "Chunking strategy affects both recall and precision significantly. "
    "Small-to-big retrieval expands context at generation time. "
    "Provenance tracking ensures every result can be traced back. "
) * 5  # ~500 words total — enough to produce multiple chunks

_MARKDOWN_TEXT = """\
# Billing

This section covers billing topics and payment methods.
Customers can pay with credit cards or bank transfers.
Invoices are issued monthly and sent by email.

## Refunds

Refund requests must be submitted within 30 days of purchase.
Approved refunds are processed within 5 business days.
Partial refunds are available for unused subscription periods.

## Payment Methods

We accept Visa, Mastercard, and PayPal.
Bank transfers are available for enterprise accounts.
All transactions are secured with TLS encryption.

# Support

Contact our support team for any billing or technical questions.
Support tickets are tracked in our internal system.
Response time is typically under 24 hours for standard queries.

## Contact

Email support at help@example.com for assistance.
Phone support is available Monday through Friday nine to five.
"""


@pytest.fixture()
def store():
    """Provide a fresh in-memory SQLiteKBStore for each test."""
    with SQLiteKBStore(":memory:") as s:
        yield s


@pytest.fixture()
def pipeline(store):
    """Provide an IngestPipeline wired to the in-memory store."""
    return IngestPipeline(store)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIngestReturnsIds:
    def test_returns_nonempty_strings(self, pipeline):
        source_id, doc_id = pipeline.ingest(
            text=_SIMPLE_TEXT,
            source_uri="https://example.com/doc1",
            domain="general",
        )
        assert isinstance(source_id, str) and source_id
        assert isinstance(doc_id, str) and doc_id

    def test_source_id_has_expected_prefix(self, pipeline):
        source_id, _ = pipeline.ingest(
            text=_SIMPLE_TEXT,
            source_uri="https://example.com/doc2",
            domain="general",
        )
        assert source_id.startswith("src_")

    def test_doc_id_has_expected_prefix(self, pipeline):
        _, doc_id = pipeline.ingest(
            text=_SIMPLE_TEXT,
            source_uri="https://example.com/doc3",
            domain="general",
        )
        assert doc_id.startswith("doc_")


class TestChunkCount:
    def test_chunks_are_stored(self, pipeline, store):
        pipeline.ingest(
            text=_SIMPLE_TEXT,
            source_uri="https://example.com/count-test",
            domain="general",
        )
        assert store.count_chunks() > 0

    def test_multiple_chunks_for_long_text(self, pipeline, store):
        pipeline.ingest(
            text=_SIMPLE_TEXT,
            source_uri="https://example.com/multi-chunk",
            domain="general",
        )
        # With a 500-word text and 150-word child chunks we expect at least 3
        assert store.count_chunks() >= 3


class TestSmallToBig:
    def test_child_chunks_have_parent_id(self, pipeline, store):
        pipeline.ingest(
            text=_SIMPLE_TEXT,
            source_uri="https://example.com/parent-test",
            domain="general",
        )
        all_ids = store.filter_chunks()
        children_with_parent = []
        for cid in all_ids:
            chunk = store.get_chunk(cid)
            if chunk and chunk.parent_id is not None:
                children_with_parent.append(chunk)
        assert len(children_with_parent) > 0, "Expected child chunks with parent_id set"

    def test_parent_id_references_existing_chunk(self, pipeline, store):
        pipeline.ingest(
            text=_SIMPLE_TEXT,
            source_uri="https://example.com/parent-ref",
            domain="general",
        )
        all_ids = store.filter_chunks()
        # Collect all chunk ids including parents (parents have superseded_by=None too)
        conn = store._conn
        all_chunk_ids = {
            r["id"] for r in conn.execute("SELECT id FROM chunks").fetchall()
        }
        for cid in all_ids:
            chunk = store.get_chunk(cid)
            if chunk and chunk.parent_id is not None:
                assert chunk.parent_id in all_chunk_ids, (
                    f"Child {cid} references missing parent {chunk.parent_id}"
                )


class TestSectionPath:
    def test_section_path_populated_for_markdown(self, pipeline, store):
        pipeline.ingest(
            text=_MARKDOWN_TEXT,
            source_uri="https://example.com/markdown-test",
            domain="billing",
        )
        all_ids = store.filter_chunks()
        chunks_with_path = [
            store.get_chunk(cid)
            for cid in all_ids
        ]
        paths = [c.section_path for c in chunks_with_path if c and c.section_path]
        assert len(paths) > 0, "Expected at least one chunk with a non-empty section_path"

    def test_billing_section_path_contains_billing(self, pipeline, store):
        pipeline.ingest(
            text=_MARKDOWN_TEXT,
            source_uri="https://example.com/billing-path",
            domain="billing",
        )
        all_ids = store.filter_chunks()
        billing_paths = []
        for cid in all_ids:
            chunk = store.get_chunk(cid)
            if chunk and "Billing" in chunk.section_path:
                billing_paths.append(chunk.section_path)
        assert len(billing_paths) > 0, "Expected chunks with 'Billing' in section_path"

    def test_subsection_path_depth(self, pipeline, store):
        pipeline.ingest(
            text=_MARKDOWN_TEXT,
            source_uri="https://example.com/depth-test",
            domain="billing",
        )
        all_ids = store.filter_chunks()
        deep_paths = []
        for cid in all_ids:
            chunk = store.get_chunk(cid)
            if chunk and len(chunk.section_path) >= 2:
                deep_paths.append(chunk.section_path)
        assert len(deep_paths) > 0, (
            "Expected chunks with section_path depth >= 2 (e.g. ['Billing', 'Refunds'])"
        )

    def test_distinct_section_paths(self, pipeline, store):
        pipeline.ingest(
            text=_MARKDOWN_TEXT,
            source_uri="https://example.com/distinct-sections",
            domain="billing",
        )
        all_ids = store.filter_chunks()
        all_paths = set()
        for cid in all_ids:
            chunk = store.get_chunk(cid)
            if chunk and chunk.section_path:
                all_paths.add(tuple(chunk.section_path))
        assert len(all_paths) > 1, "Expected chunks from multiple distinct sections"


class TestKeywords:
    def test_keywords_populated(self, pipeline, store):
        pipeline.ingest(
            text=_SIMPLE_TEXT,
            source_uri="https://example.com/keywords-test",
            domain="general",
        )
        all_ids = store.filter_chunks()
        chunks_with_keywords = [
            store.get_chunk(cid)
            for cid in all_ids
            if store.get_chunk(cid) and store.get_chunk(cid).keywords  # type: ignore[union-attr]
        ]
        assert len(chunks_with_keywords) > 0

    def test_keywords_exclude_stop_words(self, pipeline, store):
        pipeline.ingest(
            text=_SIMPLE_TEXT,
            source_uri="https://example.com/stopwords-test",
            domain="general",
        )
        stop_words = {"the", "and", "for", "that", "with"}
        all_ids = store.filter_chunks()
        for cid in all_ids:
            chunk = store.get_chunk(cid)
            if chunk:
                for kw in chunk.keywords:
                    assert kw not in stop_words, (
                        f"Stop word '{kw}' found in chunk keywords"
                    )

    def test_keywords_exclude_short_words(self, pipeline, store):
        pipeline.ingest(
            text=_SIMPLE_TEXT,
            source_uri="https://example.com/short-words-test",
            domain="general",
        )
        all_ids = store.filter_chunks()
        for cid in all_ids:
            chunk = store.get_chunk(cid)
            if chunk:
                for kw in chunk.keywords:
                    assert len(kw) > 3, (
                        f"Short word '{kw}' (len={len(kw)}) found in keywords"
                    )

    def test_keywords_contain_expected_word(self, pipeline, store):
        # "knowledge" appears in _SIMPLE_TEXT and is long + non-stop
        pipeline.ingest(
            text=_SIMPLE_TEXT,
            source_uri="https://example.com/expected-kw",
            domain="general",
        )
        all_ids = store.filter_chunks()
        found = False
        for cid in all_ids:
            chunk = store.get_chunk(cid)
            if chunk and "knowledge" in chunk.keywords:
                found = True
                break
        assert found, "Expected 'knowledge' to appear in at least one chunk's keywords"


class TestVectors:
    def test_vector_returned_for_ingested_chunk(self, pipeline, store):
        pipeline.ingest(
            text=_SIMPLE_TEXT,
            source_uri="https://example.com/vector-test",
            domain="general",
        )
        all_ids = store.filter_chunks()
        assert all_ids, "No chunks found after ingestion"
        # Pick the first child chunk (has parent_id)
        for cid in all_ids:
            chunk = store.get_chunk(cid)
            if chunk and chunk.parent_id is not None:
                vector = store.get_chunk_vector(cid)
                assert vector is not None, f"No vector found for chunk {cid}"
                assert len(vector) == EMBEDDING_DIM, (
                    f"Expected {EMBEDDING_DIM}-dim vector, got {len(vector)}"
                )
                break

    def test_vector_is_64_dimensional(self, pipeline, store):
        pipeline.ingest(
            text=_SIMPLE_TEXT,
            source_uri="https://example.com/dim-test",
            domain="general",
        )
        all_ids = store.filter_chunks()
        for cid in all_ids[:3]:  # spot-check first few
            v = store.get_chunk_vector(cid)
            if v is not None:
                assert len(v) == 64


class TestUpsert:
    def test_reingest_same_uri_does_not_double_count(self, pipeline, store):
        uri = "https://example.com/upsert-test"
        pipeline.ingest(
            text=_SIMPLE_TEXT,
            source_uri=uri,
            domain="general",
        )
        count_after_first = store.count_chunks()
        assert count_after_first > 0

        # Re-ingest the exact same text and URI
        pipeline.ingest(
            text=_SIMPLE_TEXT,
            source_uri=uri,
            domain="general",
        )
        count_after_second = store.count_chunks()
        # UPSERT semantics: count should not double
        assert count_after_second == count_after_first, (
            f"Chunk count doubled on re-ingest: {count_after_first} → {count_after_second}"
        )

    def test_different_uris_produce_separate_records(self, pipeline, store):
        pipeline.ingest(
            text=_SIMPLE_TEXT,
            source_uri="https://example.com/uri-a",
            domain="general",
        )
        count_a = store.count_chunks()
        pipeline.ingest(
            text=_SIMPLE_TEXT,
            source_uri="https://example.com/uri-b",
            domain="general",
        )
        count_b = store.count_chunks()
        assert count_b > count_a, "Different URIs should produce additional chunks"
