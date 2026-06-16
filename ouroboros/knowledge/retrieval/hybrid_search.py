"""Hybrid retrieval: dense vector search + BM25 sparse search + RRF fusion.

Pipeline for a single :class:`~ouroboros.knowledge.schemas.KBQuery`:

1. **Metadata pre-filter** — narrow candidates via SQLiteKBStore.filter_chunks().
2. **Dense search** — embed the query; rank candidates by cosine similarity.
3. **Sparse BM25 search** — rank candidates using the BM25 index.
4. **RRF fusion** — combine ranks via Reciprocal Rank Fusion (rrf_k=60 by default).
5. **Build results** — return RetrievedChunk objects, top-k_rerank, descending score.

No external dependencies — stdlib only (plus project-internal modules).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ouroboros.core.embedding import cosine, embed
from ouroboros.knowledge.retrieval.bm25 import BM25Index
from ouroboros.knowledge.schemas import KBQuery, RetrievedChunk
from ouroboros.knowledge.storage.sqlite_store import SQLiteKBStore

if TYPE_CHECKING:
    pass


class HybridSearch:
    """Hybrid retriever combining dense and sparse search with RRF fusion.

    Parameters
    ----------
    store:
        The :class:`~ouroboros.knowledge.storage.sqlite_store.SQLiteKBStore`
        instance to use for metadata filtering and chunk/vector retrieval.
    rrf_k:
        The RRF smoothing constant ``k`` in ``1/(k + rank)``. Larger values
        reduce the weight of high ranks. Defaults to 60 (standard literature
        value).
    """

    def __init__(self, store: SQLiteKBStore, rrf_k: int = 60) -> None:
        self._store = store
        self._rrf_k = rrf_k
        self._bm25 = BM25Index()
        # Track which chunk_ids are currently indexed (for incremental rebuilds)
        self._indexed_ids: list[str] = []

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self, chunk_ids: list[str]) -> None:
        """Load chunks from the store and build (or rebuild) the BM25 index.

        Only chunks that exist in the store are indexed.  Chunks with no text
        content are silently skipped.

        Parameters
        ----------
        chunk_ids:
            The set of chunk ids to include in the BM25 index.  Typically this
            is the full set of non-superseded, accessible chunks.
        """
        ids_to_index: list[str] = []
        texts: list[str] = []

        for cid in chunk_ids:
            chunk = self._store.get_chunk(cid)
            if chunk is None:
                continue
            text = chunk.content.strip()
            if not text:
                continue
            ids_to_index.append(cid)
            texts.append(text)

        self._bm25.index(ids_to_index, texts)
        self._indexed_ids = ids_to_index

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: KBQuery) -> list[RetrievedChunk]:
        """Run hybrid retrieval for *query*.

        Steps
        -----
        1. Metadata pre-filter via ``store.filter_chunks()``.
        2. Dense search: embed query → cosine similarity → top-``k_dense`` ranks.
        3. Sparse BM25 search restricted to the candidate set → top-``k_sparse``.
        4. RRF fusion: ``rrf_score = Σ 1/(rrf_k + rank)`` over sets the chunk
           appears in.  Take top ``k_rerank`` chunk ids.
        5. Build and return :class:`RetrievedChunk` objects, sorted descending.

        Parameters
        ----------
        query:
            The retrieval request, including text, filters, and ``k_*`` budget
            parameters.

        Returns
        -------
        list[RetrievedChunk]
            Up to ``query.k_rerank`` results, sorted by descending RRF score.
        """
        # ------------------------------------------------------------------
        # 1. Metadata pre-filter
        # ------------------------------------------------------------------
        candidate_ids: list[str] = self._store.filter_chunks(
            domain=query.domain,
            access_levels=query.access_levels,
            valid_now=query.valid_now,
        )

        if not candidate_ids:
            return []

        candidate_set: set[str] = set(candidate_ids)

        # ------------------------------------------------------------------
        # 2. Dense search (cosine similarity)
        # ------------------------------------------------------------------
        query_vec = embed(query.text)

        chunks_and_vecs = self._store.get_all_chunks_with_vectors(candidate_ids)

        # Sort candidates by cosine similarity descending
        dense_scored: list[tuple[str, float]] = []
        for chunk, vec in chunks_and_vecs:
            sim = cosine(query_vec, vec)
            dense_scored.append((chunk.id, sim))

        dense_scored.sort(key=lambda x: x[1], reverse=True)

        # Take top k_dense and record rank (1-based)
        dense_top = dense_scored[: query.k_dense]
        # chunk_id → dense rank
        dense_rank_map: dict[str, int] = {
            cid: rank + 1 for rank, (cid, _) in enumerate(dense_top)
        }

        # ------------------------------------------------------------------
        # 3. Sparse BM25 search (restricted to candidate set)
        # ------------------------------------------------------------------
        sparse_results = self._bm25.search(
            query.text,
            top_k=query.k_sparse,
            candidate_ids=candidate_set,
        )

        # chunk_id → sparse rank
        sparse_rank_map: dict[str, int] = {
            cid: rank + 1 for rank, (cid, _) in enumerate(sparse_results)
        }

        # ------------------------------------------------------------------
        # 4. RRF fusion
        # ------------------------------------------------------------------
        all_ids: set[str] = set(dense_rank_map) | set(sparse_rank_map)
        rrf_k = self._rrf_k

        rrf_scores: dict[str, float] = {}
        for cid in all_ids:
            score = 0.0
            if cid in dense_rank_map:
                score += 1.0 / (rrf_k + dense_rank_map[cid])
            if cid in sparse_rank_map:
                score += 1.0 / (rrf_k + sparse_rank_map[cid])
            rrf_scores[cid] = score

        # Sort descending and take top k_rerank
        fused_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_ids = [cid for cid, _ in fused_sorted[: query.k_rerank]]

        # ------------------------------------------------------------------
        # 5. Build RetrievedChunk objects
        # ------------------------------------------------------------------
        # We need the Chunk objects for the top ids.
        # get_all_chunks_with_vectors already fetched them, so build a map.
        chunk_map = {chunk.id: chunk for chunk, _ in chunks_and_vecs}

        results: list[RetrievedChunk] = []
        for cid in top_ids:
            chunk = chunk_map.get(cid)
            if chunk is None:
                # Chunk has no vector — try fetching it directly
                chunk = self._store.get_chunk(cid)
            if chunk is None:
                continue

            rrf_score = rrf_scores[cid]
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=rrf_score,
                    rrf_score=rrf_score,
                    dense_rank=dense_rank_map.get(cid),
                    sparse_rank=sparse_rank_map.get(cid),
                )
            )

        # Already sorted by descending rrf_score due to fused_sorted ordering
        return results
