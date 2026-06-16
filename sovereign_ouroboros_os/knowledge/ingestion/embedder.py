"""Batch embedder for knowledge-base chunks.

Wraps the core :func:`~sovereign_ouroboros_os.core.embedding.embed` function
and drives it over a list of :class:`~sovereign_ouroboros_os.knowledge.schemas.Chunk`
objects, persisting each vector into the store via
:meth:`~sovereign_ouroboros_os.knowledge.storage.sqlite_store.SQLiteKBStore.upsert_vector`.
"""

from __future__ import annotations

from sovereign_ouroboros_os.core.embedding import embed
from sovereign_ouroboros_os.knowledge.schemas import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    Chunk,
)
from sovereign_ouroboros_os.knowledge.storage.sqlite_store import SQLiteKBStore


class Embedder:
    """Batch-embed chunks and persist their vectors into the KB store.

    The embedder uses the best available embedding backend (API → CharNGram →
    Hash) as selected by :func:`~sovereign_ouroboros_os.core.embedding.embed`.
    It mutates each chunk's ``vector`` field in-place (``Chunk`` is a plain
    ``@dataclass`` and therefore mutable) and calls
    :meth:`SQLiteKBStore.upsert_vector` so the vector is durable immediately.
    """

    def __init__(self) -> None:
        self._model = EMBEDDING_MODEL
        self._dim = EMBEDDING_DIM

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(
        self, chunks: list[Chunk], store: SQLiteKBStore
    ) -> list[Chunk]:
        """Embed each chunk's content and upsert its vector into *store*.

        Parameters
        ----------
        chunks:
            List of :class:`Chunk` objects to embed. The ``vector`` field of
            each chunk is mutated in-place.
        store:
            Open KB store used to persist the embedding vectors.

        Returns
        -------
        list[Chunk]
            The same list that was passed in, with ``vector`` populated on
            every chunk.
        """
        for chunk in chunks:
            vector = embed(chunk.content, dim=self._dim)
            chunk.vector = vector
            store.upsert_vector(chunk.id, vector, model=self._model, dim=self._dim)
        return chunks
