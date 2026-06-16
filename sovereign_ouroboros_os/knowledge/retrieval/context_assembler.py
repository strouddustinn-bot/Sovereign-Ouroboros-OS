"""Context assembly: parent expansion, dedup, token-budget packing, citations.

Takes a ranked list of :class:`~sovereign_ouroboros_os.knowledge.schemas.RetrievedChunk`
objects and produces:

* A single context string ready for insertion into an LLM prompt.
* A list of citation dicts identifying the provenance of each included passage.

Pipeline
--------
1. **Parent expansion** — replace child chunks with their parent when a parent
   exists.  Marks the result with ``expanded=True``.
2. **Dedup by content_hash** — keep only the highest-scoring occurrence of any
   content hash; skip any subsequent duplicate.
3. **Token-budget packing** — greedily include passages in descending-score
   order until the word-count budget is reached.  The last chunk may push the
   total up to 110 % of ``max_tokens`` before being excluded.
4. **Formatting** — each passage is prefixed with its citation index and
   section path / title.
5. **Citations** — one dict per included passage, with source provenance.

No external dependencies — stdlib only (plus project-internal modules).
"""

from __future__ import annotations

from sovereign_ouroboros_os.knowledge.schemas import Chunk, RetrievedChunk
from sovereign_ouroboros_os.knowledge.storage.sqlite_store import SQLiteKBStore


class ContextAssembler:
    """Assemble retrieved chunks into a prompt-ready context string.

    Parameters
    ----------
    store:
        The store used to fetch parent chunks during expansion.
    max_tokens:
        Approximate word-count budget for the assembled context.  (Word count
        is used as a fast proxy for token count; real tokenisation is not
        required here.)
    """

    def __init__(self, store: SQLiteKBStore, max_tokens: int = 2000) -> None:
        self._store = store
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(
        self,
        hits: list[RetrievedChunk],
        expand_to_parent: bool = True,
    ) -> tuple[str, list[dict]]:
        """Build the final context string and citation list.

        Parameters
        ----------
        hits:
            Ranked list of retrieved chunks (descending score).
        expand_to_parent:
            When ``True``, replace child chunks (those with a ``parent_id``)
            with their parent chunk if the parent can be fetched from the store.

        Returns
        -------
        context_text:
            Passages joined with ``"\\n\\n---\\n\\n"``, capped at
            ``max_tokens`` words.
        citations:
            List of citation dicts, one per included passage::

                {
                    "source_uri":   str,
                    "title":        str | None,
                    "section_path": list[str],
                    "version":      int,
                    "chunk_id":     str,
                }
        """
        if not hits:
            return ("", [])

        # ------------------------------------------------------------------
        # 1. Parent expansion
        # ------------------------------------------------------------------
        expanded_hits: list[RetrievedChunk] = []
        for hit in hits:
            if expand_to_parent and hit.chunk.parent_id is not None:
                parent = self._store.get_chunk(hit.chunk.parent_id)
                if parent is not None:
                    expanded_hits.append(
                        RetrievedChunk(
                            chunk=parent,
                            score=hit.score,
                            rrf_score=hit.rrf_score,
                            dense_rank=hit.dense_rank,
                            sparse_rank=hit.sparse_rank,
                            expanded=True,
                        )
                    )
                    continue
            expanded_hits.append(hit)

        # ------------------------------------------------------------------
        # 2. Dedup by content_hash (keep highest-scoring per hash)
        # ------------------------------------------------------------------
        seen_hashes: set[str] = set()
        deduped: list[RetrievedChunk] = []
        for hit in expanded_hits:
            h = hit.chunk.content_hash
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            deduped.append(hit)

        # Ensure descending order by score (caller should already have them
        # sorted, but guard against unsorted input)
        deduped.sort(key=lambda h: h.score, reverse=True)

        # ------------------------------------------------------------------
        # 3. Token-budget packing (word-count proxy)
        # ------------------------------------------------------------------
        max_tokens = self._max_tokens
        running_words = 0
        included: list[RetrievedChunk] = []

        for hit in deduped:
            word_count = len(hit.chunk.content.split())
            if running_words == 0:
                # Always include at least one chunk
                included.append(hit)
                running_words += word_count
            elif running_words + word_count <= max_tokens * 1.1:
                included.append(hit)
                running_words += word_count
            else:
                # Budget exceeded — stop
                break

        if not included:
            return ("", [])

        # ------------------------------------------------------------------
        # 4 & 5. Format passages and build citations
        # ------------------------------------------------------------------
        passages: list[str] = []
        citations: list[dict] = []

        for i, hit in enumerate(included):
            chunk: Chunk = hit.chunk
            md = chunk.metadata

            # Build header: section path takes priority over title
            if chunk.section_path:
                header = ", ".join(chunk.section_path)
            elif md.title:
                header = md.title
            else:
                header = "untitled"

            passage = f"[{i + 1}] ({header})\n{chunk.content}"
            passages.append(passage)

            citations.append(
                {
                    "source_uri": md.source_uri,
                    "title": md.title,
                    "section_path": list(chunk.section_path),
                    "version": md.version,
                    "chunk_id": chunk.id,
                }
            )

        context_text = "\n\n---\n\n".join(passages)
        return (context_text, citations)
