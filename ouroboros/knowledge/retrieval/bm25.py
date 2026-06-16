"""Pure-Python Okapi BM25 index for sparse keyword retrieval.

Implements the standard Okapi BM25 scoring formula:
    score(q, d) = Σ_t IDF(t) * tf(t,d) * (k1+1) / (tf(t,d) + k1*(1 - b + b*dl/avgdl))

where:
    IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
    tf(t,d) = term frequency of t in document d
    dl       = document length (token count)
    avgdl    = average document length across the corpus
    N        = total number of documents

No external dependencies — stdlib only.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def _tokenize(text: str) -> list[str]:
    """Lowercase-split tokenizer that drops single-character tokens."""
    return [tok for tok in text.lower().split() if len(tok) >= 2]


class BM25Index:
    """Okapi BM25 inverted-index over a flat collection of text chunks.

    Parameters
    ----------
    k1:
        Term-frequency saturation parameter. Higher values make term frequency
        matter more; typical values are 1.2–2.0. Defaults to 1.5.
    b:
        Length normalisation parameter. 0 = no normalisation, 1 = full
        normalisation. Defaults to 0.75.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

        # Populated by index()
        self._chunk_ids: list[str] = []
        # chunk_id → list of tokens
        self._doc_tokens: dict[str, list[str]] = {}
        # chunk_id → document length
        self._doc_len: dict[str, int] = {}
        # term → list of (chunk_id, term_freq)
        self._inverted: dict[str, list[tuple[str, int]]] = defaultdict(list)
        # term → document frequency
        self._df: dict[str, int] = {}
        # Corpus statistics
        self._n: int = 0           # total number of documents
        self._avgdl: float = 0.0   # average document length

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def index(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Build (or rebuild) the BM25 index from parallel chunk_ids + texts.

        Parameters
        ----------
        chunk_ids:
            Ordered list of unique chunk identifiers.
        texts:
            Parallel list of raw text strings for each chunk.

        Raises
        ------
        ValueError
            If *chunk_ids* and *texts* have different lengths.
        """
        if len(chunk_ids) != len(texts):
            raise ValueError(
                f"chunk_ids and texts must have equal length, got "
                f"{len(chunk_ids)} vs {len(texts)}"
            )

        # Reset state
        self._chunk_ids = list(chunk_ids)
        self._doc_tokens = {}
        self._doc_len = {}
        self._inverted = defaultdict(list)
        self._df = {}

        total_tokens = 0

        # First pass: tokenize and build per-document term frequencies
        # term_freq_per_doc: chunk_id → {term: count}
        term_freq_per_doc: dict[str, dict[str, int]] = {}

        for cid, text in zip(chunk_ids, texts):
            tokens = _tokenize(text)
            self._doc_tokens[cid] = tokens
            dl = len(tokens)
            self._doc_len[cid] = dl
            total_tokens += dl

            tf: dict[str, int] = defaultdict(int)
            for tok in tokens:
                tf[tok] += 1
            term_freq_per_doc[cid] = dict(tf)

        self._n = len(chunk_ids)
        self._avgdl = total_tokens / self._n if self._n > 0 else 0.0

        # Second pass: build inverted index and document frequencies
        for cid, tf in term_freq_per_doc.items():
            for term, freq in tf.items():
                self._inverted[term].append((cid, freq))

        for term, postings in self._inverted.items():
            self._df[term] = len(postings)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _idf(self, term: str) -> float:
        """Compute IDF for a term using the Robertson-Spärck Jones formula."""
        df = self._df.get(term, 0)
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1)

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Score the indexed corpus against *query* and return top-k hits.

        Parameters
        ----------
        query:
            Raw query string. Tokenized identically to the indexed documents.
        top_k:
            Maximum number of results to return.
        candidate_ids:
            Optional set of chunk ids to restrict scoring to. When provided,
            only chunks in this set are scored (used for metadata pre-filtering).

        Returns
        -------
        list of (chunk_id, bm25_score) tuples sorted by descending score.
        """
        if self._n == 0:
            return []

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        scores: dict[str, float] = defaultdict(float)

        for term in query_terms:
            if term not in self._inverted:
                continue
            idf = self._idf(term)
            for cid, tf in self._inverted[term]:
                # Skip if candidate filter is active and chunk not in it
                if candidate_ids is not None and cid not in candidate_ids:
                    continue
                dl = self._doc_len[cid]
                avgdl = self._avgdl
                k1 = self.k1
                b = self.b
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                scores[cid] += idf * tf_norm

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
