"""Structure-aware small-to-big chunker for the Ouroboros knowledge base.

Produces two layers of chunks from a Document:

* **Parent chunks** — overlapping 600-word windows that provide broad context.
* **Child chunks** — 150-word sub-windows that are the actual retrievable units.
  Each child's ``parent_id`` is set to its enclosing parent, enabling the
  small-to-big retrieval pattern: retrieve the precise child, expand to the
  richer parent for generation.

The chunker is structure-aware: it detects Markdown headings and ALL-CAPS
section headers to avoid splitting across section boundaries, maintains a
running ``section_path`` hierarchy, and populates BM25 ``keywords`` and
``entities`` for every chunk.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from sovereign_ouroboros_os.knowledge.schemas import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    Chunk,
    ChunkMetadata,
    Document,
)

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Word-count thresholds treated as proxy for token counts.
_DEFAULT_SMALL_SIZE: int = 150
_DEFAULT_PARENT_SIZE: int = 600

#: Overlap in words between successive parent windows.
_PARENT_OVERLAP: int = 100

#: Overlap in words between successive child windows within a parent.
_CHILD_OVERLAP: int = 30

#: Sentence-end characters used to avoid mid-sentence splits.
_SENTENCE_END = frozenset(".!?\n")

#: Common English stop words excluded from BM25 keyword extraction.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "that", "this", "with", "from", "have",
        "been", "will", "they", "their", "there", "what", "when",
        "which", "were", "also", "about", "into", "more", "than",
        "some", "such", "your", "each", "over", "these", "those",
        "then", "them", "both", "very",
    }
)

#: Maximum keywords per chunk.
_MAX_KEYWORDS: int = 30

#: Maximum entities per chunk.
_MAX_ENTITIES: int = 20

# Regex patterns for structure detection
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_ALL_CAPS_LINE = re.compile(r"^[A-Z][A-Z\s\d\-_]{3,}$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_heading(line: str) -> tuple[int, str] | None:
    """Return (depth, title) if *line* is a Markdown heading or ALL-CAPS header.

    Markdown depth maps directly to heading level (1–6).
    ALL-CAPS lines are assigned depth 1 (treated as top-level sections).
    Returns ``None`` if the line is not a heading.
    """
    m = _MARKDOWN_HEADING.match(line.rstrip())
    if m:
        return len(m.group(1)), m.group(2).strip()
    stripped = line.strip()
    if stripped and _ALL_CAPS_LINE.match(stripped):
        return 1, stripped.title()
    return None


def _extract_keywords(text: str) -> list[str]:
    """Extract lowercase non-stop words longer than 3 chars, sorted, max 30."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    seen: set[str] = set()
    result: list[str] = []
    for w in words:
        if len(w) > 3 and w not in _STOP_WORDS and w not in seen:
            seen.add(w)
            result.append(w)
    return sorted(result)[:_MAX_KEYWORDS]


def _extract_entities(text: str) -> list[str]:
    """Extract words that start with a capital letter but are NOT sentence-starts.

    A word is considered an entity candidate when:
    - It begins with an uppercase letter.
    - It is not the first word of a sentence (preceded by ``.``, ``!``, ``?``,
      a newline, or is the very first word in the text).

    Returns a deduplicated list of up to 20 entity strings.
    """
    # Tokenise with positional context
    tokens = list(re.finditer(r"[A-Za-z][A-Za-z\-']*", text))
    entities: list[str] = []
    seen: set[str] = set()
    for i, m in enumerate(tokens):
        word = m.group()
        if not word[0].isupper():
            continue
        # Check if this is a sentence-start by looking at preceding character
        start = m.start()
        preceding = text[:start].rstrip()
        if not preceding:
            continue  # first word — skip
        last_char = preceding[-1] if preceding else ""
        if last_char in ".!?\n":
            continue  # sentence start — skip
        if word not in seen:
            seen.add(word)
            entities.append(word)
        if len(entities) >= _MAX_ENTITIES:
            break
    return entities


def _find_sentence_boundary(words: list[str], target: int) -> int:
    """Find a word index at or near *target* that falls after a sentence end.

    Scans backward from *target* looking for a word ending in ``.``, ``!``,
    or ``?``.  Falls back to *target* if no boundary is found within
    ``target // 2`` words.
    """
    limit = max(0, target - target // 2)
    for i in range(target, limit, -1):
        if i < len(words) and words[i - 1].rstrip("\"')}]").endswith((".", "!", "?")):
            return i
    return target


def _build_section_path(
    stack: list[tuple[int, str]],
) -> list[str]:
    """Return ordered list of heading titles from the current heading stack."""
    return [title for _, title in stack]


def _scan_section_tree(text: str) -> list[dict]:
    """Build a simple section-tree dict from document headings (for Document.section_tree)."""
    root: list[dict] = []
    stack: list[tuple[int, list[dict]]] = []  # (depth, children list)

    for line in text.splitlines():
        result = _detect_heading(line)
        if result is None:
            continue
        depth, title = result
        node: dict = {"heading": title, "depth": depth, "children": []}

        # Pop stack until we find a parent shallower than current depth
        while stack and stack[-1][0] >= depth:
            stack.pop()

        if stack:
            stack[-1][1].append(node)
        else:
            root.append(node)

        stack.append((depth, node["children"]))

    return root


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class Chunker:
    """Structure-aware, small-to-big chunker.

    Parameters
    ----------
    small_size:
        Target word-count for child (retrievable) chunks.
    parent_size:
        Target word-count for parent (context-expansion) chunks.
    """

    def __init__(
        self,
        small_size: int = _DEFAULT_SMALL_SIZE,
        parent_size: int = _DEFAULT_PARENT_SIZE,
    ) -> None:
        self.small_size = small_size
        self.parent_size = parent_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, doc: Document, metadata: ChunkMetadata) -> list[Chunk]:
        """Chunk *doc* into parent + child layers.

        Parameters
        ----------
        doc:
            The normalised document to chunk.
        metadata:
            Shared metadata template applied to every chunk.

        Returns
        -------
        list[Chunk]
            Interleaved list of parent chunks (with ``parent_id=None``) and
            child chunks (with ``parent_id`` pointing to the enclosing parent).
            Child chunks are retrievable; parent chunks provide context.
        """
        text = doc.content
        all_chunks: list[Chunk] = []
        child_index: int = 0  # global sequential index across all children

        # Split text into lines for structure analysis
        lines = text.splitlines(keepends=True)

        # Pre-process: build (word_index → section_stack) mapping by scanning
        # lines and tracking heading state.
        words, word_section_stacks = self._tokenise_with_sections(lines)

        if not words:
            return []

        # Generate parent windows
        parent_start = 0
        while parent_start < len(words):
            parent_end = _find_sentence_boundary(
                words, min(parent_start + self.parent_size, len(words))
            )
            if parent_end <= parent_start:
                parent_end = min(parent_start + self.parent_size, len(words))

            parent_words = words[parent_start:parent_end]
            parent_content = " ".join(parent_words)
            parent_section_path = _build_section_path(
                word_section_stacks[parent_start]
            )
            parent_content_hash = Chunk.make_content_hash(parent_content)
            parent_chunk_index = child_index  # will be refined below; placeholder
            parent_id = Chunk.make_id(
                doc.source_id, parent_start, parent_content_hash
            )

            # Build parent Chunk (parent_id=None)
            parent_chunk = Chunk(
                id=parent_id,
                document_id=doc.id,
                source_id=doc.source_id,
                content=parent_content,
                content_hash=parent_content_hash,
                chunk_index=parent_start,  # word-offset as stable index
                section_path=parent_section_path,
                tokens=len(parent_words),
                vector=(),  # filled by Embedder
                vector_model=EMBEDDING_MODEL,
                vector_dim=EMBEDDING_DIM,
                keywords=_extract_keywords(parent_content),
                entities=_extract_entities(parent_content),
                metadata=metadata,
                parent_id=None,
                tokens_estimate=len(parent_words),
            )

            # Generate child windows within this parent
            child_start = parent_start
            children: list[Chunk] = []
            while child_start < parent_end:
                child_end = _find_sentence_boundary(
                    words, min(child_start + self.small_size, parent_end)
                )
                if child_end <= child_start:
                    child_end = min(child_start + self.small_size, parent_end)

                child_words = words[child_start:child_end]
                child_content = " ".join(child_words)
                child_section_path = _build_section_path(
                    word_section_stacks[child_start]
                )
                child_content_hash = Chunk.make_content_hash(child_content)
                child_id = Chunk.make_id(
                    doc.source_id, child_index, child_content_hash
                )

                child_chunk = Chunk(
                    id=child_id,
                    document_id=doc.id,
                    source_id=doc.source_id,
                    content=child_content,
                    content_hash=child_content_hash,
                    chunk_index=child_index,
                    section_path=child_section_path,
                    tokens=len(child_words),
                    vector=(),  # filled by Embedder
                    vector_model=EMBEDDING_MODEL,
                    vector_dim=EMBEDDING_DIM,
                    keywords=_extract_keywords(child_content),
                    entities=_extract_entities(child_content),
                    metadata=metadata,
                    parent_id=parent_id,
                    tokens_estimate=len(parent_words),
                )
                children.append(child_chunk)
                child_index += 1

                # Advance child window with overlap
                next_child_start = child_end - _CHILD_OVERLAP
                if next_child_start <= child_start:
                    next_child_start = child_end
                child_start = next_child_start

            all_chunks.append(parent_chunk)
            all_chunks.extend(children)

            # Advance parent window with overlap
            next_parent_start = parent_end - _PARENT_OVERLAP
            if next_parent_start <= parent_start:
                next_parent_start = parent_end
            parent_start = next_parent_start

        return all_chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tokenise_with_sections(
        self, lines: list[str]
    ) -> tuple[list[str], list[list[tuple[int, str]]]]:
        """Split the document into words and build a per-word section stack.

        Returns
        -------
        words:
            Flat list of every word in the document (in order).
        word_section_stacks:
            Parallel list: ``word_section_stacks[i]`` is the heading stack
            active when word *i* was emitted (a list of ``(depth, title)``
            tuples, ordered from outermost to innermost heading).
        """
        words: list[str] = []
        word_section_stacks: list[list[tuple[int, str]]] = []

        # Current heading stack: list of (depth, title), shallowest first
        heading_stack: list[tuple[int, str]] = []

        for line in lines:
            stripped = line.rstrip("\n").rstrip()
            result = _detect_heading(stripped)
            if result is not None:
                depth, title = result
                # Pop deeper/equal levels from stack
                while heading_stack and heading_stack[-1][0] >= depth:
                    heading_stack.pop()
                heading_stack.append((depth, title))
                # Don't emit heading text as content words —
                # the heading itself is captured in section_path
                continue

            # Emit content words
            line_words = stripped.split()
            snapshot = list(heading_stack)  # capture current state
            for w in line_words:
                words.append(w)
                word_section_stacks.append(snapshot)

        return words, word_section_stacks
