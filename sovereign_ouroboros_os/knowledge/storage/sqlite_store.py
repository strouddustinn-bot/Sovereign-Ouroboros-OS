"""SQLite-backed storage for the Ouroboros knowledge base.

Implements four tables following the layered architecture:
- ``sources``     — provenance roots (SourceRecord)
- ``documents``   — normalised full documents (Document)
- ``chunks``      — retrievable units with metadata (Chunk, minus the vector)
- ``chunk_vectors`` — dense embeddings stored as JSON, keyed by (chunk_id, model)

Design notes
------------
- Postgres/pgvector is the production recommendation; SQLite + JSON columns gives
  the same schema in a zero-dependency dev/test environment.
- Raw artifacts are *not* stored here (they belong in an object store); only the
  normalised content and metadata are stored.
- Vectors are separated into their own table so re-embedding on model change is
  a targeted DELETE + re-insert rather than a full row rewrite.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from sovereign_ouroboros_os.knowledge.schemas import (
    ACCESS_LEVELS,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    Chunk,
    ChunkMetadata,
    Document,
    SourceRecord,
)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sources (
    id          TEXT PRIMARY KEY,
    uri         TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,
    checksum    TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    access_level TEXT NOT NULL DEFAULT 'public',
    authority   REAL NOT NULL DEFAULT 0.8,
    status      TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    source_id    TEXT NOT NULL REFERENCES sources(id),
    title        TEXT NOT NULL,
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    section_tree TEXT NOT NULL DEFAULT '[]',   -- JSON
    language     TEXT NOT NULL DEFAULT 'en',
    version      INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL REFERENCES documents(id),
    source_id    TEXT NOT NULL REFERENCES sources(id),
    parent_id    TEXT REFERENCES chunks(id),
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    section_path TEXT NOT NULL DEFAULT '[]',   -- JSON array
    tokens       INTEGER NOT NULL DEFAULT 0,
    keywords     TEXT NOT NULL DEFAULT '[]',   -- JSON array
    entities     TEXT NOT NULL DEFAULT '[]',   -- JSON array
    -- ChunkMetadata fields flattened for filtering
    domain       TEXT NOT NULL DEFAULT '',
    language     TEXT NOT NULL DEFAULT 'en',
    source_type  TEXT NOT NULL DEFAULT 'doc',
    source_uri   TEXT NOT NULL DEFAULT '',
    version      INTEGER NOT NULL DEFAULT 1,
    access_level TEXT NOT NULL DEFAULT 'public',
    authority    REAL NOT NULL DEFAULT 0.8,
    valid_from   TEXT,
    valid_until  TEXT,
    superseded_by TEXT REFERENCES chunks(id),
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    title        TEXT,
    topics       TEXT NOT NULL DEFAULT '[]'    -- JSON array
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_domain   ON chunks(domain);
CREATE INDEX IF NOT EXISTS idx_chunks_source   ON chunks(source_id);

CREATE TABLE IF NOT EXISTS chunk_vectors (
    chunk_id TEXT NOT NULL REFERENCES chunks(id),
    model    TEXT NOT NULL,
    dim      INTEGER NOT NULL,
    vector   TEXT NOT NULL,            -- JSON array of floats
    PRIMARY KEY (chunk_id, model)
);
"""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class SQLiteKBStore:
    """SQLite-backed knowledge base storage adapter.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Use ``":memory:"`` for ephemeral
        in-process storage (tests).  Defaults to ``"./kb.db"``.
    """

    def __init__(self, db_path: str = "./kb.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Source records
    # ------------------------------------------------------------------

    def upsert_source(self, src: SourceRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO sources
                (id, uri, source_type, checksum, ingested_at, last_seen_at,
                 access_level, authority, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                last_seen_at = excluded.last_seen_at,
                checksum     = excluded.checksum,
                status       = excluded.status
            """,
            (
                src.id, src.uri, src.source_type, src.checksum,
                src.ingested_at, src.last_seen_at,
                src.access_level, src.authority, src.status,
            ),
        )
        self._conn.commit()

    def get_source(self, source_id: str) -> SourceRecord | None:
        row = self._conn.execute(
            "SELECT * FROM sources WHERE id = ?", (source_id,)
        ).fetchone()
        return _row_to_source(row) if row else None

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def upsert_document(self, doc: Document) -> None:
        self._conn.execute(
            """
            INSERT INTO documents
                (id, source_id, title, content, content_hash,
                 section_tree, language, version, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                content      = excluded.content,
                content_hash = excluded.content_hash,
                section_tree = excluded.section_tree,
                version      = excluded.version,
                updated_at   = excluded.updated_at
            """,
            (
                doc.id, doc.source_id, doc.title, doc.content,
                doc.content_hash, json.dumps(doc.section_tree),
                doc.language, doc.version, doc.created_at, doc.updated_at,
            ),
        )
        self._conn.commit()

    def get_document(self, doc_id: str) -> Document | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        return _row_to_document(row) if row else None

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def upsert_chunk(self, chunk: Chunk) -> None:
        md = chunk.metadata
        self._conn.execute(
            """
            INSERT INTO chunks
                (id, document_id, source_id, parent_id, content, content_hash,
                 chunk_index, section_path, tokens, keywords, entities,
                 domain, language, source_type, source_uri, version,
                 access_level, authority, valid_from, valid_until,
                 superseded_by, created_at, updated_at, title, topics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                content       = excluded.content,
                content_hash  = excluded.content_hash,
                keywords      = excluded.keywords,
                valid_until   = excluded.valid_until,
                superseded_by = excluded.superseded_by,
                updated_at    = excluded.updated_at
            """,
            (
                chunk.id, chunk.document_id, chunk.source_id, chunk.parent_id,
                chunk.content, chunk.content_hash, chunk.chunk_index,
                json.dumps(chunk.section_path), chunk.tokens,
                json.dumps(chunk.keywords), json.dumps(chunk.entities),
                md.domain, md.language, md.source_type, md.source_uri,
                md.version, md.access_level, md.authority,
                md.valid_from, md.valid_until, md.superseded_by,
                md.created_at, md.updated_at, md.title,
                json.dumps(md.topics),
            ),
        )
        self._conn.commit()

    def upsert_vector(
        self, chunk_id: str, vector: tuple[float, ...],
        model: str = EMBEDDING_MODEL, dim: int = EMBEDDING_DIM
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO chunk_vectors (chunk_id, model, dim, vector)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chunk_id, model) DO UPDATE SET
                vector = excluded.vector, dim = excluded.dim
            """,
            (chunk_id, model, dim, json.dumps(list(vector))),
        )
        self._conn.commit()

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if not row:
            return None
        vec_row = self._conn.execute(
            "SELECT vector, model, dim FROM chunk_vectors WHERE chunk_id = ? LIMIT 1",
            (chunk_id,),
        ).fetchone()
        return _row_to_chunk(row, vec_row)

    def get_chunk_vector(
        self, chunk_id: str, model: str = EMBEDDING_MODEL
    ) -> tuple[float, ...] | None:
        row = self._conn.execute(
            "SELECT vector FROM chunk_vectors WHERE chunk_id = ? AND model = ?",
            (chunk_id, model),
        ).fetchone()
        return tuple(json.loads(row["vector"])) if row else None

    def filter_chunks(
        self,
        domain: str | None = None,
        access_levels: list[str] | None = None,
        valid_now: bool = True,
        source_type: str | None = None,
    ) -> list[str]:
        """Return chunk ids passing the metadata pre-filter."""
        clauses: list[str] = ["superseded_by IS NULL"]
        params: list[Any] = []

        if domain:
            clauses.append("domain = ?")
            params.append(domain)

        levels = access_levels or list(ACCESS_LEVELS)
        placeholders = ",".join("?" * len(levels))
        clauses.append(f"access_level IN ({placeholders})")
        params.extend(levels)

        if valid_now:
            clauses.append("(valid_until IS NULL OR valid_until > datetime('now'))")

        if source_type:
            clauses.append("source_type = ?")
            params.append(source_type)

        sql = "SELECT id FROM chunks WHERE " + " AND ".join(clauses)
        rows = self._conn.execute(sql, params).fetchall()
        return [r["id"] for r in rows]

    def get_all_chunks_with_vectors(
        self, chunk_ids: list[str], model: str = EMBEDDING_MODEL
    ) -> list[tuple[Chunk, tuple[float, ...]]]:
        """Batch-fetch chunks and their vectors. Returns only chunks that have a vector."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        chunk_rows = {
            r["id"]: r
            for r in self._conn.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids
            ).fetchall()
        }
        vec_rows = {
            r["chunk_id"]: r
            for r in self._conn.execute(
                f"""SELECT chunk_id, vector, model, dim FROM chunk_vectors
                    WHERE chunk_id IN ({placeholders}) AND model = ?""",
                chunk_ids + [model],
            ).fetchall()
        }
        results: list[tuple[Chunk, tuple[float, ...]]] = []
        for cid in chunk_ids:
            if cid in chunk_rows and cid in vec_rows:
                chunk = _row_to_chunk(chunk_rows[cid], vec_rows[cid])
                vec = tuple(json.loads(vec_rows[cid]["vector"]))
                if chunk:
                    results.append((chunk, vec))
        return results

    def supersede_chunk(self, old_id: str, new_id: str) -> None:
        """Mark *old_id* as superseded by *new_id*."""
        from datetime import datetime, timezone

        self._conn.execute(
            "UPDATE chunks SET superseded_by = ?, updated_at = ? WHERE id = ?",
            (new_id, datetime.now(timezone.utc).isoformat(), old_id),
        )
        self._conn.commit()

    def count_chunks(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE superseded_by IS NULL"
        ).fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteKBStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Row → dataclass helpers
# ---------------------------------------------------------------------------


def _row_to_source(row: sqlite3.Row) -> SourceRecord:
    return SourceRecord(
        id=row["id"], uri=row["uri"], source_type=row["source_type"],
        checksum=row["checksum"], ingested_at=row["ingested_at"],
        last_seen_at=row["last_seen_at"], access_level=row["access_level"],
        authority=row["authority"], status=row["status"],
    )


def _row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        id=row["id"], source_id=row["source_id"], title=row["title"],
        content=row["content"], content_hash=row["content_hash"],
        section_tree=json.loads(row["section_tree"]),
        language=row["language"], version=row["version"],
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


def _row_to_chunk(
    row: sqlite3.Row,
    vec_row: sqlite3.Row | None = None,
) -> Chunk:
    vector: tuple[float, ...] = ()
    model = EMBEDDING_MODEL
    dim = EMBEDDING_DIM
    if vec_row is not None:
        vector = tuple(json.loads(vec_row["vector"]))
        model = vec_row["model"]
        dim = vec_row["dim"]

    md = ChunkMetadata(
        domain=row["domain"], language=row["language"],
        source_type=row["source_type"], source_uri=row["source_uri"],
        version=row["version"], access_level=row["access_level"],
        authority=row["authority"], created_at=row["created_at"],
        updated_at=row["updated_at"], title=row["title"],
        topics=json.loads(row["topics"]),
        valid_from=row["valid_from"], valid_until=row["valid_until"],
        superseded_by=row["superseded_by"],
    )
    return Chunk(
        id=row["id"], document_id=row["document_id"],
        source_id=row["source_id"], parent_id=row["parent_id"],
        content=row["content"], content_hash=row["content_hash"],
        chunk_index=row["chunk_index"],
        section_path=json.loads(row["section_path"]),
        tokens=row["tokens"], vector=vector, vector_model=model,
        vector_dim=dim, keywords=json.loads(row["keywords"]),
        entities=json.loads(row["entities"]), metadata=md,
    )
