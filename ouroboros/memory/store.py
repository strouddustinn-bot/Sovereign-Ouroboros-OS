"""SQLite-backed persistent memory store for the Ouroboros agent.

:class:`AgentMemory` persists three classes of information between process
runs:

* **loop_history** – every :class:`~ouroboros.ouroboros_loop.LoopResult`
  the agent has processed, used for semantic recall.
* **world_state** – key/value fact store mirroring
  :class:`~ouroboros.core.types.WorldState`.
* **skill_registry** – skills that have been synthesised or composed and
  should survive restarts.

The module purposefully depends only on the stdlib ``sqlite3`` module.
Embeddings are delegated to :func:`~ouroboros.core.embedding.embed`
and :func:`~ouroboros.core.embedding.cosine` so no numerical
logic is duplicated here.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import TYPE_CHECKING, Any

from ouroboros.core.embedding import cosine, embed
from ouroboros.core.types import Skill

if TYPE_CHECKING:
    from ouroboros.ouroboros_loop import LoopResult

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS loop_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    step INTEGER NOT NULL,
    succeeded INTEGER NOT NULL,
    blocked INTEGER NOT NULL,
    skill_used TEXT,
    timeline_score REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS world_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS skill_registry (
    name TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    synthesized INTEGER NOT NULL,
    registered_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS metering (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    task_hash TEXT NOT NULL,
    step INTEGER,
    succeeded INTEGER,
    blocked INTEGER,
    skill_used TEXT,
    synthesized INTEGER,
    kb_hits INTEGER,
    n_prototypes INTEGER,
    timeline_score REAL,
    ts TEXT DEFAULT (datetime('now'))
);
"""


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class AgentMemory:
    """SQLite-backed persistent memory for the Ouroboros agent.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Use ``":memory:"`` for an
        in-process, ephemeral database (useful in tests).  Defaults to
        ``"./ouroboros_memory.db"``.
    """

    def __init__(self, db_path: str = "./ouroboros_memory.db") -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Loop history
    # ------------------------------------------------------------------

    def save_result(self, result: LoopResult) -> None:
        """Persist a :class:`LoopResult` row to ``loop_history``.

        Parameters
        ----------
        result:
            The completed loop result to store.
        """
        skill_used: str | None = None
        timeline_score: float | None = None

        if result.execution is not None:
            skill_used = result.execution.skill_used

        if result.timeline is not None:
            timeline_score = result.timeline.score

        self._conn.execute(
            """
            INSERT INTO loop_history
                (task, step, succeeded, blocked, skill_used, timeline_score)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result.task,
                result.step,
                int(result.succeeded),
                int(result.blocked),
                skill_used,
                timeline_score,
            ),
        )
        self._conn.commit()

    def recall_similar(self, task: str, k: int = 3) -> list[dict[str, Any]]:
        """Return the *k* past results most semantically similar to *task*.

        Similarity is computed as cosine distance between character n-gram
        embeddings (via :func:`~ouroboros.core.embedding.embed`).
        If fewer than *k* rows exist, all rows are returned.

        Parameters
        ----------
        task:
            The query task string.
        k:
            Maximum number of results to return.

        Returns
        -------
        list[dict]:
            Each dict contains: ``task``, ``step``, ``succeeded``,
            ``skill_used``, ``similarity``.
        """
        rows = self._conn.execute(
            "SELECT task, step, succeeded, skill_used FROM loop_history"
        ).fetchall()

        if not rows:
            return []

        query_vec = embed(task)
        scored: list[tuple[float, dict[str, Any]]] = []

        for row in rows:
            past_task: str = row["task"]
            sim = cosine(query_vec, embed(past_task))
            scored.append(
                (
                    sim,
                    {
                        "task": past_task,
                        "step": row["step"],
                        "succeeded": bool(row["succeeded"]),
                        "skill_used": row["skill_used"],
                        "similarity": sim,
                    },
                )
            )

        scored.sort(key=lambda t: t[0], reverse=True)
        return [entry for _, entry in scored[:k]]

    # ------------------------------------------------------------------
    # World state
    # ------------------------------------------------------------------

    def save_world_state(self, state_facts: dict[str, Any]) -> None:
        """Upsert *state_facts* into the ``world_state`` table.

        Each value is JSON-serialised before storage so arbitrary Python
        objects can be persisted without a custom adapter.

        Parameters
        ----------
        state_facts:
            Mapping of fact key → value.  Values must be JSON-serialisable.
        """
        for key, value in state_facts.items():
            self._conn.execute(
                """
                INSERT INTO world_state (key, value, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, json.dumps(value)),
            )
        self._conn.commit()

    def load_world_state(self) -> dict[str, Any]:
        """Load all ``world_state`` rows and return them as a plain dict.

        JSON values are deserialised back to Python objects.

        Returns
        -------
        dict:
            Mapping of fact key → deserialized value.
        """
        rows = self._conn.execute(
            "SELECT key, value FROM world_state"
        ).fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    # ------------------------------------------------------------------
    # Skill registry
    # ------------------------------------------------------------------

    def save_skill(self, skill: Skill) -> None:
        """Persist *skill* metadata to the ``skill_registry`` table.

        If a skill with the same name already exists, the row is updated
        in-place (upsert).

        Parameters
        ----------
        skill:
            The skill to persist.  Only ``name``, ``source``, and
            ``synthesized`` are stored; the callable ``fn`` is not
            serialisable and is intentionally omitted.
        """
        self._conn.execute(
            """
            INSERT INTO skill_registry (name, source, synthesized, registered_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(name) DO UPDATE SET
                source = excluded.source,
                synthesized = excluded.synthesized,
                registered_at = excluded.registered_at
            """,
            (skill.name, skill.source, int(skill.synthesized)),
        )
        self._conn.commit()

    def load_skills(self) -> list[dict[str, Any]]:
        """Return all ``skill_registry`` rows as a list of dicts.

        Returns
        -------
        list[dict]:
            Each dict contains: ``name``, ``source``, ``synthesized``,
            ``registered_at``.
        """
        rows = self._conn.execute(
            "SELECT name, source, synthesized, registered_at FROM skill_registry"
        ).fetchall()
        return [
            {
                "name": row["name"],
                "source": row["source"],
                "synthesized": bool(row["synthesized"]),
                "registered_at": row["registered_at"],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Metering
    # ------------------------------------------------------------------

    def record_usage(
        self,
        tenant_id: str,
        result: "LoopResult",
        kb_hits: int = 0,
    ) -> None:
        """Insert a metering row for one completed loop run.

        Parameters
        ----------
        tenant_id:
            The tenant identifier for multi-tenant environments.
        result:
            The completed :class:`~ouroboros.ouroboros_loop.LoopResult`.
        kb_hits:
            Number of knowledge-base hits returned during the recall stage.
        """
        task_hash = hashlib.sha256(result.task.encode()).hexdigest()[:16]
        skill_used: str | None = None
        synthesized: int | None = None
        if result.execution is not None:
            skill_used = result.execution.skill_used
            synthesized = int(result.execution.synthesized)
        timeline_score: float | None = None
        if result.timeline is not None:
            timeline_score = result.timeline.score
        n_prototypes = len(result.prototypes) if result.prototypes is not None else 0
        self._conn.execute(
            """
            INSERT INTO metering
                (tenant_id, task_hash, step, succeeded, blocked, skill_used,
                 synthesized, kb_hits, n_prototypes, timeline_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tenant_id,
                task_hash,
                result.step,
                int(result.succeeded),
                int(result.blocked),
                skill_used,
                synthesized,
                kb_hits,
                n_prototypes,
                timeline_score,
            ),
        )
        self._conn.commit()

    def get_usage_summary(self, tenant_id: str = "default") -> dict:
        """Return aggregated usage metrics for *tenant_id*.

        Returns
        -------
        dict with keys:
            ``total_runs``, ``succeeded``, ``blocked``, ``unique_tasks``,
            ``avg_score``.
        """
        row = self._conn.execute(
            """
            SELECT
                COUNT(*)                        AS total_runs,
                SUM(succeeded)                  AS succeeded,
                SUM(blocked)                    AS blocked,
                COUNT(DISTINCT task_hash)       AS unique_tasks,
                AVG(timeline_score)             AS avg_score
            FROM metering
            WHERE tenant_id = ?
            """,
            (tenant_id,),
        ).fetchone()
        return {
            "total_runs": row["total_runs"] or 0,
            "succeeded": row["succeeded"] or 0,
            "blocked": row["blocked"] or 0,
            "unique_tasks": row["unique_tasks"] or 0,
            "avg_score": row["avg_score"],
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "AgentMemory":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
