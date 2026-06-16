"""Persistent SQLite-backed memory for the Sovereign Ouroboros agent.

Provides :class:`AgentMemory`, a lightweight store that persists loop history,
world-state facts, and the skill registry across process restarts.  All I/O
goes through the stdlib :mod:`sqlite3` module – no external ORM is required.
"""

from __future__ import annotations

from sovereign_ouroboros_os.memory.store import AgentMemory

__all__ = ["AgentMemory"]
