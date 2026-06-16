"""Tests for AgentMemory (SQLite-backed persistent memory).

All tests use ``:memory:`` as the database path to keep them fully isolated
and side-effect-free – no files are left on disk after the test run.
"""

from __future__ import annotations

import pytest

from ouroboros.core.types import ExecutionResult, Skill
from ouroboros.memory import AgentMemory


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_loop_result(
    task: str = "echo hello world",
    step: int = 1,
    succeeded: bool = True,
    blocked: bool = False,
    skill_used: str = "echo",
    timeline_score: float = 0.9,
) -> object:
    """Build a minimal LoopResult-like object without importing OuroborosLoop.

    We use a plain namespace object so the memory tests remain independent of
    the full loop machinery.
    """
    from ouroboros.core.types import (
        ExecutionResult,
        ProposedAction,
        Timeline,
    )
    from ouroboros.ethos_compiler import GateResult

    # The real LoopResult is a dataclass; we instantiate it directly.
    from ouroboros.ouroboros_loop import LoopResult

    execution: ExecutionResult | None = None
    if succeeded:
        execution = ExecutionResult(
            ok=True,
            output={"skill": skill_used},
            skill_used=skill_used,
            synthesized=False,
        )

    action = ProposedAction(intent=task)
    timeline = Timeline(
        id="t1",
        proposed_action=action,
        score=timeline_score,
        rationale="test",
    )
    gate = GateResult(allowed=not blocked, violations=[])

    return LoopResult(
        task=task,
        prototypes=[],
        timeline=timeline,
        gate=gate,
        execution=execution,
        federation=None,
        step=step,
        blocked=blocked,
    )


# ---------------------------------------------------------------------------
# save_result / load_world_state interaction
# ---------------------------------------------------------------------------


def test_save_result_persists_loop_history() -> None:
    """save_result stores a row that can be retrieved via recall_similar."""
    with AgentMemory(":memory:") as mem:
        result = _make_loop_result(task="index the corpus", step=1)
        mem.save_result(result)

        rows = mem.recall_similar("index the corpus", k=5)
        assert len(rows) == 1
        assert rows[0]["task"] == "index the corpus"
        assert rows[0]["succeeded"] is True
        assert rows[0]["skill_used"] == "echo"


# ---------------------------------------------------------------------------
# recall_similar – semantic ordering
# ---------------------------------------------------------------------------


def test_recall_similar_returns_most_similar_not_most_recent() -> None:
    """recall_similar ranks by embedding similarity, not insertion order."""
    with AgentMemory(":memory:") as mem:
        # Insert three tasks in order.  The third (most recent) is unrelated;
        # the first two are about corpus indexing.
        mem.save_result(_make_loop_result(task="index the corpus", step=1))
        mem.save_result(_make_loop_result(task="index the document set", step=2))
        mem.save_result(_make_loop_result(task="play music loudly", step=3))

        similar = mem.recall_similar("index all documents", k=2)

        # Both top results should be indexing tasks, not the music task.
        assert len(similar) == 2
        tasks_returned = {r["task"] for r in similar}
        assert "play music loudly" not in tasks_returned
        assert "index the corpus" in tasks_returned or "index the document set" in tasks_returned


def test_recall_similar_returns_fewer_than_k_when_not_enough_rows() -> None:
    with AgentMemory(":memory:") as mem:
        mem.save_result(_make_loop_result(task="only one task"))
        rows = mem.recall_similar("only one task", k=10)
        assert len(rows) == 1


def test_recall_similar_empty_history() -> None:
    with AgentMemory(":memory:") as mem:
        assert mem.recall_similar("anything") == []


def test_recall_similar_includes_similarity_field() -> None:
    with AgentMemory(":memory:") as mem:
        mem.save_result(_make_loop_result(task="summarize the notes"))
        rows = mem.recall_similar("summarize the notes", k=1)
        assert "similarity" in rows[0]
        assert isinstance(rows[0]["similarity"], float)


# ---------------------------------------------------------------------------
# save_world_state / load_world_state
# ---------------------------------------------------------------------------


def test_save_and_load_world_state() -> None:
    with AgentMemory(":memory:") as mem:
        facts = {"task_a": {"skill": "echo", "score": 0.9}, "count": 42}
        mem.save_world_state(facts)
        loaded = mem.load_world_state()
        assert loaded["count"] == 42
        assert loaded["task_a"]["skill"] == "echo"


def test_world_state_upserts_existing_key() -> None:
    with AgentMemory(":memory:") as mem:
        mem.save_world_state({"key": "original"})
        mem.save_world_state({"key": "updated"})
        loaded = mem.load_world_state()
        assert loaded["key"] == "updated"


def test_load_world_state_empty() -> None:
    with AgentMemory(":memory:") as mem:
        assert mem.load_world_state() == {}


def test_world_state_preserves_nested_structures() -> None:
    with AgentMemory(":memory:") as mem:
        nested = {"outer": {"inner": [1, 2, 3]}}
        mem.save_world_state(nested)
        assert mem.load_world_state()["outer"]["inner"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# save_skill / load_skills
# ---------------------------------------------------------------------------


def test_save_and_load_skills_round_trip() -> None:
    with AgentMemory(":memory:") as mem:
        skill = Skill(name="my_skill", fn=lambda i, p: {}, source="builtin", synthesized=False)
        mem.save_skill(skill)

        skills = mem.load_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "my_skill"
        assert skills[0]["source"] == "builtin"
        assert skills[0]["synthesized"] is False


def test_save_skill_upserts_on_duplicate_name() -> None:
    with AgentMemory(":memory:") as mem:
        skill_v1 = Skill(name="mutable_skill", fn=lambda i, p: {}, source="builtin", synthesized=False)
        skill_v2 = Skill(name="mutable_skill", fn=lambda i, p: {}, source="synthesized", synthesized=True)
        mem.save_skill(skill_v1)
        mem.save_skill(skill_v2)

        skills = mem.load_skills()
        assert len(skills) == 1
        assert skills[0]["source"] == "synthesized"
        assert skills[0]["synthesized"] is True


def test_load_skills_empty() -> None:
    with AgentMemory(":memory:") as mem:
        assert mem.load_skills() == []


def test_save_multiple_skills() -> None:
    with AgentMemory(":memory:") as mem:
        for name in ("alpha", "beta", "gamma"):
            mem.save_skill(Skill(name=name, fn=lambda i, p: {}, source="builtin", synthesized=False))
        skills = mem.load_skills()
        assert {s["name"] for s in skills} == {"alpha", "beta", "gamma"}


# ---------------------------------------------------------------------------
# OuroborosLoop integration
# ---------------------------------------------------------------------------


def test_ouroboros_loop_saves_to_memory() -> None:
    """OuroborosLoop.run() persists the result when memory= is provided."""
    from ouroboros import OuroborosLoop

    with AgentMemory(":memory:") as mem:
        loop = OuroborosLoop(memory=mem)
        loop.run("echo the system status")

        rows = mem.recall_similar("echo", k=5)
        assert len(rows) == 1


def test_ouroboros_loop_without_memory_is_unchanged() -> None:
    """OuroborosLoop works exactly as before when memory is not provided."""
    from ouroboros import OuroborosLoop

    loop = OuroborosLoop()
    result = loop.run("summarize the quarterly report")
    assert result.succeeded
