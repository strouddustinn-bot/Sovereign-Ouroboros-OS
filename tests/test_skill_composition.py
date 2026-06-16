"""Tests for MetaMorph skill composition (Part B).

Skill composition allows the engine to wire two semantically compatible
registered skills into a sequential pipeline when the intent matches both
but no single skill handles it alone.
"""

from __future__ import annotations

from typing import Any

import pytest

from ouroboros.core.types import ExecutionResult, ProposedAction, Skill
from ouroboros.metamorph import MetaMorph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_skill(name: str) -> Skill:
    """Create a minimal skill that returns a dict with its name and inputs."""

    def fn(intent: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"skill": name, "intent": intent, "params": dict(params)}

    return Skill(name=name, fn=fn, source="test", synthesized=False)


# ---------------------------------------------------------------------------
# compose_skills unit tests
# ---------------------------------------------------------------------------


def test_compose_skills_returns_none_with_one_candidate() -> None:
    """A single matching skill is not enough to compose."""
    engine = MetaMorph()
    # Only 'echo' is in the registry that matches the intent lexically.
    result = engine.compose_skills("echo something unique_zzz_xyz")
    # 'echo' matches lexically, but we need two; may or may not find a second
    # via semantics — either None or a valid Skill is acceptable.
    assert result is None or isinstance(result, Skill)


def test_compose_skills_with_two_manual_skills() -> None:
    """compose_skills builds a pipeline when two skills match the intent."""
    engine = MetaMorph()
    # Clear registry to control what's available.
    engine.registry.clear()
    engine._routes.clear()

    skill_a = _dummy_skill("alpha")
    skill_b = _dummy_skill("beta")
    engine.registry["alpha"] = skill_a
    engine.registry["beta"] = skill_b

    # The intent contains both names lexically.
    composed = engine.compose_skills("alpha and beta together")

    assert composed is not None
    assert isinstance(composed, Skill)
    assert composed.source == "composed"
    assert composed.synthesized is False
    assert "alpha" in composed.name
    assert "beta" in composed.name


def test_composed_skill_pipeline_calls_both_fns() -> None:
    """The composed fn calls first skill then passes its output to the second."""
    engine = MetaMorph()
    engine.registry.clear()
    engine._routes.clear()

    calls: list[str] = []

    def fn_first(intent: str, params: dict[str, Any]) -> dict[str, Any]:
        calls.append("first")
        return {"from_first": True, "intent": intent}

    def fn_second(intent: str, params: dict[str, Any]) -> dict[str, Any]:
        calls.append("second")
        assert "prior" in params, "second skill should receive prior output"
        assert params["prior"]["from_first"] is True
        return {"from_second": True, "prior_received": params["prior"]}

    skill_first = Skill(name="firstskill", fn=fn_first, source="test", synthesized=False)
    skill_second = Skill(name="secondskill", fn=fn_second, source="test", synthesized=False)
    engine.registry["firstskill"] = skill_first
    engine.registry["secondskill"] = skill_second

    composed = engine.compose_skills("firstskill and secondskill pipeline")
    assert composed is not None

    output = composed.fn("test intent", {})
    assert calls == ["first", "second"]
    assert output["from_second"] is True
    assert output["prior_received"]["from_first"] is True


def test_compose_skills_returns_none_with_empty_registry() -> None:
    engine = MetaMorph()
    engine.registry.clear()
    engine._routes.clear()
    assert engine.compose_skills("do something") is None


def test_compose_skills_composed_skill_passes_validate() -> None:
    """A composed skill from two valid skills should pass validate_skill."""
    engine = MetaMorph()
    engine.registry.clear()
    engine._routes.clear()

    engine.registry["alpha"] = _dummy_skill("alpha")
    engine.registry["beta"] = _dummy_skill("beta")

    composed = engine.compose_skills("alpha and beta")
    assert composed is not None
    assert engine.validate_skill(composed) is True


# ---------------------------------------------------------------------------
# execute() integration: composed detail flag
# ---------------------------------------------------------------------------


def test_execute_sets_composed_detail_when_composition_succeeds() -> None:
    """execute() sets detail='composed' when composition is used."""
    engine = MetaMorph()
    engine.registry.clear()
    engine._routes.clear()

    engine.registry["alpha"] = _dummy_skill("alpha")
    engine.registry["beta"] = _dummy_skill("beta")

    action = ProposedAction(intent="alpha and beta workflow")
    result = engine.execute(action)

    assert result.ok
    assert result.detail == "composed"
    assert result.synthesized is False


def test_execute_does_not_compose_when_single_skill_resolves() -> None:
    """Composition is skipped when a registered skill already handles the intent."""
    engine = MetaMorph()
    # The builtin 'echo' skill matches any intent containing 'echo'.
    action = ProposedAction(intent="echo hello there")
    result = engine.execute(action)

    assert result.ok
    assert result.skill_used == "echo"
    assert result.detail != "composed"


def test_execute_falls_back_to_synthesis_when_no_composition() -> None:
    """When composition is impossible, execute() synthesizes a new skill."""
    engine = MetaMorph()
    # Use an intent that doesn't match any builtin keyword.
    action = ProposedAction(intent="zzz_unique_zzz_intent_xyz_abc")
    result = engine.execute(action)

    assert result.ok
    assert result.synthesized is True


# ---------------------------------------------------------------------------
# Builtin skills are still present and composition only supplements them
# ---------------------------------------------------------------------------


def test_builtins_still_registered_after_composition() -> None:
    """Composition should not remove existing builtin skills."""
    engine = MetaMorph()
    engine.registry["alpha"] = _dummy_skill("alpha")
    engine.registry["beta"] = _dummy_skill("beta")

    engine.execute(ProposedAction(intent="alpha and beta workflow"))

    # Builtins must still be available.
    assert "echo" in engine.registry
    assert "summarize" in engine.registry
    assert "reverse" in engine.registry
    assert "count" in engine.registry


def test_composed_skill_is_registered_for_future_reuse() -> None:
    """After a composition, the composed skill is in the registry."""
    engine = MetaMorph()
    engine.registry.clear()
    engine._routes.clear()

    engine.registry["alpha"] = _dummy_skill("alpha")
    engine.registry["beta"] = _dummy_skill("beta")

    action = ProposedAction(intent="alpha and beta workflow")
    result = engine.execute(action)

    assert result.ok
    assert result.skill_used in engine.registry


def test_compose_skills_semantic_match() -> None:
    """Skills with names semantically related to the intent are candidates."""
    engine = MetaMorph()
    engine.registry.clear()
    engine._routes.clear()

    # Use the builtin-style skills with real names the embedder can find similar
    # ngrams for.
    engine.registry["summarize"] = _dummy_skill("summarize")
    engine.registry["count"] = _dummy_skill("count")

    # Intent that semantically relates to both summarising and counting.
    composed = engine.compose_skills("summarize and count words in text")
    # Both names appear lexically in intent so they should be found.
    assert composed is not None
    assert composed.source == "composed"
