"""Tests for the MetaMorph self-modifying execution engine."""

import sys

import pytest

from ouroboros.core import ProposedAction
from ouroboros.metamorph import MetaMorph


def test_builtin_skill_handles_matching_intent():
    engine = MetaMorph()
    result = engine.execute(ProposedAction(intent="please reverse this string"))

    assert result.ok
    assert result.synthesized is False
    assert result.skill_used == "reverse"
    assert result.output["reversed"] == "please reverse this string"[::-1]


def test_builtins_seeded_in_registry():
    engine = MetaMorph()
    assert set(engine.skills) >= {"echo", "reverse", "count", "summarize"}


def test_unknown_intent_triggers_synthesis():
    engine = MetaMorph()
    before = set(engine.skills)

    result = engine.execute(ProposedAction(intent="transmogrify the widget"))

    assert result.ok
    assert result.synthesized is True
    assert result.skill_used not in before
    assert result.skill_used in engine.skills
    assert len(engine.skills) == len(before) + 1


def test_synthesized_skill_returns_usable_output():
    engine = MetaMorph()
    result = engine.execute(ProposedAction(intent="quux the gizmo", params={"x": 1}))

    assert result.ok
    out = result.output
    assert out["synthesized"] is True
    assert out["handled_intent"] == "quux the gizmo"
    assert out["token_count"] == 3
    assert out["transformed"] == "quux the gizmo"[::-1]
    assert out["params"] == {"x": 1}


def test_synthesis_is_deterministic():
    engine_a = MetaMorph()
    engine_b = MetaMorph()

    a = engine_a.execute(ProposedAction(intent="frobnicate the foobar"))
    b = engine_b.execute(ProposedAction(intent="frobnicate the foobar"))

    assert a.skill_used == b.skill_used
    assert a.synthesized == b.synthesized
    assert a.output == b.output


def test_second_call_reuses_synthesized_skill():
    engine = MetaMorph()
    first = engine.execute(ProposedAction(intent="zorptangle now"))
    count_after_first = len(engine.skills)

    second = engine.execute(ProposedAction(intent="zorptangle now"))

    assert first.synthesized is True
    # Skill is now registered, so the keyword resolves it: no new synthesis.
    assert second.synthesized is False
    assert second.skill_used == first.skill_used
    assert len(engine.skills) == count_after_first


def test_synthesis_does_not_leak_into_module_globals():
    import ouroboros.metamorph.evolution as evolution

    engine = MetaMorph()
    skill = engine.synthesize_skill("leaky sentinel intent")

    # The synthesized function name must not appear as a host module global.
    assert not hasattr(evolution, skill.name)

    # Its namespace builtins are restricted to the safe allowlist only.
    builtins_ns = skill.fn.__globals__["__builtins__"]
    assert "open" not in builtins_ns
    assert "__import__" not in builtins_ns
    assert "eval" not in builtins_ns
    assert "exec" not in builtins_ns
    assert set(builtins_ns) == set(evolution._SAFE_BUILTINS)


def test_validate_skill_accepts_well_formed_skill():
    engine = MetaMorph()
    candidate = engine.synthesize_skill("validate me")
    assert engine.validate_skill(candidate) is True


def test_validate_skill_rejects_non_dict_output():
    from ouroboros.core import Skill

    engine = MetaMorph()
    bad = Skill(name="bad", fn=lambda intent, params: "not a dict")
    assert engine.validate_skill(bad) is False


def test_engine_satisfies_evolver_protocol():
    from ouroboros.core import Evolver

    assert isinstance(MetaMorph(), Evolver)


def test_close_shuts_down_executor_and_is_idempotent():
    engine = MetaMorph()
    # Pool is live and usable before close.
    result = engine.execute(ProposedAction(intent="reverse this"))
    assert result.ok
    engine.close()
    assert engine._executor._shutdown is True
    # Idempotent — a second close must not raise.
    engine.close()


def test_context_manager_closes_executor():
    with MetaMorph() as engine:
        assert engine.execute(ProposedAction(intent="reverse this")).ok
    assert engine._executor._shutdown is True


@pytest.mark.skipif(
    sys.implementation.name != "cpython",
    reason="weakref finalizer timing is CPython-specific; other implementations may not synchronously collect on gc.collect()",
)
def test_finalizer_shuts_down_executor_on_gc():
    import gc

    engine = MetaMorph()
    executor = engine._executor
    assert executor._shutdown is False
    del engine
    gc.collect()
    # The weakref finalizer should have shut the pool down once unreferenced.
    assert executor._shutdown is True
