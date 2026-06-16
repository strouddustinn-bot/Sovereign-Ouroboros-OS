"""End-to-end tests for the integrated Ouroboros loop."""

from sovereign_ouroboros_os import OuroborosLoop
from sovereign_ouroboros_os.ouroboros_loop import DEFAULT_PRINCIPLES


def test_full_loop_runs_end_to_end():
    loop = OuroborosLoop()
    result = loop.run("summarize the research notes")

    assert result.task == "summarize the research notes"
    assert len(result.prototypes) == loop.imagine_k
    assert result.timeline.proposed_action.intent
    assert result.gate.allowed
    assert result.execution is not None and result.execution.ok
    assert result.federation is not None
    assert result.federation.shards == loop.n_peers
    assert result.succeeded


def test_loop_blocks_unethical_action():
    # A harmful task must be stopped at the EthosCompiler stage; nothing runs.
    loop = OuroborosLoop()
    result = loop.run("harm the production database")

    assert result.blocked
    assert not result.gate.allowed
    assert result.execution is None
    assert result.federation is None
    assert not result.succeeded


def test_delete_task_is_mitigated_then_permitted():
    # ChronoWeave should collapse to the mitigated branch (backup + confirm),
    # which the EthosCompiler then permits.
    loop = OuroborosLoop()
    result = loop.run("delete the stale build cache")

    action = result.timeline.proposed_action
    assert action.backup_exists is True
    assert action.confirmed is True
    assert result.gate.allowed
    assert result.succeeded


def test_loop_is_deterministic():
    a = OuroborosLoop().run("design a compression scheme")
    b = OuroborosLoop().run("design a compression scheme")
    assert a.timeline.proposed_action.intent == b.timeline.proposed_action.intent
    assert a.timeline.score == b.timeline.score
    assert a.execution.output == b.execution.output
    assert a.federation.reconstructed == b.federation.reconstructed


def test_state_evolves_across_turns():
    loop = OuroborosLoop()
    results = loop.run_many(
        ["index the corpus", "compress the index"]
    )
    assert [r.task for r in results] == ["index the corpus", "compress the index"]
    # Two successful turns advance the world-state step counter.
    assert loop.state.step == 2
    assert len(loop.history) == 2


def test_default_principles_are_loaded():
    loop = OuroborosLoop()
    assert len(loop.ethos.principles) == len(DEFAULT_PRINCIPLES)


def test_validate_shortcut_still_works():
    # Backwards-compatible single-stage gate.
    loop = OuroborosLoop(principles=["Do not harm users."])
    assert loop.validate({"intent": "assist the user"}).allowed
    assert not loop.validate({"intent": "harm the user"}).allowed
