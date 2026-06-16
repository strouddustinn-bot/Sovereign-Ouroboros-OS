"""Tests for the ChronoWeave counterfactual timeline engine."""

from ouroboros.chronoweave import ChronoWeave
from ouroboros.core import (
    Prototype,
    Simulator,
    Timeline,
    WorldState,
    embed,
)


def _prototypes() -> list[Prototype]:
    return [
        Prototype(label="purge approach", latent=embed("purge approach"), confidence=0.8),
        Prototype(label="archive approach", latent=embed("archive approach"), confidence=0.6),
    ]


def test_satisfies_simulator_protocol():
    assert isinstance(ChronoWeave(), Simulator)


def test_simulate_returns_single_timeline():
    weaver = ChronoWeave()
    state = WorldState()
    result = weaver.simulate("summarize the report", _prototypes(), state)
    assert isinstance(result, Timeline)
    assert result.id
    assert result.rationale
    assert result.trajectory


def test_weave_returns_multiple_sorted_timelines():
    weaver = ChronoWeave()
    timelines = weaver.weave("summarize the report", _prototypes(), WorldState())
    # 2 prototypes * 3 branches.
    assert len(timelines) == 6
    scores = [t.score for t in timelines]
    assert scores == sorted(scores, reverse=True)


def test_determinism():
    state = WorldState()
    a = ChronoWeave().simulate("delete the cache", _prototypes(), state)
    b = ChronoWeave().simulate("delete the cache", _prototypes(), state)
    assert a == b

    weave_a = ChronoWeave().weave("delete the cache", _prototypes(), state)
    weave_b = ChronoWeave().weave("delete the cache", _prototypes(), state)
    assert [t.id for t in weave_a] == [t.id for t in weave_b]
    assert [t.score for t in weave_a] == [t.score for t in weave_b]


def test_delete_task_collapses_to_mitigated_branch():
    weaver = ChronoWeave()
    winner = weaver.simulate("delete old records", _prototypes(), WorldState())
    action = winner.proposed_action
    assert action.backup_exists is True
    assert action.confirmed is True
    assert action.audit_logged is True


def test_winning_score_is_max_over_woven_timelines():
    weaver = ChronoWeave()
    task = "delete old records"
    state = WorldState()
    winner = weaver.simulate(task, _prototypes(), state)
    timelines = weaver.weave(task, _prototypes(), state)
    assert winner.score == max(t.score for t in timelines)
    assert winner == timelines[0]


def test_share_task_never_shares_externally_when_mitigated():
    weaver = ChronoWeave()
    winner = weaver.simulate("share the dataset", _prototypes(), WorldState())
    assert winner.proposed_action.shares_external is False
    assert winner.proposed_action.exposes_pii is False


def test_unique_branch_ids():
    timelines = ChronoWeave().weave("delete old records", _prototypes(), WorldState())
    ids = [t.id for t in timelines]
    assert len(ids) == len(set(ids))
