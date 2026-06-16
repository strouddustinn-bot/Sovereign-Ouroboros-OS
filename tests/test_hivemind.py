"""Tests for the HiveMind federated-intelligence layer."""

from ouroboros.core.contracts import Expander
from ouroboros.core.types import FederatedResult
from ouroboros.hivemind import HiveMind


def test_satisfies_expander_protocol():
    assert isinstance(HiveMind(), Expander)


def test_round_trip_ascii():
    hive = HiveMind(n_peers=5)
    task = "optimize the global routing table"
    assert hive.reconstruct(hive.shard(task)) == task


def test_round_trip_unicode():
    hive = HiveMind(n_peers=4)
    task = "fragmenter le problème 🜂 над сетью узлов"
    assert hive.reconstruct(hive.shard(task)) == task


def test_round_trip_single_peer():
    hive = HiveMind(n_peers=1)
    task = "lonely sovereign node"
    assert hive.reconstruct(hive.shard(task)) == task


def test_privacy_no_share_equals_plaintext():
    hive = HiveMind(n_peers=5)
    task = "the secret payload no peer may see"
    raw = task.encode("utf-8")
    shares = hive.shard(task)

    # No individual share is the plaintext...
    for share in shares:
        assert share != raw
    # ...and at least one share genuinely differs from the plaintext bytes.
    assert any(share != raw for share in shares)


def test_shares_are_non_trivial_and_correct_length():
    hive = HiveMind(n_peers=3)
    task = "distributed cognition"
    raw = task.encode("utf-8")
    shares = hive.shard(task)

    assert len(shares) == 3
    for share in shares:
        assert len(share) == len(raw)


def test_expand_returns_federated_result_shape():
    hive = HiveMind(n_peers=5)
    result = hive.expand("synthesize a consensus answer", seed=0)

    assert isinstance(result, FederatedResult)
    assert result.task == "synthesize a consensus answer"
    assert result.shards == 5
    assert result.contributors == [f"peer-{i}" for i in range(5)]


def test_expand_synthesizes_collective_answer():
    hive = HiveMind(n_peers=4)
    result = hive.expand("derive a plan", seed=7)

    assert "collective_answer" in result.reconstructed
    assert result.reconstructed["peers"] == 4
    assert len(result.reconstructed["contributions"]) == 4
    assert 0.0 <= result.reconstructed["confidence"] <= 1.0


def test_expand_is_deterministic():
    a = HiveMind(n_peers=5).expand("reproducible task", seed=0)
    b = HiveMind(n_peers=5).expand("reproducible task", seed=0)
    assert a == b


def test_expand_integrity_check_recovers_task():
    hive = HiveMind(n_peers=6)
    task = "integrity must round-trip ✓"
    result = hive.expand(task, seed=1)
    # The internal integrity assertion passed, and the synthesized aggregate
    # is bound to the original task.
    assert result.task == task
    assert hive.reconstruct(hive.shard(task)) == task


def test_peers_never_see_plaintext():
    hive = HiveMind(n_peers=5)
    task = "no peer reconstructs the whole"
    raw = task.encode("utf-8")
    shares = hive.shard(task)
    for peer, share in zip(hive.peers, shares):
        partial = peer.compute(share)
        # The partial carries only derived metrics, never the raw task.
        assert raw not in str(partial).encode("utf-8")
        assert partial["share_bytes"] == len(raw)
