"""Tests for the NetworkHiveMind real-TCP federation layer."""

from __future__ import annotations

import asyncio

import pytest

from sovereign_ouroboros_os.core.types import FederatedResult
from sovereign_ouroboros_os.hivemind import NetworkHiveMind
from sovereign_ouroboros_os.hivemind.federation import HiveMind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Execute *coro* in a fresh asyncio event loop."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Basic shape tests (via expand_sync)
# ---------------------------------------------------------------------------


class TestExpandSyncShape:
    """expand_sync must return a properly structured FederatedResult."""

    def test_returns_federated_result(self):
        hive = NetworkHiveMind(n_peers=3)
        result = hive.expand_sync("plan the mission", seed=0)
        assert isinstance(result, FederatedResult)

    def test_shards_equals_n_peers(self):
        for n in (1, 3, 5):
            hive = NetworkHiveMind(n_peers=n)
            result = hive.expand_sync("shards test", seed=0)
            assert result.shards == n, f"expected {n} shards, got {result.shards}"

    def test_contributors_contains_all_peer_ids(self):
        n = 4
        hive = NetworkHiveMind(n_peers=n)
        result = hive.expand_sync("contributors test", seed=0)
        expected = [f"peer-{i}" for i in range(n)]
        assert result.contributors == expected

    def test_task_field_preserved(self):
        task = "the task string must survive the network round-trip"
        result = NetworkHiveMind(n_peers=3).expand_sync(task, seed=0)
        assert result.task == task

    def test_reconstructed_has_collective_answer(self):
        result = NetworkHiveMind(n_peers=3).expand_sync("collective answer key", seed=0)
        assert "collective_answer" in result.reconstructed

    def test_reconstructed_peer_count(self):
        n = 4
        result = NetworkHiveMind(n_peers=n).expand_sync("peer count", seed=0)
        assert result.reconstructed["peers"] == n

    def test_reconstructed_contributions_length(self):
        n = 3
        result = NetworkHiveMind(n_peers=n).expand_sync("contributions length", seed=0)
        assert len(result.reconstructed["contributions"]) == n

    def test_confidence_in_unit_interval(self):
        result = NetworkHiveMind(n_peers=5).expand_sync("confidence", seed=0)
        assert 0.0 <= result.reconstructed["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same task + seed must produce the same result within one call."""

    def test_same_task_same_result(self):
        task = "deterministic task"
        a = NetworkHiveMind(n_peers=3).expand_sync(task, seed=7)
        b = NetworkHiveMind(n_peers=3).expand_sync(task, seed=7)
        assert a == b

    def test_different_seeds_differ(self):
        task = "seed sensitivity"
        a = NetworkHiveMind(n_peers=3).expand_sync(task, seed=0)
        b = NetworkHiveMind(n_peers=3).expand_sync(task, seed=99)
        # collective_answer folds the seed in, so results should differ.
        assert a.reconstructed["collective_answer"] != b.reconstructed["collective_answer"]

    def test_different_tasks_differ(self):
        a = NetworkHiveMind(n_peers=3).expand_sync("task alpha", seed=0)
        b = NetworkHiveMind(n_peers=3).expand_sync("task beta", seed=0)
        assert a.reconstructed["collective_answer"] != b.reconstructed["collective_answer"]


# ---------------------------------------------------------------------------
# Secret-sharing round-trip
# ---------------------------------------------------------------------------


class TestSecretSharingRoundTrip:
    """The shares produced by NetworkHiveMind must still satisfy the
    additive secret-sharing guarantee (XOR reconstruction)."""

    def test_reconstruct_recovers_task(self):
        task = "reconstruct me please"
        hive = NetworkHiveMind(n_peers=5)
        inner = HiveMind(n_peers=5)
        shares = inner.shard(task)
        assert inner.reconstruct(shares) == task

    def test_network_expand_passes_integrity_check(self):
        """expand_sync would raise AssertionError if reconstruction failed."""
        task = "integrity check over TCP ✓"
        result = NetworkHiveMind(n_peers=4).expand_sync(task, seed=1)
        assert result.task == task

    def test_unicode_round_trip(self):
        task = "fragmenter le problème 🜂 над сетью узлов"
        result = NetworkHiveMind(n_peers=3).expand_sync(task, seed=0)
        assert result.task == task
        assert result.shards == 3


# ---------------------------------------------------------------------------
# Connection count
# ---------------------------------------------------------------------------


class TestConnectionCount:
    """Exactly n_peers TCP connections should be used per expand call."""

    def test_exactly_n_peers_connections_used(self):
        n = 5
        result = NetworkHiveMind(n_peers=n).expand_sync("connection count", seed=0)
        # Each contributor corresponds to one peer that handled one connection.
        assert len(result.contributors) == n

    def test_contributors_are_unique(self):
        n = 5
        result = NetworkHiveMind(n_peers=n).expand_sync("unique peers", seed=0)
        assert len(set(result.contributors)) == n

    def test_three_peers(self):
        result = NetworkHiveMind(n_peers=3).expand_sync("three peers", seed=0)
        assert len(result.contributors) == 3
        assert result.contributors == ["peer-0", "peer-1", "peer-2"]


# ---------------------------------------------------------------------------
# Lifecycle: async context manager
# ---------------------------------------------------------------------------


class TestAsyncContextManager:
    """NetworkHiveMind must work as an async context manager."""

    def test_async_context_manager(self):
        async def _run_hive():
            async with NetworkHiveMind(n_peers=3) as hive:
                return await hive.expand("async ctx manager", seed=0)

        result = asyncio.run(_run_hive())
        assert isinstance(result, FederatedResult)
        assert result.shards == 3

    def test_servers_stopped_after_async_exit(self):
        async def _run():
            hive = NetworkHiveMind(n_peers=2)
            async with hive:
                await hive.expand("lifecycle test", seed=0)
            # After __aexit__ the server list must be empty.
            assert hive._servers == []
            assert hive._peer_addresses == []

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Lifecycle: sync context manager
# ---------------------------------------------------------------------------


class TestSyncContextManager:
    """NetworkHiveMind must work as a synchronous context manager."""

    def test_sync_context_manager(self):
        with NetworkHiveMind(n_peers=3) as hive:
            result = hive.expand_sync("sync ctx manager", seed=0)
        assert isinstance(result, FederatedResult)
        assert result.shards == 3

    def test_servers_stopped_after_sync_exit(self):
        hive = NetworkHiveMind(n_peers=2)
        with hive:
            hive.expand_sync("sync lifecycle", seed=0)
        assert hive._servers == []
        assert hive._peer_addresses == []


# ---------------------------------------------------------------------------
# Multiple sequential expand calls
# ---------------------------------------------------------------------------


class TestMultipleCalls:
    """The hive should handle multiple sequential expand calls correctly."""

    def test_multiple_expand_sync_calls(self):
        results = []
        for i in range(3):
            r = NetworkHiveMind(n_peers=3).expand_sync(f"task-{i}", seed=i)
            results.append(r)
        assert len(results) == 3
        # All collective answers should differ because tasks differ.
        answers = {r.reconstructed["collective_answer"] for r in results}
        assert len(answers) == 3

    def test_reuse_via_async_context(self):
        async def _run():
            async with NetworkHiveMind(n_peers=3) as hive:
                r1 = await hive.expand("first call", seed=0)
                r2 = await hive.expand("second call", seed=0)
            return r1, r2

        r1, r2 = asyncio.run(_run())
        assert r1.task == "first call"
        assert r2.task == "second call"
        assert r1.reconstructed["collective_answer"] != r2.reconstructed["collective_answer"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_requires_at_least_one_peer(self):
        with pytest.raises(ValueError):
            NetworkHiveMind(n_peers=0)
