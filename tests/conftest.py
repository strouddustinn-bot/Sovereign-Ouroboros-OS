"""Shared pytest fixtures for the Ouroboros test suite."""

from __future__ import annotations

import time

import pytest

from ouroboros.api.app import _rate_buckets, _rate_lock


@pytest.fixture
def rate_limited_tenant():
    """Yield a tenant ID whose rate bucket is exhausted; restore on teardown."""
    tenant_id = "fixture-rate-limited"
    with _rate_lock:
        _rate_buckets[tenant_id] = (0.0, time.monotonic())
    yield tenant_id
    with _rate_lock:
        _rate_buckets.pop(tenant_id, None)


@pytest.fixture
def full_bucket_tenant():
    """Yield a tenant ID with a full rate bucket; restore on teardown."""
    tenant_id = "fixture-full-bucket"
    with _rate_lock:
        _rate_buckets[tenant_id] = (60.0, time.monotonic())
    yield tenant_id
    with _rate_lock:
        _rate_buckets.pop(tenant_id, None)
