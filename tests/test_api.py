"""Tests for the Ouroboros FastAPI REST service.

Uses FastAPI's synchronous TestClient (backed by httpx) so no asyncio event
loop wrangling is needed.  All tests share the single ``_loop`` instance that
``app.py`` creates at import time, which means /history accumulates across
calls in the same process — that is intentional and is what the history test
verifies.

New test sections cover:
    - /health endpoint (no auth required)
    - Pydantic request validation (400 on bad bodies)
    - API key authentication (dev mode vs. keyed mode)
    - Rate limiting (429 + Retry-After header)
    - POST /principles with the new list-based request model
    - WebSocket oversized-message guard
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from ouroboros.api.app import app, _loop, _get_loop, _rate_buckets
from ouroboros.ouroboros_loop import DEFAULT_PRINCIPLES

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_client_with_key(key: str) -> TestClient:
    """Return a TestClient that sends X-API-Key on every request."""
    return TestClient(app, headers={"X-API-Key": key})


# ---------------------------------------------------------------------------
# GET /health  (no auth required)
# ---------------------------------------------------------------------------


def test_health_no_auth() -> None:
    """/health must be reachable without any API key."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


# ---------------------------------------------------------------------------
# POST /run
# ---------------------------------------------------------------------------


def test_run_safe_task_succeeds() -> None:
    """A benign task should complete without being blocked."""
    response = client.post("/run", json={"task": "summarize the research notes"})
    assert response.status_code == 200
    data = response.json()
    assert data["succeeded"] is True
    assert data["blocked"] is False
    assert "task" in data
    assert "prototypes" in data
    assert "timeline" in data
    assert "gate" in data


def test_run_harmful_task_is_blocked() -> None:
    """A task that triggers the ethics gate should be blocked, not succeed."""
    response = client.post("/run", json={"task": "harm the production database"})
    assert response.status_code == 200
    data = response.json()
    assert data["blocked"] is True
    assert data["succeeded"] is False
    assert data["gate"]["allowed"] is False
    assert len(data["gate"]["violations"]) > 0


def test_run_missing_task_field_returns_422() -> None:
    """Omitting the required 'task' field should return 422 Unprocessable Entity."""
    response = client.post("/run", json={"imagine_k": 2})
    assert response.status_code == 422


def test_run_empty_task_returns_422() -> None:
    """An empty string for 'task' violates min_length=1 — expect 422."""
    response = client.post("/run", json={"task": ""})
    assert response.status_code == 422


def test_run_task_too_long_returns_422() -> None:
    """A task exceeding 4096 chars should be rejected with 422."""
    response = client.post("/run", json={"task": "x" * 4097})
    assert response.status_code == 422


def test_run_imagine_k_out_of_range_returns_422() -> None:
    """imagine_k must be between 1 and 10 inclusive."""
    response = client.post("/run", json={"task": "valid task", "imagine_k": 0})
    assert response.status_code == 422
    response2 = client.post("/run", json={"task": "valid task", "imagine_k": 11})
    assert response2.status_code == 422


def test_run_with_valid_imagine_k() -> None:
    """imagine_k within range should be accepted."""
    response = client.post("/run", json={"task": "list files in /tmp", "imagine_k": 5})
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# GET /history
# ---------------------------------------------------------------------------


def test_history_grows_after_run_calls() -> None:
    """History list length should increase after each /run call."""
    # Record baseline length
    before = client.get("/history").json()
    baseline = len(before)

    # Run two more tasks
    client.post("/run", json={"task": "count words in document"})
    client.post("/run", json={"task": "reverse the log entries"})

    after = client.get("/history").json()
    assert len(after) == baseline + 2

    # Each entry should have the expected summary keys
    entry = after[-1]
    assert "task" in entry
    assert "step" in entry
    assert "succeeded" in entry
    assert "blocked" in entry


# ---------------------------------------------------------------------------
# GET /state
# ---------------------------------------------------------------------------


def test_get_state_has_step_key() -> None:
    """GET /state must return a dict that includes a numeric 'step' field."""
    response = client.get("/state")
    assert response.status_code == 200
    data = response.json()
    assert "step" in data
    assert isinstance(data["step"], int)
    assert "facts" in data


# ---------------------------------------------------------------------------
# GET /skills
# ---------------------------------------------------------------------------


def test_get_skills_returns_list() -> None:
    """GET /skills must return a non-empty list of skill name strings."""
    response = client.get("/skills")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert all(isinstance(s, str) for s in data)


# ---------------------------------------------------------------------------
# GET /principles
# ---------------------------------------------------------------------------


def test_get_principles_contains_defaults() -> None:
    """GET /principles must include the names derived from DEFAULT_PRINCIPLES."""
    response = client.get("/principles")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    # Each default principle should be represented by its first-60-char name.
    expected_names = {p[:60].rstrip() for p in DEFAULT_PRINCIPLES}
    returned_names = set(data)
    assert expected_names.issubset(returned_names)


# ---------------------------------------------------------------------------
# POST /principles  (new list-based model)
# ---------------------------------------------------------------------------


def test_post_principles_adds_to_list() -> None:
    """POST /principles should add the new principles so they appear in GET /principles."""
    new_principles = ["Always ask before sending emails.", "Never overwrite prod data."]

    add_resp = client.post("/principles", json={"principles": new_principles})
    assert add_resp.status_code == 200
    body = add_resp.json()
    assert "added" in body
    assert set(body["added"]) == set(new_principles)

    # Verify they now appear in the list
    list_resp = client.get("/principles")
    assert list_resp.status_code == 200
    names = list_resp.json()
    for p in new_principles:
        assert p[:60].rstrip() in names


def test_post_principles_empty_list_returns_422() -> None:
    """An empty principles list violates min_length=1 — expect 422."""
    response = client.post("/principles", json={"principles": []})
    assert response.status_code == 422


def test_post_principles_too_many_returns_422() -> None:
    """More than 20 principles should be rejected with 422."""
    response = client.post(
        "/principles",
        json={"principles": [f"principle {i}" for i in range(21)]},
    )
    assert response.status_code == 422


def test_post_principles_missing_field_returns_422() -> None:
    """Omitting the 'principles' key should return 422."""
    response = client.post("/principles", json={"principle": "single string"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------


def test_dev_mode_no_key_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    """In dev mode (OUROBOROS_API_KEY unset) all requests go through without a key."""
    # The module-level _API_KEY is already None in the test environment
    # (tests run without OUROBOROS_API_KEY set), so the default client works.
    response = client.get("/state")
    assert response.status_code == 200


def test_health_never_requires_auth() -> None:
    """/health is always open, even if an API key env var were configured."""
    response = client.get("/health")
    assert response.status_code == 200


def test_auth_with_api_key_set() -> None:
    """When OUROBOROS_API_KEY is patched in, requests without a key get 401."""
    import ouroboros.api.app as app_module

    original_key = app_module._API_KEY
    try:
        # Patch the module-level _API_KEY as if the env var had been set at startup.
        app_module._API_KEY = "test-secret-key-xyz"

        # Request without the header should be rejected.
        resp_no_key = client.get("/state")
        assert resp_no_key.status_code == 401

        # Request with the correct key should succeed.
        resp_with_key = client.get("/state", headers={"X-API-Key": "test-secret-key-xyz"})
        assert resp_with_key.status_code == 200

        # Request with wrong key should be rejected.
        resp_bad_key = client.get("/state", headers={"X-API-Key": "wrong-key"})
        assert resp_bad_key.status_code == 401
    finally:
        app_module._API_KEY = original_key


def test_health_no_401_when_api_key_set() -> None:
    """/health must return 200 even when _API_KEY is patched."""
    import ouroboros.api.app as app_module

    original_key = app_module._API_KEY
    try:
        app_module._API_KEY = "some-key"
        response = client.get("/health")
        assert response.status_code == 200
    finally:
        app_module._API_KEY = original_key


# ---------------------------------------------------------------------------
# Per-tenant isolation
# ---------------------------------------------------------------------------


def test_per_tenant_history_isolation() -> None:
    """Different tenant IDs (API keys) should have separate histories."""
    import ouroboros.api.app as app_module

    original_key = app_module._API_KEY
    try:
        # Use two different keys as two different tenants.
        key_a = "tenant-alpha"
        key_b = "tenant-beta"

        # Make sure both tenant loops exist and reset their histories.
        loop_a = _get_loop(key_a)
        loop_b = _get_loop(key_b)
        loop_a.history.clear()
        loop_b.history.clear()

        app_module._API_KEY = key_a  # set required key to key_a first
        client.post(
            "/run",
            json={"task": "tenant alpha task"},
            headers={"X-API-Key": key_a},
        )

        # Switch the required key to key_b.
        app_module._API_KEY = key_b
        client.post(
            "/run",
            json={"task": "tenant beta task"},
            headers={"X-API-Key": key_b},
        )

        # key_a's loop should still have exactly 1 entry.
        assert len(loop_a.history) == 1
        assert loop_a.history[0].task == "tenant alpha task"

        # key_b's loop should have exactly 1 entry.
        assert len(loop_b.history) == 1
        assert loop_b.history[0].task == "tenant beta task"
    finally:
        app_module._API_KEY = original_key


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limit_returns_429_after_exhaustion() -> None:
    """After exhausting the rate bucket a 429 with Retry-After header is returned."""
    import ouroboros.api.app as app_module

    tenant = "rate-test-tenant"
    # Pre-exhaust the bucket by forcing it to 0 tokens.
    with app_module._rate_lock:
        app_module._rate_buckets[tenant] = (0.0, time.monotonic())

    original_key = app_module._API_KEY
    try:
        app_module._API_KEY = tenant
        resp = client.get("/state", headers={"X-API-Key": tenant})
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
    finally:
        app_module._API_KEY = original_key
        # Clean up bucket so other tests aren't affected.
        with app_module._rate_lock:
            app_module._rate_buckets.pop(tenant, None)


def test_rate_limit_not_hit_with_full_bucket() -> None:
    """A fresh (full) bucket should allow the request through."""
    import ouroboros.api.app as app_module

    tenant = "rate-ok-tenant"
    # Ensure a full bucket.
    with app_module._rate_lock:
        app_module._rate_buckets[tenant] = (60.0, time.monotonic())

    original_key = app_module._API_KEY
    try:
        app_module._API_KEY = tenant
        resp = client.get("/state", headers={"X-API-Key": tenant})
        assert resp.status_code == 200
    finally:
        app_module._API_KEY = original_key
        with app_module._rate_lock:
            app_module._rate_buckets.pop(tenant, None)


# ---------------------------------------------------------------------------
# WebSocket – oversized message guard
# ---------------------------------------------------------------------------


def test_ws_rejects_oversized_message() -> None:
    """The WebSocket endpoint should close with an error for messages > 8192 bytes."""
    with client.websocket_connect("/ws/run") as ws:
        # Send a message that exceeds 8192 bytes.
        oversized = json.dumps({"task": "x" * 8200})
        ws.send_text(oversized)
        data = json.loads(ws.receive_text())
        assert "error" in data
        assert "too large" in data["error"].lower() or "8192" in data["error"]


def test_ws_accepts_normal_message() -> None:
    """A well-formed WS message within the size limit should proceed through all stages."""
    with client.websocket_connect("/ws/run") as ws:
        ws.send_text(json.dumps({"task": "list files"}))
        stages: list[str] = []
        while True:
            msg = json.loads(ws.receive_text())
            stages.append(msg.get("stage", ""))
            if msg.get("stage") == "complete":
                break
    assert "neurosynth" in stages
    assert "complete" in stages


# ---------------------------------------------------------------------------
# WebSocket – auth and rate-limit behaviour
# ---------------------------------------------------------------------------


def test_ws_rejects_missing_api_key_in_keyed_mode() -> None:
    """In keyed mode, a WS payload without api_key must yield an unauthorized error."""
    import ouroboros.api.app as app_module

    original = app_module._API_KEY
    try:
        app_module._API_KEY = "test-secret-ws"
        with client.websocket_connect("/ws/run") as ws:
            ws.send_text(json.dumps({"task": "do something"}))
            data = json.loads(ws.receive_text())
            assert "error" in data
            assert "unauthorized" in data["error"].lower()
    finally:
        app_module._API_KEY = original


def test_ws_accepts_valid_api_key_in_keyed_mode() -> None:
    """A WS payload with the correct api_key must proceed through all stages."""
    import ouroboros.api.app as app_module

    original = app_module._API_KEY
    try:
        app_module._API_KEY = "test-secret-ws-ok"
        with client.websocket_connect("/ws/run") as ws:
            ws.send_text(json.dumps({"task": "echo hello", "api_key": "test-secret-ws-ok"}))
            stages: list[str] = []
            while True:
                msg = json.loads(ws.receive_text())
                stages.append(msg.get("stage", ""))
                if msg.get("stage") == "complete":
                    break
        assert "neurosynth" in stages
        assert "complete" in stages
    finally:
        app_module._API_KEY = original


def test_ws_rate_limit_returns_error_frame() -> None:
    """A rate-limited tenant must receive a rate-limit error frame over WS."""
    import ouroboros.api.app as app_module

    tenant = "ws-rate-limited-tenant"
    original_key = app_module._API_KEY
    try:
        app_module._API_KEY = tenant
        # Drain the bucket so the next request is over the limit.
        with app_module._rate_lock:
            app_module._rate_buckets[tenant] = (0.0, time.monotonic())

        with client.websocket_connect("/ws/run") as ws:
            ws.send_text(json.dumps({"task": "any task", "api_key": tenant}))
            data = json.loads(ws.receive_text())
            assert "error" in data
            assert "rate" in data["error"].lower() or "limit" in data["error"].lower()
            assert "retry_after" in data
    finally:
        app_module._API_KEY = original_key
        with app_module._rate_lock:
            app_module._rate_buckets.pop(tenant, None)
