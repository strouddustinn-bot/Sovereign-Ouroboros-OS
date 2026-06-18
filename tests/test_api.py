"""Tests for the Ouroboros FastAPI REST service.

Uses FastAPI's synchronous TestClient (backed by httpx) so no asyncio event
loop wrangling is needed.

IMPORTANT – shared test state
==============================
All tests share the OuroborosLoop instance created at app import time.
This means:
- /history accumulates across tests in the same process.
- Per-tenant isolation tests rely on separate tenant IDs and _get_loop().
- Tests that modify _API_KEY or _rate_buckets must clean up on exit.

Use the ``_patched_api_key()`` context manager and the ``rate_limited_tenant``/
``full_bucket_tenant`` fixtures (conftest.py) to avoid leaving state behind
that could break later tests.

Test sections:
    - GET /health              (no auth required)
    - POST /run                (basic + validation)
    - GET /history
    - GET /state
    - GET /skills
    - GET /principles
    - POST /principles
    - GET /metrics
    - API key authentication
    - Per-tenant isolation
    - Rate limiting
    - WebSocket – basic functionality
    - WebSocket – authentication
    - WebSocket – rate limiting
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from ouroboros.api.app import app, _loop, _get_loop, _rate_buckets, WS_MAX_MESSAGE_BYTES
from ouroboros.ouroboros_loop import DEFAULT_PRINCIPLES

client = TestClient(app)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_API_KEY = "test-secret-key-xyz"
TEST_TASK_BENIGN = "summarize the research notes"
TEST_TASK_BLOCKED = "harm the production database"
# Imported from app so tests stay in sync with the server-side limit.
WS_MESSAGE_MAX_SIZE = WS_MAX_MESSAGE_BYTES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _patched_api_key(key: str | None) -> Iterator[None]:
    """Temporarily patch module-level ``_API_KEY``; restore on exit."""
    import ouroboros.api.app as app_module
    original = app_module._API_KEY
    app_module._API_KEY = key
    try:
        yield
    finally:
        app_module._API_KEY = original


def _fresh_client_with_key(key: str) -> TestClient:
    """Return a TestClient that sends X-API-Key on every request."""
    return TestClient(app, headers={"X-API-Key": key})


def _ws_consume_stages(
    ws,
    expected_final_stage: str = "complete",
    max_frames: int = 50,
) -> list[dict]:
    """Consume stage frames from *ws* until *expected_final_stage* is seen.

    Returns all consumed messages including the terminal one.
    Raises RuntimeError if *max_frames* is exceeded (guards against hangs on
    protocol regressions that never emit the expected final stage).
    """
    messages: list[dict] = []
    while True:
        if len(messages) >= max_frames:
            raise RuntimeError(
                f"_ws_consume_stages: exceeded {max_frames} frames without "
                f"seeing stage={expected_final_stage!r}. Got: {messages}"
            )
        msg = json.loads(ws.receive_text())
        messages.append(msg)
        if msg.get("stage") == expected_final_stage:
            break
    return messages


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


def test_health_never_requires_auth() -> None:
    """/health must return 200 even when an API key is configured."""
    with _patched_api_key("some-key"):
        assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# POST /run
# ---------------------------------------------------------------------------


def test_run_safe_task_succeeds() -> None:
    """A benign task should complete without being blocked."""
    response = client.post("/run", json={"task": TEST_TASK_BENIGN})
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
    response = client.post("/run", json={"task": TEST_TASK_BLOCKED})
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
    assert client.post("/run", json={"task": "valid task", "imagine_k": 0}).status_code == 422
    assert client.post("/run", json={"task": "valid task", "imagine_k": 11}).status_code == 422


def test_run_with_valid_imagine_k() -> None:
    """imagine_k within [1, 10] should be accepted."""
    response = client.post("/run", json={"task": "list files in /tmp", "imagine_k": 5})
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# GET /history
# ---------------------------------------------------------------------------


def test_history_grows_after_run_calls() -> None:
    """History list length should increase after each /run call."""
    baseline = len(client.get("/history").json())

    client.post("/run", json={"task": "count words in document"})
    client.post("/run", json={"task": "reverse the log entries"})

    after = client.get("/history").json()
    assert len(after) == baseline + 2
    entry = after[-1]
    assert "task" in entry
    assert "step" in entry
    assert "succeeded" in entry
    assert "blocked" in entry


# ---------------------------------------------------------------------------
# GET /state
# ---------------------------------------------------------------------------


def test_get_state_has_step_key() -> None:
    """GET /state must return a dict with a numeric 'step' field and 'facts'."""
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
    """GET /principles must include all DEFAULT_PRINCIPLES names."""
    response = client.get("/principles")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    expected_names = {p[:60].rstrip() for p in DEFAULT_PRINCIPLES}
    assert expected_names.issubset(set(data))


# ---------------------------------------------------------------------------
# POST /principles
# ---------------------------------------------------------------------------


def test_post_principles_adds_to_list() -> None:
    """POST /principles should add new principles visible via GET /principles."""
    new_principles = ["Always ask before sending emails.", "Never overwrite prod data."]
    add_resp = client.post("/principles", json={"principles": new_principles})
    assert add_resp.status_code == 200
    body = add_resp.json()
    assert "added" in body
    assert set(body["added"]) == set(new_principles)

    names = client.get("/principles").json()
    for p in new_principles:
        assert p[:60].rstrip() in names


def test_post_principles_empty_list_returns_422() -> None:
    """An empty principles list violates min_length=1 — expect 422."""
    assert client.post("/principles", json={"principles": []}).status_code == 422


def test_post_principles_too_many_returns_422() -> None:
    """More than 20 principles should be rejected with 422."""
    response = client.post(
        "/principles",
        json={"principles": [f"principle {i}" for i in range(21)]},
    )
    assert response.status_code == 422


def test_post_principles_missing_field_returns_422() -> None:
    """Omitting the 'principles' key should return 422."""
    assert client.post("/principles", json={"principle": "single string"}).status_code == 422


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


def test_metrics_endpoint_returns_stats() -> None:
    """GET /metrics must return per-tenant loop metrics with expected integer fields."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    expected_keys = {"loop_step", "history_count", "skill_count", "principle_count"}
    assert expected_keys <= data.keys()
    assert all(isinstance(data[k], int) for k in expected_keys)


def test_metrics_requires_auth() -> None:
    """GET /metrics must return 401 when a key is configured but not supplied."""
    with _patched_api_key(TEST_API_KEY):
        assert client.get("/metrics").status_code == 401


def test_metrics_is_per_tenant() -> None:
    """Each tenant sees their own history count, not another tenant's."""
    key_a = "metrics-tenant-alpha"
    key_b = "metrics-tenant-beta"

    loop_a = _get_loop(key_a)
    loop_b = _get_loop(key_b)
    loop_a.history.clear()
    loop_b.history.clear()

    with _patched_api_key(key_a):
        client.post("/run", json={"task": "alpha-only task"}, headers={"X-API-Key": key_a})
        resp_a = client.get("/metrics", headers={"X-API-Key": key_a})
    assert resp_a.status_code == 200
    assert resp_a.json()["history_count"] == 1

    with _patched_api_key(key_b):
        resp_b = client.get("/metrics", headers={"X-API-Key": key_b})
    assert resp_b.status_code == 200
    assert resp_b.json()["history_count"] == 0


# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------


def test_dev_mode_no_key_needed() -> None:
    """In dev mode (OUROBOROS_API_KEY unset) all requests pass without a key."""
    assert client.get("/state").status_code == 200


def test_auth_with_api_key_set() -> None:
    """When _API_KEY is set, missing/wrong/correct keys are gated correctly."""
    with _patched_api_key(TEST_API_KEY):
        assert client.get("/state").status_code == 401
        assert client.get("/state", headers={"X-API-Key": TEST_API_KEY}).status_code == 200
        assert client.get("/state", headers={"X-API-Key": "wrong-key"}).status_code == 401


# ---------------------------------------------------------------------------
# Per-tenant isolation
# ---------------------------------------------------------------------------


def test_per_tenant_history_isolation() -> None:
    """Different tenant IDs (API keys) must have separate independent histories."""
    key_a, key_b = "tenant-alpha", "tenant-beta"

    loop_a = _get_loop(key_a)
    loop_b = _get_loop(key_b)
    loop_a.history.clear()
    loop_b.history.clear()

    with _patched_api_key(key_a):
        client.post("/run", json={"task": "tenant alpha task"}, headers={"X-API-Key": key_a})

    with _patched_api_key(key_b):
        client.post("/run", json={"task": "tenant beta task"}, headers={"X-API-Key": key_b})

    assert len(loop_a.history) == 1
    assert loop_a.history[0].task == "tenant alpha task"
    assert len(loop_b.history) == 1
    assert loop_b.history[0].task == "tenant beta task"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limit_returns_429_after_exhaustion(rate_limited_tenant: str) -> None:
    """After exhausting the rate bucket a 429 with Retry-After header is returned."""
    with _patched_api_key(rate_limited_tenant):
        resp = client.get("/state", headers={"X-API-Key": rate_limited_tenant})
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


def test_rate_limit_not_hit_with_full_bucket(full_bucket_tenant: str) -> None:
    """A full rate bucket should allow the request through."""
    with _patched_api_key(full_bucket_tenant):
        resp = client.get("/state", headers={"X-API-Key": full_bucket_tenant})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# WebSocket – basic functionality
# ---------------------------------------------------------------------------


def test_ws_rejects_oversized_message() -> None:
    """Messages exceeding WS_MESSAGE_MAX_SIZE bytes must receive an error frame."""
    with client.websocket_connect("/ws/run") as ws:
        oversized = json.dumps({"task": "x" * (WS_MESSAGE_MAX_SIZE + 1)})
        ws.send_text(oversized)
        data = json.loads(ws.receive_text())
        assert "error" in data
        assert "too large" in data["error"].lower() or str(WS_MESSAGE_MAX_SIZE) in data["error"]


def test_ws_accepts_normal_message() -> None:
    """A well-formed WS message should produce all cognitive stage frames."""
    with client.websocket_connect("/ws/run") as ws:
        ws.send_text(json.dumps({"task": "list files"}))
        messages = _ws_consume_stages(ws)
    stage_names = [m.get("stage", "") for m in messages]
    assert "neurosynth" in stage_names
    assert "complete" in stage_names


# ---------------------------------------------------------------------------
# WebSocket – authentication
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "api_key,should_pass",
    [
        (None, False),          # missing api_key in payload
        ("wrong-key", False),   # incorrect api_key
        (TEST_API_KEY, True),   # correct api_key
    ],
)
def test_ws_api_key_validation(api_key: str | None, should_pass: bool) -> None:
    """WS auth: missing/wrong key → unauthorized frame + disconnect; correct key → stages."""
    with _patched_api_key(TEST_API_KEY):
        with client.websocket_connect("/ws/run") as ws:
            payload: dict = {"task": "echo hello"}
            if api_key is not None:
                payload["api_key"] = api_key
            ws.send_text(json.dumps(payload))

            if not should_pass:
                first = json.loads(ws.receive_text())
                assert "error" in first
                assert "unauthorized" in first["error"].lower()
                with pytest.raises(WebSocketDisconnect):
                    ws.receive_text()
            else:
                messages = _ws_consume_stages(ws)
                assert all("error" not in m for m in messages), (
                    f"Unexpected error frame: {messages}"
                )
                stage_names = [m.get("stage", "") for m in messages]
                assert "neurosynth" in stage_names
                assert "complete" in stage_names
                final = messages[-1]
                assert final.get("stage") == "complete"
                assert "succeeded" in final


# ---------------------------------------------------------------------------
# WebSocket – rate limiting
# ---------------------------------------------------------------------------


def test_ws_rate_limit_returns_error_frame(rate_limited_tenant: str) -> None:
    """A rate-limited tenant must receive a rate-limit error frame over WS."""
    with _patched_api_key(rate_limited_tenant):
        with client.websocket_connect("/ws/run") as ws:
            ws.send_text(json.dumps({"task": "any task", "api_key": rate_limited_tenant}))
            data = json.loads(ws.receive_text())
            assert "error" in data
            assert "rate" in data["error"].lower() or "limit" in data["error"].lower()
            assert "retry_after" in data
