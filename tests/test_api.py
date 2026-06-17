"""Tests for the Ouroboros FastAPI REST service.

Uses FastAPI's synchronous TestClient (backed by httpx) so no asyncio event
loop wrangling is needed.  All tests share the single ``_loop`` instance that
``app.py`` creates at import time, which means /history accumulates across
calls — that is intentional and is what the history test verifies.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from ouroboros.api.app import app, _loop
from ouroboros.ouroboros_loop import DEFAULT_PRINCIPLES

client = TestClient(app)


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
# POST /principles
# ---------------------------------------------------------------------------


def test_post_principle_adds_to_list() -> None:
    """POST /principles should add the new principle so it appears in GET /principles."""
    new_principle = "Always ask before sending emails."

    # Add it
    add_resp = client.post("/principles", json={"principle": new_principle})
    assert add_resp.status_code == 200
    assert add_resp.json()["added"] == new_principle

    # Verify it now appears in the list
    list_resp = client.get("/principles")
    assert list_resp.status_code == 200
    names = list_resp.json()
    expected_name = new_principle[:60].rstrip()
    assert expected_name in names
