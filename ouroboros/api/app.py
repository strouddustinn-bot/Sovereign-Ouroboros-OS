"""FastAPI application exposing the Ouroboros cognitive loop over HTTP + WebSocket.

REST endpoints
--------------
GET  /health           – liveness probe (no auth required)
POST /run              – execute one full loop cycle, returns LoopResult as JSON
GET  /history          – list of past run summaries
GET  /state            – current WorldState (step + facts)
GET  /skills           – registered MetaMorph skill names
GET  /principles       – active EthicalPrinciple names
POST /principles       – hot-load new principles

WebSocket
---------
GET  /ws/run           – stream one JSON message per cognitive stage, then summary
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import threading
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from ouroboros.core.logging import get_logger
from ouroboros.core.types import (
    ExecutionResult,
    FederatedResult,
    Prototype,
    Timeline,
)
from ouroboros.ethos_compiler import GateResult
from ouroboros.ouroboros_loop import (
    DEFAULT_PRINCIPLES,
    LoopResult,
    OuroborosLoop,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_API_KEY: str | None = os.environ.get("OUROBOROS_API_KEY") or None

_raw_cors = os.environ.get("OUROBOROS_CORS_ORIGINS", "")
if _raw_cors.strip():
    _CORS_ORIGINS: list[str] = [o.strip() for o in _raw_cors.split(",") if o.strip()]
else:
    _CORS_ORIGINS = ["http://localhost:3000", "http://localhost:8080"]

# Explicit dev-mode flag: set OUROBOROS_DEV_MODE=1 in local environments.
# Using a dedicated flag (rather than inferring from API key absence) means a
# misconfigured production deploy that forgets to set OUROBOROS_API_KEY will
# NOT silently open CORS to "*".
_DEV_MODE = os.getenv("OUROBOROS_DEV_MODE", "").lower() in {"1", "true", "yes"}
_allow_origins = ["*"] if _DEV_MODE else _CORS_ORIGINS

# WebSocket message size cap (bytes).  Exported so tests can stay in sync.
WS_MAX_MESSAGE_BYTES: int = 8192

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sovereign Ouroboros OS API",
    version="0.1.0",
    description="REST + WebSocket interface to the five-layer cognitive loop.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Per-tenant loop registry
# ---------------------------------------------------------------------------

_loop_registry: dict[str, OuroborosLoop] = {}
_registry_lock = threading.Lock()


def _get_loop(tenant_id: str) -> OuroborosLoop:
    """Return (or lazily create) the OuroborosLoop for *tenant_id*."""
    with _registry_lock:
        if tenant_id not in _loop_registry:
            _loop_registry[tenant_id] = OuroborosLoop()
        return _loop_registry[tenant_id]


# Backwards-compatible handle used by existing tests that import `_loop` directly.
# In dev mode (no API key) the default tenant is "default", so this alias
# always points at the same instance that API calls without a key will use.
_loop: OuroborosLoop = _get_loop("default")

# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=4096)
    imagine_k: int = Field(3, ge=1, le=10)


class AddPrinciplesRequest(BaseModel):
    principles: list[str] = Field(..., min_length=1, max_length=20)


# ---------------------------------------------------------------------------
# Rate limiter (stdlib token-bucket, one bucket per tenant)
# ---------------------------------------------------------------------------

_RATE_LIMIT = 60          # requests
_RATE_WINDOW = 60.0       # seconds

if importlib.util.find_spec("slowapi") is not None:
    # Use slowapi if available (not installed in this environment, kept as hook).
    from slowapi import Limiter, _rate_limit_exceeded_handler  # type: ignore[import]
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    _limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = _limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    _USE_SLOWAPI = True
else:
    _USE_SLOWAPI = False

# Simple in-memory token bucket: {tenant_id: (tokens, last_refill_time)}
_rate_buckets: dict[str, tuple[float, float]] = {}
_rate_lock = threading.Lock()


def _check_rate_limit(tenant_id: str) -> None:
    """Raise HTTP 429 if the tenant has exceeded the rate limit."""
    now = time.monotonic()
    with _rate_lock:
        tokens, last_refill = _rate_buckets.get(tenant_id, (float(_RATE_LIMIT), now))
        # Refill tokens proportionally to elapsed time.
        elapsed = now - last_refill
        tokens = min(float(_RATE_LIMIT), tokens + elapsed * (_RATE_LIMIT / _RATE_WINDOW))
        if tokens < 1.0:
            retry_after = math.ceil((1.0 - tokens) * _RATE_WINDOW / _RATE_LIMIT)
            _rate_buckets[tenant_id] = (tokens, now)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded.",
                headers={"Retry-After": str(retry_after)},
            )
        _rate_buckets[tenant_id] = (tokens - 1.0, now)


# ---------------------------------------------------------------------------
# Authentication dependency
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_tenant(api_key: str | None = Depends(_api_key_header)) -> str:
    """Validate the X-API-Key header and return the resolved tenant_id.

    Dev/test mode (OUROBOROS_API_KEY not set): every request is accepted;
    tenant_id is "default".
    Production mode (OUROBOROS_API_KEY set): header must match exactly;
    401 otherwise.  The key itself is used as the tenant_id.
    """
    if _API_KEY is None:
        # Dev/test mode — no authentication required.
        return "default"
    if api_key == _API_KEY:
        return api_key
    raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _prototype_to_dict(p: Prototype) -> dict[str, Any]:
    """Serialise a Prototype to a JSON-safe dict (latent vector omitted)."""
    return {
        "label": p.label,
        "confidence": p.confidence,
        "modality_scores": p.modality_scores,
    }


def _timeline_to_dict(t: Timeline) -> dict[str, Any]:
    """Serialise a Timeline to a JSON-safe dict."""
    return {
        "id": t.id,
        "score": t.score,
        "rationale": t.rationale,
        "trajectory": list(t.trajectory),
        "proposed_action": {
            "intent": t.proposed_action.intent,
            "confirmed": t.proposed_action.confirmed,
            "backup_exists": t.proposed_action.backup_exists,
            "exposes_pii": t.proposed_action.exposes_pii,
            "shares_external": t.proposed_action.shares_external,
            "elevated_privileges": t.proposed_action.elevated_privileges,
            "audit_logged": t.proposed_action.audit_logged,
        },
    }


def _gate_to_dict(g: GateResult) -> dict[str, Any]:
    """Serialise a GateResult to a JSON-safe dict."""
    return {
        "allowed": g.allowed,
        "violations": list(g.violations),
    }


def _execution_to_dict(e: ExecutionResult) -> dict[str, Any]:
    """Serialise an ExecutionResult to a JSON-safe dict."""
    output = e.output
    # ExecutionResult.output may be an arbitrary object; convert to str if needed.
    if not isinstance(output, (dict, list, str, int, float, bool, type(None))):
        output = str(output)
    return {
        "ok": e.ok,
        "output": output,
        "skill_used": e.skill_used,
        "synthesized": e.synthesized,
        "detail": e.detail,
    }


def _federation_to_dict(f: FederatedResult) -> dict[str, Any]:
    """Serialise a FederatedResult to a JSON-safe dict."""
    reconstructed = f.reconstructed
    if not isinstance(reconstructed, (dict, list, str, int, float, bool, type(None))):
        reconstructed = str(reconstructed)
    return {
        "task": f.task,
        "reconstructed": reconstructed,
        "contributors": list(f.contributors),
        "shards": f.shards,
    }


def _loop_result_to_dict(r: LoopResult) -> dict[str, Any]:
    """Serialise a full LoopResult to a JSON-safe dict."""
    return {
        "task": r.task,
        "prototypes": [_prototype_to_dict(p) for p in r.prototypes],
        "timeline": _timeline_to_dict(r.timeline),
        "gate": _gate_to_dict(r.gate),
        "execution": _execution_to_dict(r.execution) if r.execution is not None else None,
        "federation": _federation_to_dict(r.federation) if r.federation is not None else None,
        "step": r.step,
        "blocked": r.blocked,
        "succeeded": r.succeeded,
    }


def _loop_result_summary(r: LoopResult) -> dict[str, Any]:
    """Return a lightweight summary of a LoopResult for the /history endpoint."""
    return {
        "task": r.task,
        "step": r.step,
        "succeeded": r.succeeded,
        "blocked": r.blocked,
    }


# ---------------------------------------------------------------------------
# Health check (no auth)
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness probe — no authentication required."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/metrics")
async def metrics(tenant_id: str = Depends(_get_tenant)) -> dict[str, Any]:
    """Return per-tenant loop metrics (auth + rate-limit required)."""
    _check_rate_limit(tenant_id)
    loop = _get_loop(tenant_id)
    return {
        "loop_step": loop.state.step,
        "history_count": len(loop.history),
        "skill_count": len(loop.metamorph.skills),
        "principle_count": len(loop.ethos.principles),
    }


# ---------------------------------------------------------------------------
# REST endpoints (auth + rate-limit required)
# ---------------------------------------------------------------------------


@app.post("/run")
async def run_task(
    body: RunRequest,
    tenant_id: str = Depends(_get_tenant),
) -> dict[str, Any]:
    """Execute one full Ouroboros cycle and return the serialised LoopResult."""
    _check_rate_limit(tenant_id)
    loop = _get_loop(tenant_id)
    task_hash = hashlib.sha256(body.task.encode()).hexdigest()[:16]
    t0 = time.monotonic()
    logger.info("api.run.start tenant=%r task_hash=%s", tenant_id, task_hash)
    result = loop.run(body.task, imagine_k=body.imagine_k)
    duration = time.monotonic() - t0
    logger.info(
        "api.run.complete tenant=%r task_hash=%s duration=%.3fs succeeded=%s",
        tenant_id,
        task_hash,
        duration,
        result.succeeded,
    )
    return _loop_result_to_dict(result)


@app.get("/history")
async def get_history(tenant_id: str = Depends(_get_tenant)) -> list[dict[str, Any]]:
    """Return lightweight summaries of every past loop run (task, step, succeeded, blocked)."""
    _check_rate_limit(tenant_id)
    loop = _get_loop(tenant_id)
    return [_loop_result_summary(r) for r in loop.history]


@app.get("/state")
async def get_state(tenant_id: str = Depends(_get_tenant)) -> dict[str, Any]:
    """Return the current WorldState: the step counter and accumulated facts."""
    _check_rate_limit(tenant_id)
    loop = _get_loop(tenant_id)
    return {
        "step": loop.state.step,
        "facts": loop.state.facts,
    }


@app.get("/skills")
async def get_skills(tenant_id: str = Depends(_get_tenant)) -> list[str]:
    """Return the sorted list of skill names registered in the MetaMorph engine."""
    _check_rate_limit(tenant_id)
    loop = _get_loop(tenant_id)
    return loop.metamorph.skills


@app.get("/principles")
async def get_principles(tenant_id: str = Depends(_get_tenant)) -> list[str]:
    """Return the names (first 60 chars of each principle text) of active ethical principles."""
    _check_rate_limit(tenant_id)
    loop = _get_loop(tenant_id)
    return [p.name for p in loop.ethos.principles]


@app.post("/principles")
async def add_principle(
    body: AddPrinciplesRequest,
    tenant_id: str = Depends(_get_tenant),
) -> dict[str, Any]:
    """Hot-load new ethical principles into the running EthosCompiler.

    Body:
        principles (list[str], required) – list of natural language principle texts
    """
    _check_rate_limit(tenant_id)
    loop = _get_loop(tenant_id)
    for principle in body.principles:
        loop.ethos.add_principle(principle)
    return {"added": body.principles}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/run")
async def ws_run(websocket: WebSocket) -> None:
    """Stream per-stage cognitive events for a single Ouroboros cycle.

    Client sends:  {"task": "..."}
    Server emits one JSON message per stage, then a "complete" summary.

    Stages emitted (in order, unless ethics blocks early):
        neurosynth  -> chronoweave -> ethos -> [metamorph -> hivemind] -> complete

    Messages over 8192 bytes are rejected with an error frame.
    """
    await websocket.accept()

    raw = await websocket.receive_text()

    # Guard: reject oversized messages.
    if len(raw.encode("utf-8")) > WS_MAX_MESSAGE_BYTES:
        await websocket.send_text(
            json.dumps({"error": f"message too large (max {WS_MAX_MESSAGE_BYTES} bytes)"})
        )
        await websocket.close()
        return

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        await websocket.send_text(json.dumps({"error": "invalid JSON"}))
        await websocket.close()
        return

    task: str = payload.get("task", "")

    # Resolve tenant: check API key from WS payload (optional field) or default.
    ws_api_key: str | None = payload.get("api_key")
    if _API_KEY is None:
        tenant_id = "default"
    elif ws_api_key == _API_KEY:
        tenant_id = ws_api_key
    else:
        await websocket.send_text(json.dumps({"error": "unauthorized"}))
        await websocket.close()
        return

    try:
        _check_rate_limit(tenant_id)
    except HTTPException as exc:
        retry = exc.headers.get("Retry-After", "60") if exc.headers else "60"
        await websocket.send_text(
            json.dumps({"error": "rate limit exceeded", "retry_after": retry})
        )
        await websocket.close()
        return

    loop = _get_loop(tenant_id)

    # --- Stage 1: NeuroSynth -------------------------------------------------
    prototypes = loop.neurosynth.imagine(task, k=loop.imagine_k)
    await websocket.send_text(
        json.dumps(
            {
                "stage": "neurosynth",
                "prototypes": [_prototype_to_dict(p) for p in prototypes],
            }
        )
    )

    # --- Stage 2: ChronoWeave ------------------------------------------------
    timeline = loop.chronoweave.simulate(task, prototypes, loop.state)
    await websocket.send_text(
        json.dumps(
            {
                "stage": "chronoweave",
                "timeline_id": timeline.id,
                "score": timeline.score,
                "rationale": timeline.rationale,
            }
        )
    )

    # --- Stage 3: EthosCompiler ----------------------------------------------
    gate = loop.ethos.gate(timeline.proposed_action.as_action_dict())
    await websocket.send_text(
        json.dumps(
            {
                "stage": "ethos",
                "allowed": gate.allowed,
                "violations": list(gate.violations),
            }
        )
    )

    if not gate.allowed:
        # Ethics blocked: record in history and emit "complete" immediately.
        blocked_result = LoopResult(
            task=task,
            prototypes=prototypes,
            timeline=timeline,
            gate=gate,
            execution=None,
            federation=None,
            step=loop.state.step,
            blocked=True,
        )
        loop.history.append(blocked_result)
        await websocket.send_text(
            json.dumps(
                {
                    "stage": "complete",
                    "succeeded": False,
                    "blocked": True,
                    "step": blocked_result.step,
                }
            )
        )
        await websocket.close()
        return

    # --- Stage 4: MetaMorph --------------------------------------------------
    execution = loop.metamorph.execute(timeline.proposed_action)
    await websocket.send_text(
        json.dumps(
            {
                "stage": "metamorph",
                "skill": execution.skill_used,
                "synthesized": execution.synthesized,
                "ok": execution.ok,
            }
        )
    )

    # --- Stage 5: HiveMind ---------------------------------------------------
    federation = loop.hivemind.expand(task, execution.output)
    await websocket.send_text(
        json.dumps(
            {
                "stage": "hivemind",
                "shards": federation.shards,
                "contributors": list(federation.contributors),
            }
        )
    )

    # Advance world state (mirrors OuroborosLoop.run).
    loop.state.step += 1
    loop.state.facts[task] = {
        "skill": execution.skill_used,
        "synthesized": execution.synthesized,
        "score": timeline.score,
    }

    ws_result = LoopResult(
        task=task,
        prototypes=prototypes,
        timeline=timeline,
        gate=gate,
        execution=execution,
        federation=federation,
        step=loop.state.step,
    )
    loop.history.append(ws_result)

    await websocket.send_text(
        json.dumps(
            {
                "stage": "complete",
                "succeeded": ws_result.succeeded,
                "blocked": ws_result.blocked,
                "step": ws_result.step,
            }
        )
    )
    await websocket.close()
