"""FastAPI application exposing the Ouroboros cognitive loop over HTTP + WebSocket.

REST endpoints
--------------
POST /run              – execute one full loop cycle, returns LoopResult as JSON
GET  /history          – list of past run summaries
GET  /state            – current WorldState (step + facts)
GET  /skills           – registered MetaMorph skill names
GET  /principles       – active EthicalPrinciple names
POST /principles       – hot-load a new principle

WebSocket
---------
GET  /ws/run           – stream one JSON message per cognitive stage, then summary
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from sovereign_ouroboros_os.core.types import (
    ExecutionResult,
    FederatedResult,
    Prototype,
    Timeline,
)
from sovereign_ouroboros_os.ethos_compiler import GateResult
from sovereign_ouroboros_os.ouroboros_loop import (
    DEFAULT_PRINCIPLES,
    LoopResult,
    OuroborosLoop,
)

# ---------------------------------------------------------------------------
# Application factory & shared state
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sovereign Ouroboros OS API",
    version="0.1.0",
    description="REST + WebSocket interface to the five-layer cognitive loop.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# A single shared loop instance initialised at module import time.
# Tests that import `app` directly share this instance, which is intentional
# (it lets /history grow across calls in the same process).
_loop: OuroborosLoop = OuroborosLoop()


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
# REST endpoints
# ---------------------------------------------------------------------------


@app.post("/run")
async def run_task(body: dict[str, Any]) -> dict[str, Any]:
    """Execute one full Ouroboros cycle and return the serialised LoopResult.

    Body:
        task       (str, required)  – natural language task description
        principles (list[str], opt) – override the active principle set for
                                      this run only (the shared loop instance
                                      is not permanently mutated)
    """
    task: str = body.get("task", "")
    override_principles: list[str] | None = body.get("principles")

    if override_principles is not None:
        # Swap the principle set for this call only, then restore.
        saved_principles = [p.description for p in _loop.ethos.principles]
        _loop.ethos.load_principles(override_principles)
        try:
            result = _loop.run(task)
        finally:
            _loop.ethos.load_principles(saved_principles)
    else:
        result = _loop.run(task)

    return _loop_result_to_dict(result)


@app.get("/history")
async def get_history() -> list[dict[str, Any]]:
    """Return lightweight summaries of every past loop run (task, step, succeeded, blocked)."""
    return [_loop_result_summary(r) for r in _loop.history]


@app.get("/state")
async def get_state() -> dict[str, Any]:
    """Return the current WorldState: the step counter and accumulated facts."""
    return {
        "step": _loop.state.step,
        "facts": _loop.state.facts,
    }


@app.get("/skills")
async def get_skills() -> list[str]:
    """Return the sorted list of skill names registered in the MetaMorph engine."""
    return _loop.metamorph.skills


@app.get("/principles")
async def get_principles() -> list[str]:
    """Return the names (first 60 chars of each principle text) of active ethical principles."""
    return [p.name for p in _loop.ethos.principles]


@app.post("/principles")
async def add_principle(body: dict[str, Any]) -> dict[str, Any]:
    """Hot-load a new ethical principle into the running EthosCompiler.

    Body:
        principle (str, required) – natural language principle text
    """
    principle: str = body.get("principle", "")
    _loop.ethos.add_principle(principle)
    return {"added": principle}


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
    """
    await websocket.accept()

    raw = await websocket.receive_text()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        await websocket.send_text(json.dumps({"error": "invalid JSON"}))
        await websocket.close()
        return

    task: str = payload.get("task", "")

    # --- Stage 1: NeuroSynth -------------------------------------------------
    prototypes = _loop.neurosynth.imagine(task, k=_loop.imagine_k)
    await websocket.send_text(
        json.dumps(
            {
                "stage": "neurosynth",
                "prototypes": [_prototype_to_dict(p) for p in prototypes],
            }
        )
    )

    # --- Stage 2: ChronoWeave ------------------------------------------------
    timeline = _loop.chronoweave.simulate(task, prototypes, _loop.state)
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
    gate = _loop.ethos.gate(timeline.proposed_action.as_action_dict())
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
            step=_loop.state.step,
            blocked=True,
        )
        _loop.history.append(blocked_result)
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
    execution = _loop.metamorph.execute(timeline.proposed_action)
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
    federation = _loop.hivemind.expand(task, execution.output)
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
    _loop.state.step += 1
    _loop.state.facts[task] = {
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
        step=_loop.state.step,
    )
    _loop.history.append(ws_result)

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
