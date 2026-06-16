"""Command-line entry point: bring Project Ouroboros to life.

Usage::

    python -m ouroboros "rewrite the cache layer"
    python -m ouroboros            # runs a built-in demo reel
    python -m ouroboros serve      # starts the FastAPI server on :8000
    python -m ouroboros serve --port 9000
"""

from __future__ import annotations

import sys

from ouroboros.ouroboros_loop import LoopResult, OuroborosLoop

BANNER = r"""
   ____                        _
  / __ \__  ___________  ____ | |__   ___  _ __ ___  ___
 / / _` | | | | '__/ _ \| '_ \| '_ \ / _ \| '__/ _ \/ __|
| | (_| | |_| | | | (_) | |_) | |_) | (_) | | | (_) \__ \
 \ \__,_|\__,_|_|  \___/| .__/|_.__/ \___/|_|  \___/|___/
  \____/                |_|   Sovereign Agentic OS  v0.1.0
"""

_RULE = "─" * 64


def _render(result: LoopResult) -> None:
    """Pretty-print one trip around the Ouroboros loop."""
    print(_RULE)
    print(f"▶ TASK: {result.task}")
    print(_RULE)

    # 1. NeuroSynth
    print("\n① NeuroSynth — Imagination (the Mind's Eye)")
    for i, proto in enumerate(result.prototypes):
        mods = ", ".join(f"{m}={s:.2f}" for m, s in proto.modality_scores.items())
        print(f"   • prototype[{i}] {proto.label!r}  conf={proto.confidence:.2f}")
        if mods:
            print(f"       modalities: {mods}")

    # 2. ChronoWeave
    t = result.timeline
    print("\n② ChronoWeave — Counterfactual Simulation")
    print(f"   collapsed timeline {t.id[:8]}  score={t.score:.3f}")
    print(f"   rationale: {t.rationale}")
    if t.trajectory:
        print(f"   trajectory: {' → '.join(t.trajectory)}")
    print(f"   ⇒ proposed action: {t.proposed_action.intent!r}")

    # 3. EthosCompiler
    print("\n③ EthosCompiler — Executable Ethics")
    if result.gate.allowed:
        print("   ✓ action permitted by all compiled principles")
    else:
        print(f"   ✗ BLOCKED by: {', '.join(result.gate.violations)}")

    if result.blocked:
        print("\n⛔ Loop halted at the moral compass. Nothing was executed.")
        print(_RULE + "\n")
        return

    # 4. MetaMorph
    ex = result.execution
    assert ex is not None
    print("\n④ MetaMorph — Execution & Self-Evolution")
    tag = "SYNTHESIZED new skill" if ex.synthesized else "used existing skill"
    print(f"   {tag}: {ex.skill_used!r}  ok={ex.ok}")
    print(f"   output: {ex.output!r}")

    # 5. HiveMind
    fed = result.federation
    assert fed is not None
    print("\n⑤ HiveMind — Federated Expansion (Sovereign Node)")
    print(f"   fragmented across {fed.shards} encrypted shares")
    print(f"   contributors: {', '.join(fed.contributors)}")
    print(f"   collective synthesis: {fed.reconstructed!r}")

    print(f"\n♻  Loop closed. World-state advanced to step {result.step}.")
    print(_RULE + "\n")


def _serve(argv: list[str]) -> int:
    """Start the FastAPI server via uvicorn.

    Parses an optional ``--port PORT`` argument from *argv* (the args that
    follow ``serve``).  All other arguments are silently ignored so that
    callers can pass extra uvicorn flags in the future without breaking the
    entry point.
    """
    import uvicorn  # local import: only needed for the serve sub-command

    port = 8000
    i = 0
    while i < len(argv):
        if argv[i] in ("--port", "-p") and i + 1 < len(argv):
            try:
                port = int(argv[i + 1])
            except ValueError:
                print(f"Invalid port: {argv[i + 1]!r}", file=sys.stderr)
                return 1
            i += 2
        else:
            i += 1

    print(BANNER)
    print(f"Starting Ouroboros API on http://0.0.0.0:{port} …")
    uvicorn.run(
        "ouroboros.api.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    # -----------------------------------------------------------------------
    # serve sub-command: delegate before printing the banner or creating a loop
    # -----------------------------------------------------------------------
    if argv and argv[0] == "serve":
        return _serve(argv[1:])

    print(BANNER)

    os = OuroborosLoop()

    if argv:
        tasks = [" ".join(argv)]
    else:
        print("(no task given — running the demo reel)\n")
        tasks = [
            "summarize the quarterly research notes",
            "delete the stale build cache",
            "harm the production database",  # should be blocked by ethics
            "design a novel compression scheme",
        ]

    for result in os.run_many(tasks):
        _render(result)

    print(f"Ouroboros completed {len(tasks)} cycle(s). The serpent endures. 🐍")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
