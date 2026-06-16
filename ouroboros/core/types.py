"""Shared data types exchanged between the five cognitive layers.

These types form the contract the Ouroboros loop passes between stages:

    Prototype (NeuroSynth) -> Timeline/ProposedAction (ChronoWeave)
        -> GateResult (EthosCompiler) -> ExecutionResult (MetaMorph)
        -> FederatedResult (HiveMind)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# A dense latent vector. Kept as a plain tuple so layers need no heavy
# numerical dependency to exchange embeddings.
Vector = tuple[float, ...]


@dataclass(frozen=True)
class Prototype:
    """A candidate solution concept imagined by NeuroSynth.

    Attributes:
        label:           Human-readable summary of the imagined solution.
        latent:          Fused multi-sensory latent embedding.
        modality_scores: Per-modality salience (e.g. ``{"visual": 0.7}``).
        confidence:      Imagination confidence in ``[0, 1]``.
    """

    label: str
    latent: Vector
    modality_scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class WorldState:
    """A mutable snapshot of the agent's beliefs about the world."""

    facts: dict[str, Any] = field(default_factory=dict)
    step: int = 0

    def fork(self) -> "WorldState":
        """Return an independent copy for counterfactual rollouts."""
        return WorldState(facts=dict(self.facts), step=self.step)


@dataclass(frozen=True)
class ProposedAction:
    """A concrete action the loop intends to take.

    The flag fields mirror the metadata the EthosCompiler inspects, so an
    action can be gated directly via :meth:`as_action_dict`.
    """

    intent: str
    params: dict[str, Any] = field(default_factory=dict)
    confirmed: bool = False
    backup_exists: bool = False
    exposes_pii: bool = False
    shares_external: bool = False
    elevated_privileges: bool = False
    audit_logged: bool = True

    def as_action_dict(self) -> dict[str, Any]:
        """Flatten into the ``dict`` shape EthosCompiler.gate expects."""
        return {
            "intent": self.intent,
            "confirmed": self.confirmed,
            "backup_exists": self.backup_exists,
            "exposes_pii": self.exposes_pii,
            "shares_external": self.shares_external,
            "elevated_privileges": self.elevated_privileges,
            "audit_logged": self.audit_logged,
            **self.params,
        }


@dataclass(frozen=True)
class Timeline:
    """A single simulated future produced by ChronoWeave."""

    id: str
    proposed_action: ProposedAction
    score: float
    rationale: str
    trajectory: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Skill:
    """An executable capability in the MetaMorph registry."""

    name: str
    fn: Callable[..., Any]
    source: str = "builtin"
    synthesized: bool = False


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of executing a ProposedAction through MetaMorph."""

    ok: bool
    output: Any
    skill_used: str
    synthesized: bool = False
    detail: str = ""


@dataclass(frozen=True)
class FederatedResult:
    """Aggregate of a problem fragmented and solved across HiveMind peers."""

    task: str
    reconstructed: Any
    contributors: list[str]
    shards: int
