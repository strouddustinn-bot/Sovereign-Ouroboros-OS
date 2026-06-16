"""ChronoWeave core: the Counterfactual Timeline Engine.

ChronoWeave replaces linear planning with a small multiverse. For every
:class:`~ouroboros.core.types.Prototype` imagined upstream it
spawns several *counterfactual branches* — divergent ways the same intent could
unfold — simulates each via lightweight, transparent causal heuristics, and
then *collapses* the multiverse to the single highest-value timeline.

The "causal inference" here is deliberately dependency-free: each branch is
scored by combining semantic alignment with the task, a penalty for unmitigated
risk, and a small reward for the prototype's imagination confidence. Scoring is
fully deterministic, so the same inputs always collapse to the same timeline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ouroboros.core.embedding import cosine, embed
from ouroboros.core.types import (
    Prototype,
    ProposedAction,
    Timeline,
    WorldState,
)

# Risk keywords detected in a task, mapped to the action-dict flags the
# EthosCompiler inspects downstream. Each entry names the mitigations that make
# a risky action safe so the "cautious" branch can switch them on.
_RISK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "delete": ("delete", "remove", "erase", "wipe", "drop", "destroy"),
    "share": ("share", "expose", "leak", "send", "publish", "exfiltrate"),
    "harm": ("harm", "hurt", "damage", "kill"),
    "privacy": ("pii", "personal", "private", "secret", "credential"),
}

# Scoring weights. Kept as module constants so the heuristic stays transparent.
_ALIGNMENT_WEIGHT = 1.0
_CONFIDENCE_WEIGHT = 0.25
_RISK_PENALTY = 0.5


@dataclass(frozen=True)
class _BranchSpec:
    """A counterfactual flavour applied to a prototype.

    Attributes:
        name:     Short label for the branch (e.g. ``"cautious"``).
        mitigate: When True, the branch switches on risk mitigations so the
                  resulting action is safe to execute.
        descriptor: Human-readable phrase woven into the branch intent.
    """

    name: str
    mitigate: bool
    descriptor: str


# The three counterfactual flavours spawned per prototype. Order is stable so
# branch ids — and therefore tie-breaking — stay reproducible.
_BRANCH_SPECS: tuple[_BranchSpec, ...] = (
    _BranchSpec("optimistic", mitigate=False, descriptor="assume the happy path"),
    _BranchSpec("cautious", mitigate=True, descriptor="with safeguards in place"),
    _BranchSpec("reckless", mitigate=False, descriptor="ignore the guardrails"),
)


class ChronoWeave:
    """Simulate parallel futures and collapse them to the best timeline.

    ChronoWeave satisfies the :class:`~ouroboros.core.contracts.Simulator`
    protocol. :meth:`weave` exposes the full multiverse for introspection while
    :meth:`simulate` returns only the collapsed, highest-value timeline.

    Usage::

        weaver = ChronoWeave()
        timeline = weaver.simulate("delete stale cache", prototypes, state)
        action = timeline.proposed_action  # ready for the EthosCompiler gate
    """

    def __init__(self, branches: int = 3) -> None:
        """Configure the engine.

        Args:
            branches: Number of counterfactual branches to spawn per prototype.
                Clamped to ``[1, len(_BRANCH_SPECS)]``.
        """
        self._branches = max(1, min(branches, len(_BRANCH_SPECS)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(
        self, task: str, prototypes: list[Prototype], state: WorldState
    ) -> Timeline:
        """Simulate every prototype and collapse to the best timeline.

        Returns the single highest-value probabilistic path. Equivalent to
        ``self.weave(task, prototypes, state)[0]``.
        """
        timelines = self.weave(task, prototypes, state)
        if not timelines:
            raise ValueError("cannot simulate without at least one prototype")
        return timelines[0]

    def weave(
        self, task: str, prototypes: list[Prototype], state: WorldState
    ) -> list[Timeline]:
        """Spawn and score the full multiverse of timelines.

        Returns every simulated :class:`Timeline` sorted by descending score,
        with stable tie-breaking by timeline id so ordering is reproducible.
        """
        task_vector = embed(task)
        risks = self._detect_risks(task)
        timelines: list[Timeline] = []
        for proto_index, prototype in enumerate(prototypes):
            for spec in _BRANCH_SPECS[: self._branches]:
                timelines.append(
                    self._weave_branch(
                        task, task_vector, risks, proto_index, prototype, spec
                    )
                )
        timelines.sort(key=lambda t: (-t.score, t.id))
        return timelines

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_risks(task: str) -> frozenset[str]:
        """Return the risk categories implied by *task*."""
        lowered = task.lower()
        return frozenset(
            category
            for category, keywords in _RISK_KEYWORDS.items()
            if any(kw in lowered for kw in keywords)
        )

    def _weave_branch(
        self,
        task: str,
        task_vector: tuple[float, ...],
        risks: frozenset[str],
        proto_index: int,
        prototype: Prototype,
        spec: _BranchSpec,
    ) -> Timeline:
        """Build, score, and package a single counterfactual branch."""
        intent = f"{task} via {prototype.label} ({spec.descriptor})"
        action = self._build_action(intent, risks, spec, prototype)
        alignment = cosine(embed(intent), task_vector)
        unmitigated = bool(risks) and not spec.mitigate
        score = (
            _ALIGNMENT_WEIGHT * alignment
            + _CONFIDENCE_WEIGHT * prototype.confidence
            - (_RISK_PENALTY if unmitigated else 0.0)
        )
        timeline_id = self._branch_id(task, proto_index, prototype, spec)
        rationale = self._rationale(spec, risks, alignment, prototype, unmitigated)
        trajectory = self._trajectory(task, prototype, spec, risks)
        return Timeline(
            id=timeline_id,
            proposed_action=action,
            score=round(score, 6),
            rationale=rationale,
            trajectory=trajectory,
        )

    @staticmethod
    def _build_action(
        intent: str,
        risks: frozenset[str],
        spec: _BranchSpec,
        prototype: Prototype,
    ) -> ProposedAction:
        """Derive a ProposedAction, wiring mitigation flags per branch flavour.

        The cautious branch switches on the mitigations relevant to each
        detected risk; other branches leave them off. PII is never exposed and
        every branch is audit-logged.
        """
        confirmed = False
        backup_exists = False
        shares_external = bool({"share"} & risks)
        if spec.mitigate:
            if {"delete", "harm"} & risks:
                confirmed = True
                backup_exists = True
            if "share" in risks:
                shares_external = False
            if risks:
                confirmed = True
        return ProposedAction(
            intent=intent,
            params={"prototype": prototype.label, "branch": spec.name},
            confirmed=confirmed,
            backup_exists=backup_exists,
            exposes_pii=False,
            shares_external=shares_external,
            elevated_privileges=False,
            audit_logged=True,
        )

    @staticmethod
    def _branch_id(
        task: str, proto_index: int, prototype: Prototype, spec: _BranchSpec
    ) -> str:
        """Return a deterministic, unique id for a branch."""
        digest = hashlib.sha256(
            f"{task}|{proto_index}|{prototype.label}|{spec.name}".encode("utf-8")
        ).hexdigest()[:8]
        return f"tl-{proto_index}-{spec.name}-{digest}"

    @staticmethod
    def _rationale(
        spec: _BranchSpec,
        risks: frozenset[str],
        alignment: float,
        prototype: Prototype,
        unmitigated: bool,
    ) -> str:
        """Explain, in plain language, why a branch scored as it did."""
        parts = [
            f"{spec.name} branch of '{prototype.label}'",
            f"semantic alignment {alignment:.3f}",
            f"confidence {prototype.confidence:.2f}",
        ]
        if not risks:
            parts.append("no risks detected")
        elif unmitigated:
            parts.append(f"unmitigated risks {sorted(risks)} (penalised)")
        else:
            parts.append(f"risks {sorted(risks)} mitigated")
        return "; ".join(parts)

    @staticmethod
    def _trajectory(
        task: str,
        prototype: Prototype,
        spec: _BranchSpec,
        risks: frozenset[str],
    ) -> list[str]:
        """Describe the simulated steps along this timeline."""
        steps = [
            f"fork world-state for '{task}'",
            f"adopt prototype '{prototype.label}' on the {spec.name} branch",
        ]
        if risks:
            verb = "apply safeguards for" if spec.mitigate else "proceed despite"
            steps.append(f"{verb} risks {sorted(risks)}")
        else:
            steps.append("no risk mitigation required")
        steps.append("project outcome and score timeline")
        return steps
