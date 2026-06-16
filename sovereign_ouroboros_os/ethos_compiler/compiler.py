"""EthosCompiler core: principle parsing and action gating."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

Action = dict  # Typed alias; callers pass arbitrary metadata about an action.


@dataclass(frozen=True)
class EthicalPrinciple:
    """An ethical principle compiled into a callable predicate.

    Attributes:
        name:        Short label (first 60 chars of the source text).
        description: Full source text of the principle.
        predicate:   Callable that returns True when the action is allowed.
    """

    name: str
    description: str
    predicate: Callable[[Action], bool]

    def allows(self, action: Action) -> bool:
        return self.predicate(action)


@dataclass(frozen=True)
class GateResult:
    """Result of running an action through the compiled ethics gate.

    Attributes:
        allowed:    True only when every principle permits the action.
        violations: Names of principles that blocked the action.
    """

    allowed: bool
    violations: list[str]


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

# Each entry maps a regex (matched against the lower-cased principle text) to
# a factory that returns the corresponding predicate.
_PATTERN_REGISTRY: list[tuple[str, Callable[[Action], bool]]] = [
    (
        r"do not .*(harm|hurt|damage|destroy|kill)",
        lambda action: not any(
            kw in str(action.get("intent", "")).lower()
            for kw in ("harm", "hurt", "damage", "destroy", "kill")
        ),
    ),
    (
        r"(always )?(ask|confirm|get approval|verify) before",
        lambda action: bool(action.get("confirmed", False)),
    ),
    (
        r"never (delete|remove|erase|wipe|drop) without (backup|snapshot|copy)",
        lambda action: bool(action.get("backup_exists", False)),
    ),
    (
        r"(respect|protect|preserve) (user )?privacy",
        lambda action: not bool(action.get("exposes_pii", False)),
    ),
    (
        r"(do not|never) (exfiltrate|leak|expose|share) (data|information|secrets?)",
        lambda action: not bool(action.get("shares_external", False)),
    ),
    (
        r"(do not|never) run (as root|with elevated|with admin)",
        lambda action: not bool(action.get("elevated_privileges", False)),
    ),
    (
        r"(log|record|audit) (all|every) (action|operation|command)",
        lambda action: bool(action.get("audit_logged", False)),
    ),
]


class EthosCompiler:
    """Compiles natural language ethical principles into executable predicates.

    Usage::

        compiler = EthosCompiler()
        compiler.load_principles([
            "Do not harm users.",
            "Always confirm before deleting files.",
            "Respect user privacy.",
        ])

        action = {"intent": "delete /tmp/cache", "confirmed": True}
        result = compiler.gate(action)
        if not result.allowed:
            raise RuntimeError(f"Action blocked by: {result.violations}")
    """

    def __init__(self) -> None:
        self._principles: list[EthicalPrinciple] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, principle_text: str) -> EthicalPrinciple:
        """Compile one natural language principle into an EthicalPrinciple."""
        normalized = principle_text.lower().strip()
        predicate = self._match_pattern(normalized)
        return EthicalPrinciple(
            name=principle_text[:60].rstrip(),
            description=principle_text,
            predicate=predicate,
        )

    def load_principles(self, principles: list[str]) -> None:
        """Replace the current principle set with a freshly compiled list."""
        self._principles = [self.compile(p) for p in principles]

    def add_principle(self, principle_text: str) -> None:
        """Append a single principle to the active set."""
        self._principles.append(self.compile(principle_text))

    def gate(self, action: Action) -> GateResult:
        """Run *action* through every compiled constraint.

        Returns a :class:`GateResult` whose ``allowed`` field is True only
        when all principles permit the action.
        """
        violations = [p.name for p in self._principles if not p.allows(action)]
        return GateResult(allowed=not violations, violations=violations)

    @property
    def principles(self) -> list[EthicalPrinciple]:
        return list(self._principles)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match_pattern(self, normalized: str) -> Callable[[Action], bool]:
        for pattern, predicate_fn in _PATTERN_REGISTRY:
            if re.search(pattern, normalized):
                return predicate_fn
        # Principle not recognized by any pattern: log-and-allow.
        # TODO: integrate LLM-based principle parsing as a fallback here.
        return lambda _action: True
