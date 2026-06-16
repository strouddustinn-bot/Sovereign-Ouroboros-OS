"""EthosCompiler core: principle parsing and action gating."""

from __future__ import annotations

import re
import threading
import warnings
from dataclasses import dataclass, field
from typing import Callable

# Ensure warnings from this package are visible by default.
warnings.filterwarnings("default", category=UserWarning, module="ouroboros")

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
        reason:     Optional human-readable reason string (used for error reporting).
    """

    allowed: bool
    violations: list[str]
    reason: str = ""


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

# Intent keyword families. Conditional guards only fire when an action's intent
# actually belongs to the risky category — e.g. "never delete without backup"
# constrains deletions, not every action.
_DELETE_KEYWORDS = ("delete", "remove", "erase", "wipe", "drop", "destroy", "purge")
_MUTATING_KEYWORDS = _DELETE_KEYWORDS + (
    "send",
    "share",
    "publish",
    "deploy",
    "overwrite",
)


def _intent_contains(action: Action, keywords: tuple[str, ...]) -> bool:
    """True if the action's intent text mentions any of *keywords*."""
    intent = str(action.get("intent", "")).lower()
    return any(kw in intent for kw in keywords)


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
        # Only mutating/destructive actions need prior confirmation.
        lambda action: (
            not _intent_contains(action, _MUTATING_KEYWORDS)
            or bool(action.get("confirmed", False))
        ),
    ),
    (
        r"never (delete|remove|erase|wipe|drop) without (backup|snapshot|copy)",
        # Only deletions need a backup; other actions are unaffected.
        lambda action: (
            not _intent_contains(action, _DELETE_KEYWORDS)
            or bool(action.get("backup_exists", False))
        ),
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
        self._lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, principle_text: str) -> EthicalPrinciple:
        """Compile one natural language principle into an EthicalPrinciple."""
        normalized = principle_text.lower().strip()
        predicate = self._match_pattern(normalized, principle_text)
        return EthicalPrinciple(
            name=principle_text[:60].rstrip(),
            description=principle_text,
            predicate=predicate,
        )

    def load_principles(self, principles: list[str]) -> None:
        """Replace the current principle set with a freshly compiled list."""
        compiled = [self.compile(p) for p in principles]
        with self._lock:
            self._principles = compiled

    def add_principle(self, principle_text: str) -> None:
        """Append a single principle to the active set."""
        compiled = self.compile(principle_text)
        with self._lock:
            self._principles.append(compiled)

    def gate(self, action: Action) -> GateResult:
        """Run *action* through every compiled constraint.

        Returns a :class:`GateResult` whose ``allowed`` field is True only
        when all principles permit the action.
        """
        with self._lock:
            violations = [p.name for p in self._principles if not p.allows(action)]
        return GateResult(allowed=not violations, violations=violations)

    @property
    def principles(self) -> list[EthicalPrinciple]:
        with self._lock:
            return list(self._principles)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match_pattern(
        self, normalized: str, principle_text: str = ""
    ) -> Callable[[Action], bool]:
        for pattern, predicate_fn in _PATTERN_REGISTRY:
            if re.search(pattern, normalized):
                return predicate_fn
        # Principle not recognized by any pattern: warn and allow.
        # TODO: integrate LLM-based principle parsing as a fallback here.
        warnings.warn(
            f"EthosCompiler: no pattern matched principle {principle_text!r}; "
            "defaulting to ALLOW. Consider adding an explicit pattern.",
            stacklevel=3,
        )
        return lambda _action: True
