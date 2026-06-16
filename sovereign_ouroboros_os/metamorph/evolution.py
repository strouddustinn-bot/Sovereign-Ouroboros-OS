"""MetaMorph core: capability-gap detection and runtime skill synthesis.

This module implements the self-modifying heart of the Ouroboros loop. The
:class:`MetaMorph` engine keeps a live registry of executable
:class:`~sovereign_ouroboros_os.core.types.Skill` objects keyed by capability
keyword. When an incoming :class:`~sovereign_ouroboros_os.core.types.ProposedAction`
matches no registered skill, that is treated as a *capability gap*: a new skill
is synthesized from a deterministic source-code template, validated inside an
isolated sandbox, and hot-swapped into the registry without restarting the
process. The action is then served by the freshly minted skill.

Synthesized code is compiled and executed in a fresh namespace whose
``__builtins__`` is restricted to a tiny safe allowlist, so generated functions
cannot reach back into the host module, the filesystem, the network, or ``os``.
All generation is deterministic: the same intent always yields the same skill.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from sovereign_ouroboros_os.core.embedding import cosine, embed
from sovereign_ouroboros_os.core.types import (
    ExecutionResult,
    ProposedAction,
    Skill,
)

# Minimum cosine similarity for a skill to be considered a candidate for
# composition with a given intent.
_COMPOSITION_THRESHOLD: float = 0.35

# ---------------------------------------------------------------------------
# Sandbox configuration
# ---------------------------------------------------------------------------

# The only builtins synthesized skills are permitted to reference. Anything not
# in this allowlist (``open``, ``__import__``, ``eval``, ``exec`` ...) is simply
# absent from the exec namespace, so generated code cannot escape the sandbox.
_SAFE_BUILTINS: dict[str, Any] = {
    "len": len,
    "str": str,
    "dict": dict,
    "list": list,
    "sorted": sorted,
    "sum": sum,
    "min": min,
    "max": max,
    "range": range,
    "enumerate": enumerate,
    "reversed": reversed,
}

# A deterministic, side-effect-free probe fed to candidate skills during
# sandbox validation.
_PROBE_INTENT = "metamorph probe input"

# Non-word characters used to derive a safe Python identifier from an intent.
_IDENTIFIER_SCRUBBER = re.compile(r"\W+")


# ---------------------------------------------------------------------------
# Builtin skill implementations
# ---------------------------------------------------------------------------


def _skill_echo(intent: str, params: dict[str, Any]) -> dict[str, Any]:
    """Return the intent text verbatim alongside its parameters."""
    return {"skill": "echo", "intent": intent, "echo": intent, "params": dict(params)}


def _skill_reverse(intent: str, params: dict[str, Any]) -> dict[str, Any]:
    """Return the intent text reversed character-by-character."""
    return {"skill": "reverse", "intent": intent, "reversed": intent[::-1]}


def _skill_count(intent: str, params: dict[str, Any]) -> dict[str, Any]:
    """Return character and whitespace-delimited word counts for the intent."""
    return {
        "skill": "count",
        "intent": intent,
        "chars": len(intent),
        "words": len(intent.split()),
    }


def _skill_summarize(intent: str, params: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic single-line summary of the intent text."""
    words = intent.split()
    head = " ".join(words[:8])
    summary = head if len(words) <= 8 else head + " ..."
    return {
        "skill": "summarize",
        "intent": intent,
        "summary": summary,
        "word_count": len(words),
    }


# Capability keyword -> builtin skill factory input. Each tuple is
# ``(name, callable)``; the keyword equals the skill name for routing.
_BUILTIN_SKILLS: tuple[tuple[str, Callable[[str, dict[str, Any]], Any]], ...] = (
    ("echo", _skill_echo),
    ("reverse", _skill_reverse),
    ("count", _skill_count),
    ("summarize", _skill_summarize),
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class MetaMorph:
    """Self-modifying execution engine satisfying the ``Evolver`` protocol.

    The engine resolves each :class:`ProposedAction` to a registered
    :class:`Skill` by keyword-matching the action's ``intent``. A miss is a
    capability gap, which triggers deterministic synthesis of a new skill,
    sandboxed validation, and a hot-swap into the live registry.

    Attributes:
        registry: Live mapping of capability keyword -> :class:`Skill`.
    """

    registry: dict[str, Skill] = field(default_factory=dict)
    _routes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Seed the registry with the deterministic builtin skills."""
        if not self.registry:
            for name, fn in _BUILTIN_SKILLS:
                self.registry[name] = Skill(
                    name=name,
                    fn=fn,
                    source="builtin",
                    synthesized=False,
                )
                # Builtins route by their capability keyword.
                self._routes[name] = name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def skills(self) -> list[str]:
        """Return the sorted names of every currently registered skill."""
        return sorted(self.registry)

    def execute(self, action: ProposedAction) -> ExecutionResult:
        """Execute *action*, synthesizing a skill if none already handles it.

        Resolution order:

        1. Keyword / route lookup in the existing registry.
        2. Skill *composition* – if two registered skills are both
           semantically close to *intent*, build a composed pipeline.
        3. Deterministic synthesis from source template.

        On a miss the engine synthesizes, validates, and registers a new
        skill, then runs it -- flagging ``synthesized=True`` on the result.
        """
        skill = self._resolve(action.intent)
        synthesized = False
        detail = ""

        if skill is None:
            # --- attempt composition before falling back to synthesis ---
            composed = self.compose_skills(action.intent)
            if composed is not None and self.validate_skill(composed):
                self.registry[composed.name] = composed
                self._routes[action.intent.lower()] = composed.name
                skill = composed
                synthesized = False
                detail = "composed"

        if skill is None:
            # --- last resort: deterministic synthesis ---
            candidate = self.synthesize_skill(action.intent)
            if not self.validate_skill(candidate):
                return ExecutionResult(
                    ok=False,
                    output=None,
                    skill_used=candidate.name,
                    synthesized=True,
                    detail=(
                        f"synthesized skill {candidate.name!r} failed sandbox "
                        "validation; not registered"
                    ),
                )
            self.registry[candidate.name] = candidate
            # Route the originating intent to the new skill so identical
            # future intents resolve it directly instead of re-synthesizing.
            self._routes[action.intent.lower()] = candidate.name
            skill = candidate
            synthesized = True

        try:
            output = skill.fn(action.intent, dict(action.params))
        except Exception as exc:  # pragma: no cover - defensive guard
            return ExecutionResult(
                ok=False,
                output=None,
                skill_used=skill.name,
                synthesized=synthesized,
                detail=f"skill {skill.name!r} raised: {exc!r}",
            )

        return ExecutionResult(
            ok=True,
            output=output,
            skill_used=skill.name,
            synthesized=synthesized,
            detail=detail,
        )

    def compose_skills(self, intent: str) -> Skill | None:
        """Attempt to build a composed :class:`Skill` from two registry candidates.

        A skill is a *candidate* for the given *intent* when either:

        * Its name appears as a substring of *intent* (lexical match), or
        * The cosine similarity between ``embed(skill_name)`` and
          ``embed(intent)`` exceeds :data:`_COMPOSITION_THRESHOLD`.

        When at least two distinct candidates are found, the first two are
        wired into a sequential pipeline: the first skill's output is passed
        to the second as ``params["prior"]``.  The composed :class:`Skill` is
        returned without being registered – the caller is responsible for
        validation and registration.

        Returns ``None`` when fewer than two candidates can be identified.

        Parameters
        ----------
        intent:
            The intent string driving capability resolution.

        Returns
        -------
        Skill | None:
            A composed skill, or ``None`` if composition is not possible.
        """
        lowered = intent.lower()
        intent_vec = embed(intent)

        candidates: list[Skill] = []
        for name, skill in self.registry.items():
            # Lexical match: skill name is a substring of the intent.
            is_lexical = name in lowered
            # Semantic match: cosine similarity above threshold.
            is_semantic = cosine(embed(name), intent_vec) > _COMPOSITION_THRESHOLD

            if is_lexical or is_semantic:
                candidates.append(skill)
            if len(candidates) == 2:
                break  # we only need two

        if len(candidates) < 2:
            return None

        first, second = candidates[0], candidates[1]
        composed_name = f"composed_{first.name}__{second.name}"

        def _composed_fn(
            intent: str,
            params: dict[str, Any],
            _first: Skill = first,
            _second: Skill = second,
        ) -> dict[str, Any]:
            """Call *first*, pass its result as ``params["prior"]`` to *second*."""
            prior = _first.fn(intent, dict(params))
            merged_params = dict(params)
            merged_params["prior"] = prior
            return _second.fn(intent, merged_params)

        return Skill(
            name=composed_name,
            fn=_composed_fn,
            source="composed",
            synthesized=False,
        )

    def synthesize_skill(self, intent: str) -> Skill:
        """Generate, compile, and isolate a new skill for *intent*.

        The function source is produced from a deterministic template, compiled,
        and executed in a fresh namespace whose builtins are restricted to
        :data:`_SAFE_BUILTINS`. The resulting callable is wrapped in a
        :class:`Skill` but is *not* registered here -- the caller registers it
        only after :meth:`validate_skill` passes.
        """
        name = self._skill_name(intent)
        source = self._render_source(name, intent)

        # Isolated namespace: a brand-new dict with a restricted ``__builtins__``
        # mapping. Synthesized code therefore cannot see the host module globals.
        namespace: dict[str, Any] = {"__builtins__": dict(_SAFE_BUILTINS)}
        code = compile(source, filename=f"<metamorph:{name}>", mode="exec")
        exec(code, namespace)  # noqa: S102 - sandboxed, author-controlled template

        fn = namespace[name]
        return Skill(name=name, fn=fn, source=source, synthesized=True)

    def validate_skill(self, skill: Skill) -> bool:
        """Run *skill* on a safe probe and confirm it behaves correctly.

        A skill passes when it executes without raising and returns a ``dict``
        (the structured shape every skill in this layer produces).
        """
        try:
            result = skill.fn(_PROBE_INTENT, {})
        except Exception:
            return False
        return isinstance(result, dict)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, intent: str) -> Skill | None:
        """Return the registered skill handling *intent*, if any.

        Resolution first honours an exact intent route (used by previously
        synthesized skills), then falls back to keyword routing where a
        registered capability keyword appears within the intent text.
        """
        lowered = intent.lower()
        exact = self._routes.get(lowered)
        if exact is not None and exact in self.registry:
            return self.registry[exact]
        for keyword in sorted(self._routes):
            if keyword in lowered and self._routes[keyword] in self.registry:
                return self.registry[self._routes[keyword]]
        return None

    def _skill_name(self, intent: str) -> str:
        """Derive a deterministic, valid Python identifier from *intent*."""
        slug = _IDENTIFIER_SCRUBBER.sub("_", intent.lower()).strip("_")
        slug = slug or "intent"
        if slug[0].isdigit():
            slug = f"_{slug}"
        return f"synth_{slug}"

    def _render_source(self, name: str, intent: str) -> str:
        """Render deterministic source for a benign string-transform skill.

        The generated function returns a structured ``dict`` describing the
        handled intent. It references only names available in the sandbox and
        performs no I/O.
        """
        return (
            f"def {name}(intent, params):\n"
            f"    tokens = intent.split()\n"
            f"    return {{\n"
            f"        'skill': {name!r},\n"
            f"        'synthesized': True,\n"
            f"        'handled_intent': {intent!r},\n"
            f"        'intent': intent,\n"
            f"        'tokens': tokens,\n"
            f"        'token_count': len(tokens),\n"
            f"        'transformed': intent[::-1],\n"
            f"        'params': dict(params),\n"
            f"    }}\n"
        )
