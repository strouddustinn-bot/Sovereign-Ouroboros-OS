"""NeuroSynth core: cross-modal embodied imagination.

This module implements the "Mind's Eye" of the Ouroboros loop. Before any real
I/O happens, NeuroSynth builds internal multi-sensory mental models of a task in
a latent buffer: it synthesizes per-modality embeddings (semantic, visual,
spatial, auditory), fuses them into a unified latent vector, and proposes a
handful of distinct candidate :class:`~ouroboros.core.types.Prototype`
solutions for the downstream simulation layer to reason about.

Everything is deterministic and dependency-free: the same ``(task, k)`` always
yields identical prototypes, derived solely from the deterministic pseudo-embedder
in :mod:`ouroboros.core.embedding`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ouroboros.core.embedding import blend, cosine, embed
from ouroboros.core.types import Prototype, Vector

# Modalities the Mind's Eye imagines, paired with the fusion weight each one
# contributes to the unified latent vector. Semantic meaning dominates, with the
# embodied senses adding texture.
_MODALITY_WEIGHTS: dict[str, float] = {
    "semantic": 1.0,
    "visual": 0.7,
    "spatial": 0.6,
    "auditory": 0.5,
}

# Distinct "framings" used to seed the k candidate prototypes. Each framing
# nudges the imagined latent in a different conceptual direction, so the k
# prototypes are distinct and deterministically ordered.
_FRAMINGS: tuple[str, ...] = (
    "direct",
    "analogical",
    "decompositional",
    "exploratory",
    "adversarial",
    "minimal",
    "holistic",
)


@dataclass
class _LatentBuffer:
    """The most recent multi-sensory mental model NeuroSynth imagined.

    Attributes:
        task:      The task text that produced this buffer.
        modalities: Per-modality latent vectors (e.g. ``{"visual": (...)}``).
        fused:     The unified latent vector blended from the modalities.
    """

    task: str = ""
    modalities: dict[str, Vector] = field(default_factory=dict)
    fused: Vector = ()


class NeuroSynth:
    """Cross-Modal Embodied Imagination layer (the Mind's Eye).

    Satisfies the :class:`~ouroboros.core.contracts.Imaginer`
    protocol. For a given task it imagines several distinct candidate solutions
    by synthesizing per-modality embeddings, fusing them into a unified latent
    vector, and scoring each modality's agreement with that fused view.

    Usage::

        synth = NeuroSynth()
        prototypes = synth.imagine("plan a backup routine", k=3)
        best = prototypes[0]                  # highest confidence
        buffer = synth.last_buffer            # introspect the Mind's Eye
    """

    def __init__(self) -> None:
        self._buffer = _LatentBuffer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def imagine(
        self, task: str, k: int = 3, context: list[str] | None = None
    ) -> list[Prototype]:
        """Imagine *k* distinct candidate prototypes for *task*.

        Synthesizes the multi-sensory latent buffer for *task*, then frames it
        *k* different ways into distinct :class:`Prototype` candidates. The
        result is sorted by descending confidence and is fully deterministic.

        Args:
            task:    Natural language description of the problem to imagine.
            k:       Number of candidate prototypes to generate (clamped to >= 1).
            context: Optional list of retrieved knowledge passages that ground
                     the imagination (e.g. from the KnowledgeBase recall stage).
                     When provided, the task's latent is blended with embeddings
                     of the context passages so imagination is anchored to
                     factual background knowledge.

        Returns:
            A list of exactly ``max(k, 1)`` prototypes, best confidence first.
        """
        k = max(int(k), 1)
        self._buffer = self._build_buffer(task, context=context or [])

        prototypes = [
            self._frame_prototype(task, framing, index)
            for index, framing in enumerate(self._framings_for(k))
        ]
        prototypes.sort(key=lambda proto: (-proto.confidence, proto.label))
        return prototypes

    @property
    def last_buffer(self) -> dict[str, Vector]:
        """Per-modality latent vectors of the most recent ``imagine`` call."""
        return dict(self._buffer.modalities)

    def inspect_buffer(self) -> _LatentBuffer:
        """Return the full latent buffer for introspecting the Mind's Eye."""
        return self._buffer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_buffer(self, task: str, context: list[str] | None = None) -> _LatentBuffer:
        """Synthesize and fuse the per-modality mental model for *task*.

        When *context* passages are supplied (e.g. from KB recall), the task's
        fused latent is blended with a context centroid so imagination is
        grounded in retrieved factual knowledge.
        """
        modalities = {
            modality: embed(f"{modality} {task}")
            for modality in _MODALITY_WEIGHTS
        }
        vectors = list(modalities.values())
        weights = [_MODALITY_WEIGHTS[name] for name in modalities]
        fused = blend(vectors, weights)

        if context:
            context_vecs = [embed(c) for c in context[:5]]  # cap at 5 passages
            context_centroid = blend(context_vecs)
            # Blend task latent (dominant) with knowledge centroid (subordinate)
            fused = blend([fused, context_centroid], [1.0, 0.4])

        return _LatentBuffer(task=task, modalities=modalities, fused=fused)

    def _frame_prototype(self, task: str, framing: str, index: int) -> Prototype:
        """Imagine a single candidate prototype under one *framing*.

        The framing seed perturbs the fused latent so each of the *k* prototypes
        is distinct, while modality scores are measured against the perturbed
        view and confidence is the mean modality agreement.
        """
        framed = blend(
            [self._buffer.fused, embed(f"{framing} framing {task}")],
            [1.0, 0.35],
        )
        modality_scores = {
            name: round((cosine(vector, framed) + 1.0) / 2.0, 6)
            for name, vector in self._buffer.modalities.items()
        }
        agreement = sum(modality_scores.values()) / len(modality_scores)
        confidence = round(min(1.0, max(0.0, agreement)), 6)
        label = f"Imagined approach #{index + 1}: {framing} framing of '{task}'"
        return Prototype(
            label=label,
            latent=framed,
            modality_scores=modality_scores,
            confidence=confidence,
        )

    def _framings_for(self, k: int) -> list[str]:
        """Return *k* distinct, deterministic framing seeds."""
        framings = list(_FRAMINGS[:k])
        # If more candidates are requested than predefined framings exist,
        # extend deterministically so labels stay unique.
        index = len(_FRAMINGS)
        while len(framings) < k:
            framings.append(f"variant-{index}")
            index += 1
        return framings
