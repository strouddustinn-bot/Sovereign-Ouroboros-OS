"""NeuroSynth – Cross-Modal Embodied Imagination layer (the Mind's Eye).

Builds internal multi-sensory mental models in a latent buffer, fusing
semantic, visual, spatial, and auditory representations into unified latent
vectors, and proposes candidate solution prototypes before any real I/O is
performed.
"""

from sovereign_ouroboros_os.neurosynth.imagination import NeuroSynth

__all__ = ["NeuroSynth"]
