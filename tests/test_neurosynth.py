"""Tests for the NeuroSynth cross-modal embodied imagination layer."""

from ouroboros.core.contracts import Imaginer
from ouroboros.core.types import Prototype
from ouroboros.neurosynth import NeuroSynth

_MODALITIES = {"semantic", "visual", "spatial", "auditory"}
_TASK = "plan a resilient backup routine"


def test_satisfies_imaginer_protocol():
    assert isinstance(NeuroSynth(), Imaginer)


def test_imagine_returns_exactly_k_prototypes():
    synth = NeuroSynth()
    for k in (1, 2, 3, 5):
        prototypes = synth.imagine(_TASK, k=k)
        assert len(prototypes) == k
        assert all(isinstance(p, Prototype) for p in prototypes)


def test_imagine_is_deterministic():
    a = NeuroSynth().imagine(_TASK, k=4)
    b = NeuroSynth().imagine(_TASK, k=4)
    assert a == b


def test_prototypes_sorted_by_descending_confidence():
    prototypes = NeuroSynth().imagine(_TASK, k=5)
    confidences = [p.confidence for p in prototypes]
    assert confidences == sorted(confidences, reverse=True)


def test_modality_scores_contains_all_modalities():
    for proto in NeuroSynth().imagine(_TASK, k=3):
        assert set(proto.modality_scores) == _MODALITIES


def test_confidence_within_unit_interval():
    for proto in NeuroSynth().imagine(_TASK, k=5):
        assert 0.0 <= proto.confidence <= 1.0


def test_latent_is_unit_norm_length_64():
    for proto in NeuroSynth().imagine(_TASK, k=3):
        assert len(proto.latent) == 64
        norm = sum(x * x for x in proto.latent) ** 0.5
        assert abs(norm - 1.0) < 1e-9


def test_prototypes_are_distinct():
    prototypes = NeuroSynth().imagine(_TASK, k=4)
    labels = {p.label for p in prototypes}
    latents = {p.latent for p in prototypes}
    assert len(labels) == 4
    assert len(latents) == 4


def test_last_buffer_exposes_modality_vectors():
    synth = NeuroSynth()
    synth.imagine(_TASK, k=2)
    buffer = synth.last_buffer
    assert set(buffer) == _MODALITIES
    assert all(len(vec) == 64 for vec in buffer.values())


def test_inspect_buffer_tracks_latest_task():
    synth = NeuroSynth()
    synth.imagine("first task", k=1)
    synth.imagine("second task", k=1)
    assert synth.inspect_buffer().task == "second task"


def test_k_is_clamped_to_at_least_one():
    assert len(NeuroSynth().imagine(_TASK, k=0)) == 1
