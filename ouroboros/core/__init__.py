"""Core contracts and shared types for the Ouroboros cognitive stack."""

from ouroboros.core.contracts import (
    Evolver,
    Expander,
    Imaginer,
    Simulator,
    Validator,
)
from ouroboros.core.embedding import blend, cosine, embed
from ouroboros.core.types import (
    ExecutionResult,
    FederatedResult,
    Prototype,
    ProposedAction,
    Skill,
    Timeline,
    Vector,
    WorldState,
)

__all__ = [
    "Evolver",
    "Expander",
    "Imaginer",
    "Simulator",
    "Validator",
    "blend",
    "cosine",
    "embed",
    "ExecutionResult",
    "FederatedResult",
    "Prototype",
    "ProposedAction",
    "Skill",
    "Timeline",
    "Vector",
    "WorldState",
]
