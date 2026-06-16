"""Structural protocols each cognitive layer implements.

The Ouroboros loop depends only on these protocols, so any layer can be
swapped for an alternative implementation as long as it satisfies the shape.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ouroboros.core.types import (
    ExecutionResult,
    FederatedResult,
    Prototype,
    ProposedAction,
    Timeline,
    WorldState,
)


@runtime_checkable
class Imaginer(Protocol):
    """NeuroSynth: imagine candidate solutions before acting."""

    def imagine(self, task: str, k: int = 3) -> list[Prototype]: ...


@runtime_checkable
class Simulator(Protocol):
    """ChronoWeave: simulate futures and collapse to the best timeline."""

    def simulate(
        self, task: str, prototypes: list[Prototype], state: WorldState
    ) -> Timeline: ...


@runtime_checkable
class Validator(Protocol):
    """EthosCompiler: gate an action against compiled ethical constraints."""

    def gate(self, action: dict) -> object: ...


@runtime_checkable
class Evolver(Protocol):
    """MetaMorph: execute an action, synthesizing skills to fill gaps."""

    def execute(self, action: ProposedAction) -> ExecutionResult: ...


@runtime_checkable
class Expander(Protocol):
    """HiveMind: fragment a task across peers and synthesize results."""

    def expand(self, task: str, seed: object) -> FederatedResult: ...
