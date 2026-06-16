"""EthosCompiler – Executable Ethics layer.

Compiles high-level natural language ethical principles into callable runtime
predicates. Every action submitted to the Ouroboros loop is gated through the
compiled constraint set before execution.
"""

from ouroboros.ethos_compiler.compiler import (
    EthicalPrinciple,
    EthosCompiler,
    GateResult,
)

__all__ = ["EthicalPrinciple", "EthosCompiler", "GateResult"]
