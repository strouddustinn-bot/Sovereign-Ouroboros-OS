"""MetaMorph – Self-Modifying Architecture / Evolutionary Engine.

Detects capability gaps at execution time, synthesizes new Python skill modules
from deterministic templates, validates them in an isolated sandbox, and
hot-swaps them into a live skill registry without restarting the process.
"""

from ouroboros.metamorph.evolution import MetaMorph

__all__ = ["MetaMorph"]
