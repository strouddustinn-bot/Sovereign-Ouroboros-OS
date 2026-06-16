"""HiveMind – Federated Intelligence / Sovereign Node layer.

Fragments a complex task into encrypted secret-shares, distributes them across
a network of simulated peer nodes, and synthesizes their partial results into a
collective answer without any single peer ever observing the whole problem.
"""

from sovereign_ouroboros_os.hivemind.federation import HiveMind, PeerNode

__all__ = ["HiveMind", "PeerNode"]
