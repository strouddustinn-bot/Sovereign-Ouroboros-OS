"""HiveMind – Federated Intelligence / Sovereign Node layer.

Fragments a complex task into encrypted secret-shares, distributes them across
a network of simulated peer nodes, and synthesizes their partial results into a
collective answer without any single peer ever observing the whole problem.

``HiveMind`` (from :mod:`.federation`) simulates peers in-process.
``NetworkHiveMind`` (from :mod:`.network`) runs each peer as a real asyncio
TCP server so shares travel over genuine socket connections.
"""

from ouroboros.hivemind.federation import HiveMind, PeerNode
from ouroboros.hivemind.network import NetworkHiveMind

__all__ = ["HiveMind", "NetworkHiveMind", "PeerNode"]
