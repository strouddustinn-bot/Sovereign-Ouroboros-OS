"""HiveMind core: privacy-preserving federated problem solving.

HiveMind models a small decentralized network of simulated **peer nodes**.
A complex task is fragmented into encrypted **shares** using genuine additive
secret sharing over ``GF(256)`` (byte-wise XOR), so that no single peer ever
holds the plaintext task. Each peer computes a partial result on its own
share -- learning only random-looking bytes -- and the coordinator synthesizes
those partials into a collective answer after verifying that the shares
round-trip back to the original task.

The sharing scheme is *deterministic per task*: the random masks are derived
from a task-seeded keystream so that :meth:`HiveMind.expand` reproduces exactly
for the loop, CLI, and tests, while every individual share remains
non-trivial (never equal to the plaintext for a multi-peer network).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ouroboros.core.types import FederatedResult


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeerNode:
    """A simulated sovereign peer in the HiveMind network.

    Attributes:
        id: Stable identifier such as ``"peer-0"``.
    """

    id: str

    def compute(self, share: bytes) -> dict[str, object]:
        """Compute a partial result over *share* without the plaintext.

        The peer never receives the full task; it only sees one encrypted
        share. It returns a deterministic digest plus a couple of local
        metrics derived solely from the bytes it holds.
        """
        digest = hashlib.sha256(self.id.encode("utf-8") + share).hexdigest()
        return {
            "peer": self.id,
            "digest": digest,
            "share_bytes": len(share),
            "checksum": sum(share) % 256,
        }


@dataclass
class HiveMind:
    """Federated Intelligence layer satisfying the :class:`Expander` protocol.

    HiveMind fragments a task across ``n_peers`` peers using additive secret
    sharing over ``GF(256)``. The first ``n_peers - 1`` shares are pseudo-random
    masks derived deterministically from the task; the final share is the
    plaintext XORed with every mask, so that XORing all shares together
    reconstructs the original bytes. Each share on its own is uninformative.

    Usage::

        hive = HiveMind(n_peers=5)
        result = hive.expand("optimize the routing table", seed=0)
        assert result.shards == 5
        assert result.reconstructed["collective_answer"]
    """

    n_peers: int = 5
    peers: list[PeerNode] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_peers < 1:
            raise ValueError("HiveMind requires at least one peer")
        if not self.peers:
            self.peers = [PeerNode(id=f"peer-{i}") for i in range(self.n_peers)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self, task: str, seed: object = 0) -> FederatedResult:
        """Fragment *task* across peers and synthesize a collective answer.

        The task is split into one share per peer; each peer computes a
        partial result on its share alone. The coordinator reconstructs the
        secret to verify integrity (asserting an exact round-trip) and then
        aggregates the partials into a derived collective answer.
        """
        shares = self.shard(task)
        partials = [peer.compute(share) for peer, share in zip(self.peers, shares)]

        # Integrity: reconstruction must exactly recover the original task.
        recovered = self.reconstruct(shares)
        if recovered != task:
            raise AssertionError("HiveMind reconstruction failed integrity check")

        reconstructed = self._synthesize(task, partials, seed)
        return FederatedResult(
            task=task,
            reconstructed=reconstructed,
            contributors=[peer.id for peer in self.peers],
            shards=len(shares),
        )

    def shard(self, task: str) -> list[bytes]:
        """Split *task* into one secret share per peer.

        Returns ``n_peers`` shares of equal length. XORing all shares
        together reconstructs the UTF-8 bytes of *task*. Each individual
        share is random-looking and -- for ``n_peers > 1`` -- never equal to
        the plaintext bytes.
        """
        secret = task.encode("utf-8")
        n = len(self.peers)
        if n == 1:
            return [bytes(secret)]

        masks = [self._keystream(task, index, len(secret)) for index in range(n - 1)]

        last = bytearray(secret)
        for mask in masks:
            for i in range(len(last)):
                last[i] ^= mask[i]

        return [bytes(mask) for mask in masks] + [bytes(last)]

    def reconstruct(self, shares: list[bytes]) -> str:
        """Recombine *shares* into the original task string (inverse of shard)."""
        if not shares:
            return ""
        length = len(shares[0])
        accumulator = bytearray(length)
        for share in shares:
            if len(share) != length:
                raise ValueError("all shares must share the same length")
            for i in range(length):
                accumulator[i] ^= share[i]
        return accumulator.decode("utf-8")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _keystream(self, task: str, index: int, length: int) -> bytes:
        """Derive a deterministic, task-seeded pseudo-random mask.

        The mask is the prefix of a SHA-256 counter-mode keystream keyed by
        the task and the peer *index*. This makes :meth:`shard` reproducible
        while keeping each mask independent and uniformly distributed.
        """
        key = f"hivemind:{task}:share-{index}".encode("utf-8")
        out = bytearray()
        counter = 0
        while len(out) < length:
            block = hashlib.sha256(key + counter.to_bytes(8, "big")).digest()
            out.extend(block)
            counter += 1
        return bytes(out[:length])

    def _synthesize(
        self, task: str, partials: list[dict[str, object]], seed: object
    ) -> dict[str, object]:
        """Combine peer partials into a derived collective answer.

        The collective answer is a deterministic digest folded from every
        peer's contribution (and the seed), with a confidence that scales
        with the breadth of participation.
        """
        folded = hashlib.sha256()
        folded.update(str(seed).encode("utf-8"))
        for partial in partials:
            folded.update(str(partial["digest"]).encode("utf-8"))

        collective = folded.hexdigest()
        confidence = round(1.0 - 1.0 / (len(partials) + 1), 4)
        return {
            "collective_answer": collective,
            "confidence": confidence,
            "contributions": partials,
            "peers": len(partials),
        }
