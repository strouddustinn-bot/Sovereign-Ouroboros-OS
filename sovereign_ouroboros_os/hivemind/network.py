"""asyncio TCP network layer for HiveMind federated intelligence.

Each peer runs as a real TCP server bound to a dynamically-assigned localhost
port.  Shares are transmitted over genuine socket connections using a simple
length-prefixed JSON framing protocol, so the privacy guarantees of the
additive secret-sharing scheme are exercised over the network stack rather
than entirely in-process.

Protocol
--------
Both request and response frames use the same framing::

    ┌──────────────────────────────┬────────────────────────────────────┐
    │  4 bytes (big-endian uint32) │  N bytes (UTF-8 JSON)              │
    │  N = length of body          │                                    │
    └──────────────────────────────┴────────────────────────────────────┘

Request body::

    {"peer_id": "<str>", "share": [<int>, ...]}

Response body::

    {"peer_id": "<str>", "digest": "<hex>", "share_bytes": <int>,
     "checksum": <int>}

Usage
-----
::

    async with NetworkHiveMind(n_peers=3) as hive:
        result = await hive.expand("plan the mission", seed=42)

Or synchronously (auto-manages the event loop)::

    hive = NetworkHiveMind(n_peers=5)
    result = hive.expand_sync("plan the mission", seed=42)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import struct
from dataclasses import dataclass, field
from typing import Any

from sovereign_ouroboros_os.core.types import FederatedResult
from sovereign_ouroboros_os.hivemind.federation import HiveMind

# ---------------------------------------------------------------------------
# Framing helpers
# ---------------------------------------------------------------------------

_LENGTH_PREFIX_FMT = "!I"  # big-endian unsigned 32-bit int
_LENGTH_PREFIX_SIZE = struct.calcsize(_LENGTH_PREFIX_FMT)


async def _send_frame(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
    """Encode *payload* as a length-prefixed JSON frame and write it."""
    body = json.dumps(payload).encode("utf-8")
    header = struct.pack(_LENGTH_PREFIX_FMT, len(body))
    writer.write(header + body)
    await writer.drain()


async def _recv_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Read a length-prefixed JSON frame and decode it."""
    raw_len = await reader.readexactly(_LENGTH_PREFIX_SIZE)
    (length,) = struct.unpack(_LENGTH_PREFIX_FMT, raw_len)
    raw_body = await reader.readexactly(length)
    return json.loads(raw_body.decode("utf-8"))


# ---------------------------------------------------------------------------
# Peer server logic
# ---------------------------------------------------------------------------


async def _handle_peer_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Handle a single share computation request from the coordinator.

    Reads one request frame, computes a partial result (SHA-256 of the
    share bytes together with the peer id), and writes one response frame.
    The connection is closed after each request/response pair.
    """
    try:
        request = await _recv_frame(reader)
        peer_id: str = request["peer_id"]
        share = bytes(request["share"])  # list[int] → bytes

        # Mirror the PeerNode.compute logic so results are comparable.
        digest = hashlib.sha256(peer_id.encode("utf-8") + share).hexdigest()
        response: dict[str, Any] = {
            "peer_id": peer_id,
            "digest": digest,
            "share_bytes": len(share),
            "checksum": sum(share) % 256,
        }
        await _send_frame(writer, response)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass  # tolerate platforms where wait_closed() is unavailable


# ---------------------------------------------------------------------------
# NetworkHiveMind
# ---------------------------------------------------------------------------


@dataclass
class NetworkHiveMind:
    """HiveMind variant that routes shares over real asyncio TCP connections.

    Each of the ``n_peers`` logical peers is represented by an actual
    ``asyncio`` TCP server bound to a dynamically-assigned localhost port.
    The coordinator opens a fresh connection to every peer for each
    :meth:`expand` call, sends a share as a length-prefixed JSON frame,
    and receives the partial result the same way.

    Lifecycle
    ---------
    **Async context manager** (preferred)::

        async with NetworkHiveMind(n_peers=3) as hive:
            result = await hive.expand("solve this")

    **Explicit async**::

        hive = NetworkHiveMind()
        await hive.start()
        result = await hive.expand("solve this")
        await hive.stop()

    **Synchronous** (auto-manages event loop per call)::

        result = NetworkHiveMind().expand_sync("solve this")

    Attributes:
        n_peers:  Number of peer servers to spin up.  Defaults to 5.
    """

    n_peers: int = 5

    # Private state – not part of the public dataclass interface.
    _hive: HiveMind = field(init=False, repr=False)
    _servers: list[asyncio.AbstractServer] = field(
        init=False, repr=False, default_factory=list
    )
    _peer_addresses: list[tuple[str, int]] = field(
        init=False, repr=False, default_factory=list
    )
    _peer_ids: list[str] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.n_peers < 1:
            raise ValueError("NetworkHiveMind requires at least one peer")
        self._hive = HiveMind(n_peers=self.n_peers)
        self._servers = []
        self._peer_addresses = []
        self._peer_ids = [f"peer-{i}" for i in range(self.n_peers)]

    # ------------------------------------------------------------------
    # Async lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all peer TCP servers and record their (host, port) addresses.

        Binds each server to ``("127.0.0.1", 0)`` so the OS assigns a free
        port.  The assigned ports are stored in :attr:`_peer_addresses`.
        """
        if self._servers:
            return  # already started

        for _ in range(self.n_peers):
            server = await asyncio.start_server(
                _handle_peer_connection,
                host="127.0.0.1",
                port=0,  # let the OS pick a free port
            )
            # Retrieve the dynamically-assigned port from the first socket.
            bound_socket = server.sockets[0]
            host, port = bound_socket.getsockname()[:2]
            self._peer_addresses.append((host, port))
            self._servers.append(server)
            server.start_serving if hasattr(server, "start_serving") else None

    async def stop(self) -> None:
        """Shut down all peer TCP servers gracefully."""
        for server in self._servers:
            server.close()
            await server.wait_closed()
        self._servers.clear()
        self._peer_addresses.clear()

    # ------------------------------------------------------------------
    # Async context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "NetworkHiveMind":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Synchronous context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "NetworkHiveMind":
        """Start servers in a new (temporary) event loop for sync usage."""
        asyncio.run(self.start())
        return self

    def __exit__(self, *_: object) -> None:
        asyncio.run(self.stop())

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def expand(self, task: str, seed: object = 0) -> FederatedResult:
        """Fragment *task* across peer TCP servers and synthesize the result.

        This is the async counterpart of :meth:`HiveMind.expand`.  It uses
        the same :class:`HiveMind` secret-sharing logic to produce shares,
        transmits each share to the corresponding peer server over a real TCP
        connection, collects the partial results, and synthesises them with
        the same aggregation logic.

        Args:
            task:  The plaintext task to distribute.
            seed:  Determinism seed passed to the synthesiser.

        Returns:
            A :class:`~sovereign_ouroboros_os.core.types.FederatedResult`
            whose ``shards`` equals ``n_peers`` and whose ``contributors``
            lists every peer id.
        """
        auto_started = not self._servers
        if auto_started:
            await self.start()

        try:
            shares = self._hive.shard(task)
            partials = await asyncio.gather(
                *(
                    self._call_peer(peer_id, addr, share)
                    for peer_id, addr, share in zip(
                        self._peer_ids, self._peer_addresses, shares
                    )
                )
            )

            # Integrity check: reconstruct and verify round-trip.
            recovered = self._hive.reconstruct(shares)
            if recovered != task:
                raise AssertionError(
                    "NetworkHiveMind reconstruction failed integrity check"
                )

            reconstructed = self._synthesize(task, list(partials), seed)
            return FederatedResult(
                task=task,
                reconstructed=reconstructed,
                contributors=[p["peer_id"] for p in partials],
                shards=len(shares),
            )
        finally:
            if auto_started:
                await self.stop()

    def expand_sync(self, task: str, seed: object = 0) -> FederatedResult:
        """Synchronous wrapper around :meth:`expand`.

        Suitable for callers (such as the OuroborosLoop) that run in a purely
        synchronous context.  Manages its own ``asyncio`` event loop: if the
        servers have not been started yet they are auto-started and
        auto-stopped within the single ``asyncio.run()`` invocation.

        If the servers are already running (e.g. the caller used the context
        manager), a temporary loop is used only for the ``expand`` coroutine.

        Args:
            task:  The plaintext task to distribute.
            seed:  Determinism seed passed to the synthesiser.

        Returns:
            A :class:`~sovereign_ouroboros_os.core.types.FederatedResult`.
        """
        return asyncio.run(self.expand(task, seed))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_peer(
        self,
        peer_id: str,
        address: tuple[str, int],
        share: bytes,
    ) -> dict[str, Any]:
        """Open a TCP connection to one peer, send the share, get the partial.

        Args:
            peer_id:  Logical identifier of the peer (e.g. ``"peer-0"``).
            address:  ``(host, port)`` of the peer's TCP server.
            share:    The secret share bytes destined for this peer.

        Returns:
            The decoded response dict from the peer server.
        """
        host, port = address
        reader, writer = await asyncio.open_connection(host, port)
        try:
            await _send_frame(writer, {"peer_id": peer_id, "share": list(share)})
            response = await _recv_frame(reader)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        return response

    def _synthesize(
        self, task: str, partials: list[dict[str, Any]], seed: object
    ) -> dict[str, Any]:
        """Aggregate peer partial results into a collective answer.

        Mirrors :meth:`HiveMind._synthesize` exactly so outputs are
        structurally identical and the existing suite's shape assertions hold.

        Args:
            task:     Original plaintext task (used for binding context).
            partials: List of dicts returned by each peer server.
            seed:     Determinism seed.

        Returns:
            Dict with keys ``collective_answer``, ``confidence``,
            ``contributions``, and ``peers``.
        """
        folded = hashlib.sha256()
        folded.update(str(seed).encode("utf-8"))
        for partial in partials:
            folded.update(str(partial["digest"]).encode("utf-8"))

        collective = folded.hexdigest()
        confidence = round(1.0 - 1.0 / (len(partials) + 1), 4)

        # Reformat network partials to match PeerNode.compute output shape.
        contributions = [
            {
                "peer": p["peer_id"],
                "digest": p["digest"],
                "share_bytes": p["share_bytes"],
                "checksum": p["checksum"],
            }
            for p in partials
        ]

        return {
            "collective_answer": collective,
            "confidence": confidence,
            "contributions": contributions,
            "peers": len(partials),
        }
