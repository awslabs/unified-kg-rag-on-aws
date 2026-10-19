# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The write lease of an index namespace: one writing run at a time.

Two runs writing one namespace race on its registry records (each reads the
other's half-written state as its own) and on the stores (a reset clears what
the other indexes; an alias swap deletes the index the other writes). A
pipeline run with the doc-status registry therefore takes its namespace's
lease from the registry before its first stage and holds it until it ends:

- taking it is one conditional write (``DocStatusPort.try_acquire_lease``):
  it fails while another run holds an unexpired lease, and the run stops with
  :class:`NamespaceLeaseError` naming the holder, before it reads or writes
  anything;
- a heartbeat thread renews it every sixth of ``indexing.lease_seconds``, so
  a run that dies stops renewing and its lease expires (DynamoDB also drops
  the expired item, ``lease_expires_at`` being the table's TTL attribute);
- the lease is a write fence of the run (``shared.write_fence``): every
  registry write of the incremental path, every indexing phase and every
  store request that may write (a Neptune batch, edge write or drop, an
  OpenSearch indexing request but a GET or HEAD) first checks it; it fails
  once the run released it, and once less than a third of the term is
  left since the last renewal that succeeded, counted from when that
  renewal was *sent*. The store expires the lease no earlier than a term
  after the request was sent, so a run stops while it still holds the lease
  with a third of the term to spare, provided no single write request runs
  longer than that (keep ``indexing.lease_seconds`` at least three times the
  longest request).

The registry's own records are not fenced by the store: DynamoDB cannot
condition a ``BatchWriteItem`` (the batched registry writes) on another
item, and the Neptune and OpenSearch writes cannot be conditioned on the
registry at all, so the margin above is the fence for every write.
"""

from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from contextvars import Token
from datetime import datetime, timezone
from types import TracebackType
from typing import TYPE_CHECKING

from unified_kg_rag.shared import NamespaceLeaseError, get_logger
from unified_kg_rag.shared.write_fence import (
    WriteFence,
    install_write_fence,
    remove_write_fence,
)

if TYPE_CHECKING:
    from unified_kg_rag.ports import DocStatusPort

logger = get_logger(__name__)

# Renewals per term: the run keeps writing through three failed renewals in
# a row, and stops at the fourth with a third of the term still in reserve
# (check()).
_RENEWALS_PER_TERM = 6
# How long the end of a run waits for a renewal in flight.
_HEARTBEAT_JOIN_SECONDS = 5.0


def _when(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(timespec="seconds")


class NamespaceLease:
    """Hold ``namespace``'s write lease in ``store`` for a ``with`` block."""

    def __init__(self, store: DocStatusPort, namespace: str, seconds: float) -> None:
        self.store = store
        self.namespace = namespace
        self.seconds = seconds
        self.owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._renewed_at = 0.0
        self._lost: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._token: Token[tuple[WriteFence, ...]] | None = None

    def __enter__(self) -> NamespaceLease:
        sent = time.time()
        if not self.store.try_acquire_lease(self.namespace, self.owner, self.seconds):
            holder = self.store.lease_holder(self.namespace)
            who, until = holder if holder else ("another run", time.time())
            raise NamespaceLeaseError(
                f"Index namespace '{self.namespace}' is being written by {who} "
                f"(its lease runs until {_when(until)} unless renewed). Wait "
                "for that run to finish; a run that died stops renewing, and "
                f"its lease expires within indexing.lease_seconds ({self.seconds:g}s)."
            )
        self._renewed_at = sent
        logger.info(
            "Holding the write lease of namespace '%s' (%s)", self.namespace, self.owner
        )
        self._thread = threading.Thread(
            target=self._heartbeat, name=f"lease-{self.namespace}", daemon=True
        )
        self._thread.start()
        self._token = install_write_fence(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._token is not None:
            remove_write_fence(self._token)
            self._token = None
        # Worker threads a stop or a failure left running (their pool does
        # not wait for them) still check this lease through their copy of
        # the run's context: they must fail once it is released, before
        # another run can take it.
        held, self._lost = self._lost is None, "the run ended and released it"
        self._stop.set()
        if self._thread is not None:
            # A renewal in flight may hang on an unreachable registry: the
            # run does not wait for it (the heartbeat releases what it
            # renews once stopped).
            self._thread.join(_HEARTBEAT_JOIN_SECONDS)
        if held:
            self._release()

    def _release(self) -> None:
        try:
            self.store.release_lease(self.namespace, self.owner)
        except Exception as e:  # noqa: BLE001 - it expires by itself
            logger.warning(
                "Releasing the lease of namespace '%s' failed (%s); it "
                "expires within %gs",
                self.namespace,
                e,
                self.seconds,
            )

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.seconds / _RENEWALS_PER_TERM):
            # The store's expiry runs from when the renewal was sent, never
            # earlier: a slow answer only shortens what this run trusts.
            sent = time.time()
            try:
                held = self.store.try_acquire_lease(
                    self.namespace, self.owner, self.seconds
                )
            except Exception as e:  # noqa: BLE001 - retried until it expires
                logger.warning(
                    "Renewing the lease of namespace '%s' failed: %s",
                    self.namespace,
                    e,
                )
                if self._expiring():
                    self._lose("its renewals failed until it nearly expired")
                    return
                continue
            if not held:
                self._lose("another run took it")
                return
            if time.time() - self._renewed_at >= self.seconds:
                # The lease expired before this renewal answered (the process
                # was frozen, or the registry slow): another run may have
                # taken it, written and released it meanwhile, and the
                # renewal then took a free lease. Give it back.
                self._lose("it expired before a renewal succeeded")
                self._release()
                return
            self._renewed_at = sent
            if self._stop.is_set():
                # The run ended while this renewal was in flight, and may
                # have released the lease already: release it again.
                self._release()
                return

    def _expiring(self) -> bool:
        """Less than a third of the term left since the last renewal sent
        that succeeded."""
        return time.time() - self._renewed_at >= self.seconds * 2 / 3

    def _lose(self, why: str) -> None:
        self._lost = why
        logger.error(
            "Lost the write lease of namespace '%s': %s; the run stops before "
            "its next write",
            self.namespace,
            why,
        )

    def check(self) -> None:
        """Raise :class:`NamespaceLeaseError` if the lease was lost, or has
        less than a third of its term left (no renewal succeeded since)."""
        if self._lost is None and self._expiring():
            self._lost = "it nearly expired before a renewal succeeded"
        if self._lost is not None:
            raise NamespaceLeaseError(
                f"This run lost the write lease of index namespace "
                f"'{self.namespace}' ({self._lost}): stopping before writing "
                "over the run that holds it now. Re-run once that run ended."
            )

    def retake(self) -> bool:
        """Take the lease back after losing it, when no other run holds it
        (it was released, or expired) or the store still has it as this
        run's; returns whether this run holds it again. One conditional
        write, like a renewal, so it never takes another run's lease; the
        heartbeat is not restarted, so the term it buys is for a last short
        write (closing the run's attempts)."""
        sent = time.time()
        try:
            held = self.store.try_acquire_lease(
                self.namespace, self.owner, self.seconds
            )
        except Exception as e:  # noqa: BLE001 - the lease stays lost
            logger.warning(
                "Retaking the lease of namespace '%s' failed: %s", self.namespace, e
            )
            return False
        if held:
            self._lost = None
            self._renewed_at = sent
        return held
