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
- a heartbeat thread renews it every third of ``indexing.lease_seconds``, so
  a run that dies stops renewing and its lease expires (DynamoDB also drops
  the expired item, ``lease_expires_at`` being the table's TTL attribute);
- every registry and store write of the incremental path first calls
  :func:`check_lease`: a run whose lease another took (renewals failed past
  its expiry) stops instead of writing over the new holder.
"""

from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from types import TracebackType
from typing import TYPE_CHECKING

from unified_kg_rag.shared import NamespaceLeaseError, get_logger

if TYPE_CHECKING:
    from unified_kg_rag.ports import DocStatusPort

logger = get_logger(__name__)

# The lease of the run in this context (the pipeline sets it; worker threads
# inherit it through ContextThreadPoolExecutor).
_ACTIVE: ContextVar[NamespaceLease | None] = ContextVar("namespace_lease", default=None)


def check_lease() -> None:
    """Raise :class:`NamespaceLeaseError` if this run's lease was lost (no-op
    outside a leased run)."""
    lease = _ACTIVE.get()
    if lease is not None:
        lease.check()


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
        self._token: Token[NamespaceLease | None] | None = None

    def __enter__(self) -> NamespaceLease:
        if not self.store.try_acquire_lease(self.namespace, self.owner, self.seconds):
            holder = self.store.lease_holder(self.namespace)
            who, until = holder if holder else ("another run", time.time())
            raise NamespaceLeaseError(
                f"Index namespace '{self.namespace}' is being written by {who} "
                f"(its lease runs until {_when(until)} unless renewed). Wait "
                "for that run to finish; a run that died stops renewing, and "
                f"its lease expires within indexing.lease_seconds ({self.seconds:g}s)."
            )
        self._renewed_at = time.time()
        logger.info(
            "Holding the write lease of namespace '%s' (%s)", self.namespace, self.owner
        )
        self._thread = threading.Thread(
            target=self._heartbeat, name=f"lease-{self.namespace}", daemon=True
        )
        self._thread.start()
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._token is not None:
            _ACTIVE.reset(self._token)
            self._token = None
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._lost is None:
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
        while not self._stop.wait(self.seconds / 3):
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
                if time.time() - self._renewed_at >= self.seconds:
                    self._lose("its renewals failed until it expired")
                    return
                continue
            if not held:
                self._lose("another run took it")
                return
            self._renewed_at = time.time()

    def _lose(self, why: str) -> None:
        self._lost = why
        logger.error(
            "Lost the write lease of namespace '%s': %s; the run stops before "
            "its next write",
            self.namespace,
            why,
        )

    def check(self) -> None:
        """Raise :class:`NamespaceLeaseError` if the lease was lost."""
        if self._lost is None and time.time() - self._renewed_at >= self.seconds:
            self._lost = "it expired before a renewal succeeded"
        if self._lost is not None:
            raise NamespaceLeaseError(
                f"This run lost the write lease of index namespace "
                f"'{self.namespace}' ({self._lost}): stopping before writing "
                "over the run that holds it now. Re-run once that run ended."
            )
