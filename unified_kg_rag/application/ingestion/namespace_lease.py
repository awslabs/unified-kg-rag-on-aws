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

A request that passed the fence can still land after the lease expired:
its process was frozen (a paused VM or container, a stopped process)
between the check and the write, or the request took long. Each attempt
of a request is bounded by the store's request timeout, so every write of
a run whose lease expired has landed by that expiry plus the longest
request. A run that takes over an *expired* lease (its holder died or
froze rather than released it) therefore holds its first write until
then (``check``), so a late write of the previous holder lands before
this run's writes, which overwrite it, rather than after them. A run that
is stopped gives its lease up the same way, as expired (its pools do not
wait for the requests in flight on a stop); a run that ended or failed
releases it, and a released lease needs no wait. The registry writes
(DynamoDB) are fenced per attempt too, and one attempt counts in the
longest request.
"""

from __future__ import annotations

import functools
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable
from contextvars import Token
from datetime import datetime, timezone
from types import TracebackType
from typing import TYPE_CHECKING

from unified_kg_rag.shared import NamespaceLeaseError, get_logger, write_fence
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
# Added to the takeover wait (see the module docstring) for clock skew
# between the hosts of the two runs.
_TAKEOVER_MARGIN_SECONDS = 5.0
# How often a takeover wait checks whether the run was stopped.
_TAKEOVER_POLL_SECONDS = 1.0


def longest_write_request_seconds() -> float:
    """The longest single write request attempt of the indexing path (the
    Neptune and OpenSearch indexing request timeouts, one DynamoDB registry
    attempt); indexing.lease_seconds should be at least three times it."""
    return max(
        write_fence.INGESTION_REQUEST_TIMEOUT_SECONDS,
        write_fence.INDEXING_REQUEST_TIMEOUT_SECONDS,
        write_fence.REGISTRY_CONNECT_TIMEOUT_SECONDS
        + write_fence.REGISTRY_READ_TIMEOUT_SECONDS,
    )


def _when(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(timespec="seconds")


class NamespaceLease:
    """Hold ``namespace``'s write lease in ``store`` for a ``with`` block."""

    def __init__(
        self,
        store: DocStatusPort,
        namespace: str,
        seconds: float,
        interrupt: Callable[[], None] | None = None,
    ) -> None:
        """``interrupt`` raises when the run is to stop; a takeover wait
        calls it (see ``check``)."""
        self.store = store
        self.namespace = namespace
        self.seconds = seconds
        self.interrupt = interrupt
        # Before this time (epoch seconds) no write: a takeover of an
        # expired lease waits for the previous holder's late writes.
        self.writes_from = 0.0
        self.owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._renewed_at = 0.0
        self._lost: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._token: Token[tuple[WriteFence, ...]] | None = None
        # How the run gives the lease up when it ends (set by __exit__).
        self._give_up: Callable[[], None] = self._release
        # The expiry of the expired lease this run took over (see __enter__).
        self._previous_expiry = 0.0

    def __enter__(self) -> NamespaceLease:
        # Read first: a lease that is still there but expired was not
        # released (the holder died or froze), so its writes may be late.
        previous = self.store.lease_holder(self.namespace)
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
        # A lease still live when read was released before the acquire (its
        # run ended cleanly): no late writes to wait for.
        if previous is not None and previous[0] != self.owner and previous[1] <= sent:
            self._previous_expiry = previous[1]
            self.writes_from = (
                previous[1] + longest_write_request_seconds() + _TAKEOVER_MARGIN_SECONDS
            )
            if self.writes_from > sent:
                logger.warning(
                    "Took over the expired write lease of namespace '%s' from %s: "
                    "the first write waits until %s (%.0fs) for any write of that "
                    "run still in flight",
                    self.namespace,
                    previous[0],
                    _when(self.writes_from),
                    self.writes_from - sent,
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
        # Worker threads a stop left running (their pool does not wait for
        # them on a stop) still check this lease through their copy of the
        # run's context: they must fail once it is given up, before another
        # run can take it.
        held, self._lost = self._lost is None, "the run ended and released it"
        # A stop (SIGTERM's PipelineInterrupted, Ctrl-C, SystemExit) may leave
        # writes past the fence in flight: give the lease up as expired
        # (owner kept), so the next run waits for them as after a crash (see
        # __enter__). A run that ended or failed has none left (its pools
        # waited) and releases it.
        # A run that ends during its takeover wait (stopped, or failed before
        # writing) hands the wait on: the lease goes back expired at the
        # previous holder's expiry, so the next run waits as long.
        stopped = exc_type is not None and issubclass(
            exc_type, (KeyboardInterrupt, SystemExit)
        )
        if self.writes_from > time.time():
            self._give_up = functools.partial(self._expire, self._previous_expiry)
        elif stopped:
            self._give_up = functools.partial(self._expire, time.time())
        else:
            self._give_up = self._release
        self._stop.set()
        if self._thread is not None:
            # A renewal in flight may hang on an unreachable registry: the
            # run does not wait for it (the heartbeat gives up what it
            # renews once stopped).
            self._thread.join(_HEARTBEAT_JOIN_SECONDS)
        if held:
            self._give_up()

    def _expire(self, at: float) -> None:
        """Give the lease up as expired at ``at`` (epoch seconds, not after
        now), keeping this run as its owner."""
        try:
            self.store.try_acquire_lease(self.namespace, self.owner, at - time.time())
        except Exception as e:  # noqa: BLE001 - it expires by itself
            logger.warning(
                "Expiring the lease of namespace '%s' failed (%s); it "
                "expires within %gs",
                self.namespace,
                e,
                self.seconds,
            )

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
                # have given the lease up already: give it up again.
                self._give_up()
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
        """Before a write: raise :class:`NamespaceLeaseError` if the lease was
        lost, or has less than a third of its term left (no renewal
        succeeded since).

        After a takeover of an expired lease, wait first until the previous
        holder's writes have all landed (``writes_from``, see the module
        docstring), calling ``interrupt`` (a stop raises) and ending at once
        if the run ended."""
        while (left := self.writes_from - time.time()) > 0:
            if self.interrupt is not None:
                self.interrupt()
            if self._stop.wait(min(left, _TAKEOVER_POLL_SECONDS)):
                break
        self.check_held()

    def check_held(self) -> None:
        """:meth:`check` without the takeover wait: for a point that writes
        nothing (a stage start), which need not wait for the previous
        holder's late writes."""
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
