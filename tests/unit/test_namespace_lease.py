# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The write lease of an index namespace (AWS-free).

A run holds its namespace's lease for its whole duration, renews it from a
heartbeat thread, and stops before writing once it lost it.
"""

from __future__ import annotations

import time

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.incremental_runs import IncrementalRun
from unified_kg_rag.application.ingestion.namespace_lease import NamespaceLease
from unified_kg_rag.shared import NamespaceLeaseError
from unified_kg_rag.shared.write_fence import check_write_fence

pytestmark = pytest.mark.unit


def test_a_held_lease_stops_the_second_run_and_names_the_holder() -> None:
    store = FakeDocStatusStore()
    with NamespaceLease(store, "ns", 60) as first:
        with (
            pytest.raises(NamespaceLeaseError, match=first.owner),
            NamespaceLease(store, "ns", 60),
        ):
            pass
        # Another namespace is leased apart.
        with NamespaceLease(store, "other", 60):
            pass
    # Released on exit: the next run takes it.
    assert store.lease_holder("ns") is None
    with NamespaceLease(store, "ns", 60):
        pass


def test_the_heartbeat_renews_the_lease() -> None:
    store = FakeDocStatusStore()
    with NamespaceLease(store, "ns", 0.3) as lease:
        time.sleep(0.5)
        # Renewed every 0.1 s: still held, so another run is refused.
        assert store.lease_holder("ns")[0] == lease.owner
        assert not store.try_acquire_lease("ns", "intruder", 60)
        check_write_fence()


def test_a_run_that_lost_its_lease_stops_before_writing() -> None:
    store = FakeDocStatusStore()
    with NamespaceLease(store, "ns", 0.3):
        # The lease expired (renewals failed) and another run took it.
        store.leases["ns"] = ("intruder", time.time() + 60)
        time.sleep(0.25)
        with pytest.raises(NamespaceLeaseError, match="lost the write lease"):
            check_write_fence()
    # The lost lease is not released over the new holder.
    assert store.lease_holder("ns")[0] == "intruder"
    check_write_fence()  # outside a leased run: no-op


def test_failed_renewals_lose_the_lease_once_it_expired(monkeypatch) -> None:
    store = FakeDocStatusStore()

    def unreachable(*_args: object) -> bool:
        raise RuntimeError("registry unreachable")

    with NamespaceLease(store, "ns", 0.3):
        monkeypatch.setattr(store, "try_acquire_lease", unreachable)
        time.sleep(0.45)
        with pytest.raises(NamespaceLeaseError, match="expired"):
            check_write_fence()


def test_incremental_writes_check_the_lease() -> None:
    harness = IncrementalRun()
    corpus = {"/c/a.txt": [[("Vendor", "Depot", 1)]]}
    with NamespaceLease(harness.store, "default", 0.3):
        harness.store.leases["default"] = ("intruder", time.time() + 60)
        time.sleep(0.25)
        with pytest.raises(NamespaceLeaseError):
            harness.run(corpus)
    # Nothing was written: neither registry record nor artifact.
    assert harness.store.list_all() == []
    assert harness.graph.ids("entities") == set()


class _FrozenRenewal(FakeDocStatusStore):
    """A renewal that reaches the store ``delay`` seconds after it was sent
    (a frozen process, a slow registry), once."""

    def __init__(self) -> None:
        super().__init__()
        self.delay = 0.0
        self.renewals = 0

    def try_acquire_lease(self, namespace: str, owner: str, seconds: float) -> bool:
        delay, self.delay = self.delay, 0.0
        time.sleep(delay)
        self.renewals += 1
        return super().try_acquire_lease(namespace, owner, seconds)


def test_a_renewal_after_the_lease_expired_loses_it() -> None:
    # Old behaviour: the late renewal took the free lease again (another run
    # had taken, written and released it meanwhile), the next renewal reset
    # the clock and the run went on writing its stale delta.
    store = _FrozenRenewal()
    with NamespaceLease(store, "ns", 1.2) as lease:
        store.delay = 1.6  # the next renewal, sent at 0.2 s, lands at 1.8 s
        time.sleep(1.4)
        # The lease expired at 1.2 s: another run takes it and finishes.
        assert FakeDocStatusStore.try_acquire_lease(store, "ns", "other", 1.2)
        store.release_lease("ns", "other")
        time.sleep(1.0)  # later renewals (every 0.2 s) would have succeeded
        with pytest.raises(NamespaceLeaseError, match="expired before a renewal"):
            lease.check()
        # The late renewal's lease was given back.
        assert store.lease_holder("ns") is None


def test_a_retaken_lease_is_trusted_for_a_new_term() -> None:
    # retake() renews the term the run trusts: a lease lost because it
    # nearly expired (no renewal for two thirds of the term) passes the
    # fence again once retaken, instead of failing the next write.
    store = FakeDocStatusStore()
    with NamespaceLease(store, "ns", 60) as lease:
        lease._renewed_at -= 50  # no renewal answered for 50 of 60 s
        with pytest.raises(NamespaceLeaseError, match="nearly expired"):
            lease.check()
        assert lease.retake()
        lease.check()


def _short_requests(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    import unified_kg_rag.application.ingestion.namespace_lease as lease_mod
    from unified_kg_rag.shared import write_fence

    monkeypatch.setattr(write_fence, "INGESTION_REQUEST_TIMEOUT_SECONDS", seconds)
    monkeypatch.setattr(write_fence, "INDEXING_REQUEST_TIMEOUT_SECONDS", seconds)
    monkeypatch.setattr(lease_mod, "_TAKEOVER_MARGIN_SECONDS", 0.1)
    monkeypatch.setattr(lease_mod, "_TAKEOVER_POLL_SECONDS", 0.05)


def test_a_takeover_of_an_expired_lease_waits_for_late_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: a run frozen with a request past its fence (a paused
    # host) wrote it after the run that took over its expired lease had
    # committed, and the stale write stayed. The new holder's first write
    # now waits until the old expiry plus the longest request.
    _short_requests(monkeypatch, 0.6)
    store = FakeDocStatusStore()
    store.leases["ns"] = ("frozen-run", time.time() - 0.1)  # expired, not released
    with NamespaceLease(store, "ns", 60) as lease:
        started = time.time()
        check_write_fence()
        waited = time.time() - started
        check_write_fence()  # once waited, writes go through at once
        again = time.time() - started - waited
    assert 0.4 <= waited < 2
    assert again < 0.1
    assert lease.writes_from > 0


def test_a_released_lease_needs_no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    _short_requests(monkeypatch, 30)
    store = FakeDocStatusStore()
    with NamespaceLease(store, "ns", 60):
        started = time.time()
        check_write_fence()
    assert time.time() - started < 0.5


def test_a_takeover_wait_ends_on_a_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    from unified_kg_rag.shared import PipelineInterrupted

    _short_requests(monkeypatch, 30)
    store = FakeDocStatusStore()
    store.leases["ns"] = ("dead-run", time.time() - 1)
    stop_at = time.time() + 0.3

    def interrupt() -> None:
        if time.time() >= stop_at:
            raise PipelineInterrupted(15)

    with NamespaceLease(store, "ns", 60, interrupt=interrupt):
        started = time.time()
        with pytest.raises(PipelineInterrupted):
            check_write_fence()
    assert time.time() - started < 2


def test_a_late_write_of_a_dead_holder_lands_before_the_new_holders_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The scenario of the concurrent-run harness, in time order: run A dies
    # (or freezes) holding the lease with a write past its fence; that write
    # lands a moment after the lease expired. Run B takes the expired lease;
    # its first write must come after A's late one, which it then
    # overwrites. Before, B wrote at once and A's stale write stayed.
    import threading

    _short_requests(monkeypatch, 0.6)
    store = FakeDocStatusStore()
    expiry = time.time() + 0.3
    store.leases["ns"] = ("frozen-run", expiry)  # never released
    value: list[str] = []

    def late_write() -> None:
        time.sleep(max(0.0, expiry + 0.3 - time.time()))
        value.append("stale")  # within expiry + the longest request

    writer = threading.Thread(target=late_write)
    writer.start()
    while time.time() < expiry:
        time.sleep(0.01)
    with NamespaceLease(store, "ns", 60):
        check_write_fence()
        value.append("fresh")
    writer.join()
    assert value == ["stale", "fresh"]
