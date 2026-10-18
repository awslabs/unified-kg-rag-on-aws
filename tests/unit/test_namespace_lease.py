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
from unified_kg_rag.application.ingestion.namespace_lease import (
    NamespaceLease,
    check_lease,
)
from unified_kg_rag.shared import NamespaceLeaseError

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
        check_lease()


def test_a_run_that_lost_its_lease_stops_before_writing() -> None:
    store = FakeDocStatusStore()
    with NamespaceLease(store, "ns", 0.3):
        # The lease expired (renewals failed) and another run took it.
        store.leases["ns"] = ("intruder", time.time() + 60)
        time.sleep(0.25)
        with pytest.raises(NamespaceLeaseError, match="lost the write lease"):
            check_lease()
    # The lost lease is not released over the new holder.
    assert store.lease_holder("ns")[0] == "intruder"
    check_lease()  # outside a leased run: no-op


def test_failed_renewals_lose_the_lease_once_it_expired(monkeypatch) -> None:
    store = FakeDocStatusStore()

    def unreachable(*_args: object) -> bool:
        raise RuntimeError("registry unreachable")

    with NamespaceLease(store, "ns", 0.3):
        monkeypatch.setattr(store, "try_acquire_lease", unreachable)
        time.sleep(0.45)
        with pytest.raises(NamespaceLeaseError, match="expired"):
            check_lease()


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
