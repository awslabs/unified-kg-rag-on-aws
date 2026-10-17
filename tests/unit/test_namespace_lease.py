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
