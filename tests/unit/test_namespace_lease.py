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
    longest_write_request_seconds,
)
from unified_kg_rag.shared import NamespaceLeaseError
from unified_kg_rag.shared.exceptions import PipelineInterrupted
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
    monkeypatch.setattr(write_fence, "REGISTRY_CONNECT_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(write_fence, "REGISTRY_READ_TIMEOUT_SECONDS", seconds)
    monkeypatch.setattr(lease_mod, "_TAKEOVER_MARGIN_SECONDS", 0.1)
    monkeypatch.setattr(lease_mod, "_TAKEOVER_POLL_SECONDS", 0.05)
    # The real bound (the test session turns the wait off, see conftest).
    monkeypatch.setattr(
        lease_mod, "longest_write_request_seconds", longest_write_request_seconds
    )


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


@pytest.mark.parametrize(
    ("ending", "expired"),
    [
        (None, False),
        (RuntimeError, False),
        (GeneratorExit, False),  # not a stop: the pools waited for their writes
        (KeyboardInterrupt, True),
        (SystemExit, True),  # sys.exit() in a signal handler or a host
    ],
)
def test_a_stopped_run_gives_its_lease_up_as_expired(
    monkeypatch: pytest.MonkeyPatch, ending: type[BaseException] | None, expired: bool
) -> None:
    # Regression: a stopped run released its lease while requests already
    # past the fence were in flight (its pools do not wait on a stop), so
    # the next run took a released lease and wrote at once, before them.
    # Now a stop leaves the lease expired, owner kept, and the next run
    # waits as after a crash; an ended or failed run still releases it.
    _short_requests(monkeypatch, 30)
    store = FakeDocStatusStore()
    try:
        with NamespaceLease(store, "ns", 60):
            if ending is not None:
                raise ending
    except BaseException:  # noqa: BLE001 - the ending under test
        pass
    holder = store.lease_holder("ns")
    assert (holder is not None) == expired
    with NamespaceLease(store, "ns", 60) as nxt:
        assert (nxt.writes_from > time.time()) == expired


def test_a_lease_released_between_the_read_and_the_acquire_needs_no_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: the lease was read live, then its run released it before
    # the acquire; the takeover still waited a full term plus the bound.
    _short_requests(monkeypatch, 30)
    store = FakeDocStatusStore()
    store.leases["ns"] = ("ending-run", time.time() + 60)
    read = store.lease_holder

    def read_then_release(namespace: str):  # type: ignore[no-untyped-def]
        holder = read(namespace)
        store.release_lease(namespace, "ending-run")
        return holder

    monkeypatch.setattr(store, "lease_holder", read_then_release)
    with NamespaceLease(store, "ns", 60) as lease:
        assert lease.writes_from == 0.0


def test_a_run_stopped_during_its_takeover_wait_hands_the_wait_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: B, waiting for the late writes of A (who froze), was
    # stopped and released the lease; C then took a released lease, wrote at
    # once, and A's late write landed after C's. B now puts the lease back
    # expired at A's expiry, so C waits until the same time.
    from unified_kg_rag.shared import PipelineInterrupted

    _short_requests(monkeypatch, 30)
    store = FakeDocStatusStore()
    a_expiry = time.time() - 1
    store.leases["ns"] = ("frozen-run-a", a_expiry)
    with pytest.raises(PipelineInterrupted):
        with NamespaceLease(store, "ns", 60) as b:
            raise PipelineInterrupted(15)  # stopped while waiting
    holder = store.lease_holder("ns")
    assert holder is not None and holder[0] == b.owner
    assert holder[1] == pytest.approx(a_expiry, abs=0.5)
    # What B's stop message and JSON status give as C's first write time.
    assert b.next_writes_from == b.writes_from
    with NamespaceLease(store, "ns", 60) as c:
        assert c.writes_from == pytest.approx(b.writes_from, abs=0.5)


@pytest.mark.parametrize("handed_on", [True, False])
def test_the_stop_warning_names_whose_writes_may_be_in_flight(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, handed_on: bool
) -> None:
    # A run stopped during its takeover wait wrote nothing: the writes the
    # next run waits for are those of the run it took the lease over from.
    from unified_kg_rag.shared import PipelineInterrupted

    _short_requests(monkeypatch, 30)
    store = FakeDocStatusStore()
    if handed_on:
        store.leases["ns"] = ("frozen-run", time.time() - 1)
    with pytest.raises(PipelineInterrupted):
        with NamespaceLease(store, "ns", 60):
            raise PipelineInterrupted(15)
    (warning,) = [r.getMessage() for r in caplog.records if "Gave up" in r.getMessage()]
    whose = "of the run whose expired lease" if handed_on else "of this run"
    assert whose in warning


def test_a_stage_start_does_not_wait_for_the_takeover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A stage start checks the lease but writes nothing: the takeover wait
    # holds the first write, not parsing or extraction.
    from unified_kg_rag.shared.write_fence import check_fences_held

    _short_requests(monkeypatch, 30)
    store = FakeDocStatusStore()
    store.leases["ns"] = ("dead-run", time.time() - 1)
    with NamespaceLease(store, "ns", 60) as lease:
        started = time.time()
        check_fences_held()
        assert time.time() - started < 0.5
        assert lease.writes_from > time.time()
        lease._lose("taken")
        with pytest.raises(NamespaceLeaseError):
            check_fences_held()  # a lost lease still stops it


def test_a_takeover_wait_ends_when_the_run_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A worker thread waiting for the takeover when its run ends (the lease
    # is given up) stops waiting at once and fails its write.
    import threading

    _short_requests(monkeypatch, 30)
    store = FakeDocStatusStore()
    store.leases["ns"] = ("dead-run", time.time() - 1)
    outcome: list[BaseException] = []

    def write(lease: NamespaceLease) -> None:
        try:
            lease.check()
        except BaseException as e:  # noqa: BLE001 - collected
            outcome.append(e)

    with NamespaceLease(store, "ns", 60) as lease:
        worker = threading.Thread(target=write, args=(lease,))
        worker.start()
        time.sleep(0.2)
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert [type(e) for e in outcome] == [NamespaceLeaseError]


def test_the_pipeline_lease_ends_a_takeover_wait_on_a_stop() -> None:
    # The pipeline passes its stop to the lease (interrupt), so a SIGTERM
    # also ends a worker thread's takeover wait.
    import signal

    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
    from unified_kg_rag.domain.models import Config
    from unified_kg_rag.shared import PipelineInterrupted

    pipe = object.__new__(DataIngestionPipeline)
    pipe.config = Config()
    pipe._doc_status = FakeDocStatusStore()
    pipe._stop_signal = None
    lease = pipe._namespace_lease()
    assert isinstance(lease, NamespaceLease) and lease.interrupt is not None
    lease.interrupt()  # no stop requested: nothing happens
    pipe._stop_signal = signal.SIGTERM
    with pytest.raises(PipelineInterrupted):
        lease.interrupt()


def test_a_stopped_run_on_in_process_stores_releases_its_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On the embedded stores the stopped process takes its writes with it:
    # the next run need not wait (it waited ~3 minutes after each Ctrl-C).
    _short_requests(monkeypatch, 30)
    store = FakeDocStatusStore()
    with pytest.raises(KeyboardInterrupt):
        with NamespaceLease(store, "ns", 60, in_process_stores=True) as lease:
            raise KeyboardInterrupt
    assert store.lease_holder("ns") is None
    assert lease.next_writes_from == 0.0


def test_a_stopped_run_tells_when_the_next_run_may_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _short_requests(monkeypatch, 30)
    store = FakeDocStatusStore()
    started = time.time()
    with pytest.raises(KeyboardInterrupt):
        with NamespaceLease(store, "ns", 60) as lease:
            raise KeyboardInterrupt
    assert lease.next_writes_from >= started + 30
    with NamespaceLease(store, "ns", 60) as nxt:
        assert nxt.writes_from == pytest.approx(lease.next_writes_from, abs=1)


def test_an_in_process_takeover_waits_one_in_process_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A frozen local holder's late write is one SQLite write, bounded by the
    # stores' busy timeout, not by a store request.
    _short_requests(monkeypatch, 30)
    store = FakeDocStatusStore()
    expiry = time.time() - 1
    store.leases["ns"] = ("frozen-local-run", expiry)
    with NamespaceLease(
        store, "ns", 60, in_process_stores=True, in_process_write_seconds=2
    ) as lease:
        assert lease.writes_from == pytest.approx(expiry + 2 + 0.1, abs=0.01)


def test_a_renewal_landing_after_a_stop_leaves_the_lease_expired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The run stops while a renewal is in flight; the end of the run gives
    # up waiting for it and expires the lease (a stop's writes may still be
    # in flight). The renewal then lands and renews the lease: the heartbeat
    # must give it up the same way, as expired, not release it (a released
    # lease lets the next run write at once, before the stopped run's
    # writes still in flight).
    import unified_kg_rag.application.ingestion.namespace_lease as lease_mod

    monkeypatch.setattr(lease_mod, "_HEARTBEAT_JOIN_SECONDS", 0.05)
    store = _FrozenRenewal()
    with pytest.raises(KeyboardInterrupt):
        with NamespaceLease(store, "ns", 3.0):
            store.delay = 1.0  # the first renewal, sent at 0.5 s, lands at 1.5 s
            time.sleep(0.7)
            raise KeyboardInterrupt
    stopped = time.time()
    # The acquire, the stop's expire, the late renewal, then the heartbeat
    # giving that renewal up.
    deadline = stopped + 5
    while store.renewals < 4 and time.time() < deadline:
        time.sleep(0.05)
    assert store.renewals >= 3  # the late renewal landed
    holder = store.lease_holder("ns")
    assert holder is not None, "released, not expired"
    assert holder[1] <= stopped


@pytest.mark.parametrize(
    ("ended_with", "stop"),
    [
        (None, KeyboardInterrupt()),
        (None, PipelineInterrupted(15)),
        (KeyboardInterrupt, KeyboardInterrupt()),
    ],
)
def test_a_stop_raised_while_the_lease_exits_still_gives_it_up(
    ended_with: type[BaseException] | None, stop: BaseException
) -> None:
    # Regression: a signal handler raising in the heartbeat join skipped the
    # give-up, so the lease stayed held for lease_seconds and the rerun
    # failed with "being written by" (14 of 250 end-of-run signals).
    store = FakeDocStatusStore()
    lease = NamespaceLease(store, "ns", 600.0)
    lease.__enter__()
    thread = lease._thread
    assert thread is not None

    def join(*_args: object) -> None:
        raise stop

    thread.join = join  # type: ignore[method-assign]
    with pytest.raises(type(stop)):
        lease.__exit__(ended_with, ended_with() if ended_with else None, None)
    holder = store.lease_holder("ns")
    assert holder is None or holder[1] <= time.time()
    assert lease._stop.is_set()
    with NamespaceLease(store, "ns", 60):
        pass


def _ctrl_c() -> None:
    import os
    import signal

    os.kill(os.getpid(), signal.SIGINT)
    time.sleep(0.05)  # the handler runs at the next bytecode boundary


class _SignalledGiveUp(FakeDocStatusStore):
    """Receives SIGINT (Ctrl-C) while the lease is being given up."""

    def release_lease(self, namespace: str, owner: str) -> None:
        _ctrl_c()
        super().release_lease(namespace, owner)


@pytest.mark.parametrize("at", ["fence removal", "heartbeat join", "give-up"])
def test_a_stop_landing_anywhere_in_the_lease_exit_still_gives_it_up(
    monkeypatch: pytest.MonkeyPatch, at: str
) -> None:
    # Regression: a Ctrl-C whose handler raised before the give-up's
    # try (at the fence's removal) or inside the give-up request itself left
    # the lease held for lease_seconds. Now the stop is held until the lease
    # is given up, and then raised.
    import unified_kg_rag.application.ingestion.namespace_lease as lease_mod

    store = _SignalledGiveUp() if at == "give-up" else FakeDocStatusStore()
    if at == "fence removal":
        remove = lease_mod.remove_write_fence

        def signalled_removal(token: object) -> None:
            remove(token)  # type: ignore[arg-type]
            _ctrl_c()

        monkeypatch.setattr(lease_mod, "remove_write_fence", signalled_removal)
    with pytest.raises(KeyboardInterrupt):
        with NamespaceLease(store, "ns", 600.0) as lease:
            if at == "heartbeat join":
                thread = lease._thread
                assert thread is not None
                join = thread.join

                def signalled_join(timeout: float | None = None) -> None:
                    _ctrl_c()
                    join(timeout)

                thread.join = signalled_join  # type: ignore[method-assign]
    assert store.lease_holder("ns") is None
