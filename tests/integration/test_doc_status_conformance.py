# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One DocStatusPort conformance suite for every registry implementation.

The in-memory fake is the reference; the SQLite store (local mode) and the
DynamoDB store (moto-mocked, AWS-free) must behave identically on the port
surface: round trips, batch reads and writes, the diff partition with
``stored_scopes``, and the namespace partitions. The last tests drive whole
interrupted incremental runs through each store.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import boto3
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from moto import mock_aws

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.incremental_runs import (
    POINTS,
    Corpus,
    IncrementalRun,
    expected_state,
)
from unified_kg_rag.adapters.doc_status.dynamodb import DynamoDBDocStatusStore
from unified_kg_rag.adapters.doc_status.sqlite import SQLiteDocStatusStore
from unified_kg_rag.domain.models import (
    PENDING_CONTENT_HASH,
    Config,
    DocStatus,
    DocStatusRecord,
    ExtractionSettings,
)
from unified_kg_rag.ports import DocStatusPort
from unified_kg_rag.shared import DocStatusRegistryError

pytestmark = pytest.mark.integration


def _fake(_: Path) -> Iterator[DocStatusPort]:
    yield FakeDocStatusStore()


def _sqlite(tmp_path: Path) -> Iterator[DocStatusPort]:
    store = SQLiteDocStatusStore(Config(), tmp_path / "registry" / "doc-status.db")
    yield store
    store.close()


def _dynamodb(_: Path) -> Iterator[DocStatusPort]:
    with mock_aws():
        config = Config()
        config.providers.doc_status.enabled = True
        config.aws.dynamodb.table_name = "conformance-doc-status"
        store = DynamoDBDocStatusStore(
            config, boto_session=boto3.Session(region_name="us-east-1")
        )
        _ = store.client
        yield store


STORES: dict[str, Callable[[Path], Iterator[DocStatusPort]]] = {
    "fake": _fake,
    "sqlite": _sqlite,
    "dynamodb": _dynamodb,
}


@pytest.fixture(params=sorted(STORES))
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[DocStatusPort]:
    yield from STORES[request.param](tmp_path)


def _record(doc_id: str, content_hash: str = "h", **fields: object) -> DocStatusRecord:
    return DocStatusRecord(doc_id=doc_id, content_hash=content_hash, **fields)


def test_conforms_to_the_port(store: DocStatusPort) -> None:
    assert isinstance(store, DocStatusPort)


def test_round_trip_keeps_every_field(store: DocStatusPort) -> None:
    record = _record(
        "d1",
        "h1",
        status=DocStatus.PROCESSED,
        suffix="tenant-a",
        scope="ns|corpus-a",
        file_path="docs/d1.txt",
        text_unit_ids=["t2", "t1"],
        error_info=None,
        attempt_count=2,
        attempted_hash="h0",
        attempt_batch=3,
        crash_batch=4,
        extraction_fingerprint="x1",
        updated_at="2026-01-01T00:00:00",
    )
    store.put(record)
    fetched = store.get("d1")
    assert fetched is not None
    # DynamoDB stores the id list as a set; compare it order-free.
    assert fetched.model_dump(exclude={"text_unit_ids"}) == record.model_dump(
        exclude={"text_unit_ids"}
    )
    assert sorted(fetched.text_unit_ids) == ["t1", "t2"]


def test_missing_and_deleted_records(store: DocStatusPort) -> None:
    assert store.get("nope") is None
    store.put(_record("d1"))
    store.delete("d1")
    store.delete("never-stored")
    assert store.get("d1") is None
    assert store.list_all() == []


def test_put_overwrites(store: DocStatusPort) -> None:
    store.put(_record("d1", "h1"))
    store.put(_record("d1", "h2", status=DocStatus.FAILED, error_info="boom"))
    fetched = store.get("d1")
    assert fetched is not None
    assert (fetched.content_hash, fetched.status) == ("h2", DocStatus.FAILED)
    assert len(store.list_all()) == 1


def test_get_many_skips_unknown_ids(store: DocStatusPort) -> None:
    store.put_many([_record(f"d{i}", f"h{i}") for i in range(30)])
    found = store.get_many(["d3", "missing", "d17", "d3"])
    assert set(found) == {"d3", "d17"}
    assert found["d17"].content_hash == "h17"
    assert store.get_many([]) == {}


def test_put_many_last_record_per_doc_wins(store: DocStatusPort) -> None:
    store.put_many([_record("d1", "old"), _record("d2"), _record("d1", "new")])
    assert {r.doc_id: r.content_hash for r in store.list_all()} == {
        "d1": "new",
        "d2": "h",
    }
    store.put_many([])


def test_diff_partitions_like_the_reference(store: DocStatusPort) -> None:
    reference = FakeDocStatusStore()
    for target in (store, reference):
        target.put_many(
            [
                _record("same", "h1", status=DocStatus.PROCESSED, scope="s1"),
                _record("edited", "h2", status=DocStatus.PROCESSED, scope="s1"),
                _record("retry", "h3", status=DocStatus.FAILED, scope="s1"),
                _record("gone", "h4", status=DocStatus.PROCESSED, scope="s1"),
                _record("other", "h5", status=DocStatus.PROCESSED, scope="s2"),
                _record("legacy", "h6", status=DocStatus.PROCESSED),
                _record("busy", PENDING_CONTENT_HASH, scope="s1"),
            ]
        )
    incoming = {
        "same": "h1",
        "edited": "h2-new",
        "retry": "h3",
        "fresh": "h7",
        "busy": "h8",
    }

    for scope in (None, "s1", "s2", "unknown"):
        got = store.diff(incoming, scope=scope)
        want = reference.diff(incoming, scope=scope)
        assert sorted(got.new) == sorted(want.new), scope
        assert sorted(got.changed) == sorted(want.changed), scope
        assert sorted(got.unchanged) == sorted(want.unchanged), scope
        assert sorted(got.deleted) == sorted(want.deleted), scope
        assert got.stored_scopes == want.stored_scopes == ["s1", "s2"]

    scoped = store.diff(incoming, scope="s1")
    assert sorted(scoped.changed) == ["busy", "edited", "retry"]
    # A record without a scope is never deleted by a scoped run.
    assert scoped.deleted == ["gone"]
    assert sorted(store.diff(incoming).deleted) == ["gone", "legacy", "other"]


def test_diff_reextracts_records_of_other_extraction_settings(
    store: DocStatusPort,
) -> None:
    reference = FakeDocStatusStore()
    for target in (store, reference):
        target.put_many(
            [
                _record(
                    "current",
                    "h1",
                    status=DocStatus.PROCESSED,
                    scope="s",
                    extraction_fingerprint="x-new",
                ),
                _record(
                    "stale",
                    "h2",
                    status=DocStatus.PROCESSED,
                    scope="s",
                    extraction_fingerprint="x-old",
                ),
                # Written before records carried a fingerprint.
                _record("legacy", "h3", status=DocStatus.PROCESSED, scope="s"),
                _record(
                    "edited",
                    "h4",
                    status=DocStatus.PROCESSED,
                    scope="s",
                    extraction_fingerprint="x-old",
                ),
                _record(
                    "retry",
                    "h5",
                    status=DocStatus.FAILED,
                    scope="s",
                    extraction_fingerprint="x-old",
                ),
            ]
        )
    incoming = {
        "current": "h1",
        "stale": "h2",
        "legacy": "h3",
        "edited": "h4-new",
        "retry": "h5",
    }

    got = store.diff(incoming, scope="s", extraction_fingerprint="x-new")
    want = reference.diff(incoming, scope="s", extraction_fingerprint="x-new")
    assert sorted(got.changed) == sorted(want.changed) == ["edited", "retry", "stale"]
    assert sorted(got.unchanged) == sorted(want.unchanged) == ["current", "legacy"]
    # Only the documents that read changed because of the settings alone.
    assert got.settings_changed == want.settings_changed == ["stale"]
    # Without a fingerprint the diff compares content and status only.
    plain = store.diff(incoming, scope="s")
    assert sorted(plain.unchanged) == ["current", "legacy", "stale"]
    assert plain.settings_changed == []


def test_diff_of_an_empty_registry_marks_everything_new(store: DocStatusPort) -> None:
    delta = store.diff({"a": "1", "b": "2"}, scope="s")
    assert sorted(delta.new) == ["a", "b"]
    assert delta.deleted == delta.stored_scopes == []


def test_a_quarantined_record_is_unchanged_until_its_content_changes(
    store: DocStatusPort,
) -> None:
    store.put(_record("held", "h1", status=DocStatus.QUARANTINED, scope="s"))
    store.put(_record("edited", "h2", status=DocStatus.QUARANTINED, scope="s"))

    delta = store.diff({"held": "h1", "edited": "h2-new"}, scope="s")

    assert (delta.unchanged, delta.changed) == (["held"], ["edited"])


def test_one_owner_holds_a_namespace_lease_at_a_time(store: DocStatusPort) -> None:
    assert store.lease_holder("ns") is None
    assert store.try_acquire_lease("ns", "run-a", 60)
    # Held: another owner is refused, the holder renews, and namespaces are
    # leased apart.
    assert not store.try_acquire_lease("ns", "run-b", 60)
    assert store.try_acquire_lease("ns", "run-a", 60)
    assert store.try_acquire_lease("other", "run-b", 60)
    owner, expires_at = store.lease_holder("ns")
    assert owner == "run-a" and expires_at > time.time() + 50
    # Releasing someone else's lease does nothing; the holder's release frees it.
    store.release_lease("ns", "run-b")
    assert not store.try_acquire_lease("ns", "run-b", 60)
    store.release_lease("ns", "run-a")
    assert store.lease_holder("ns") is None
    assert store.try_acquire_lease("ns", "run-b", 60)


def test_an_expired_lease_can_be_taken(store: DocStatusPort) -> None:
    assert store.try_acquire_lease("ns", "run-a", 0.2)
    time.sleep(0.3)
    assert store.try_acquire_lease("ns", "run-b", 60)
    assert store.lease_holder("ns")[0] == "run-b"


def test_a_lease_is_no_record(store: DocStatusPort) -> None:
    store.put(_record("ns#d1", scope="ns|corpus"))
    store.try_acquire_lease("ns", "run-a", 60)

    assert [r.doc_id for r in store.list_namespace("ns")] == ["ns#d1"]
    assert [r.doc_id for r in store.list_all()] == ["ns#d1"]
    delta = store.diff({}, scope="ns|corpus")
    assert delta.deleted == ["ns#d1"]


def test_extraction_settings_are_kept_per_scope_and_are_no_record(
    store: DocStatusPort,
) -> None:
    assert store.get_extraction_settings("ns|corpus-a") is None
    settings = ExtractionSettings(version=1, values={"a.b": 1, "c": ["x", "y"]})
    store.put(_record("ns#d1", scope="ns|corpus-a"))
    store.put_extraction_settings("ns|corpus-a", "fp-1", settings)
    store.put_extraction_settings("ns|corpus-b", "fp-2", ExtractionSettings(version=1))
    assert store.get_extraction_settings("ns|corpus-a") == ("fp-1", settings)
    assert store.get_extraction_settings("ns|corpus-b")[0] == "fp-2"

    newer = ExtractionSettings(version=1, values={"a.b": 2})
    store.put_extraction_settings("ns|corpus-a", "fp-3", newer)
    assert store.get_extraction_settings("ns|corpus-a") == ("fp-3", newer)
    # Never listed or diffed as a document.
    assert [r.doc_id for r in store.list_namespace("ns")] == ["ns#d1"]
    assert [r.doc_id for r in store.list_all()] == ["ns#d1"]
    assert store.diff({}, scope="ns|corpus-a").deleted == ["ns#d1"]


def test_namespaces_partition_the_registry(store: DocStatusPort) -> None:
    # A registry key names its index namespace (compute_doc_id): a scoped
    # diff and list_namespace read that namespace's records only.
    store.put_many(
        [
            _record("tenant-a#1", scope="tenant-a|corpus"),
            _record("tenant-a#2", scope="tenant-a|other"),
            _record("tenant-b#1", scope="tenant-b|corpus"),
            _record("plain", scope="default|corpus"),
        ]
    )

    delta = store.diff({"tenant-a#1": "h"}, scope="tenant-a|corpus")
    assert delta.unchanged == ["tenant-a#1"] and delta.deleted == []
    assert delta.stored_scopes == ["tenant-a|corpus", "tenant-a|other"]
    assert sorted(r.doc_id for r in store.list_namespace("tenant-a")) == [
        "tenant-a#1",
        "tenant-a#2",
    ]
    # A key without a namespace belongs to the default one.
    assert [r.doc_id for r in store.list_namespace("default")] == ["plain"]
    assert store.list_namespace("tenant-c") == []
    assert len(store.list_all()) == 4


# --- diff partition properties ----------------------------------------------

# doc_id -> content_hash maps; ids are short tokens, hashes arbitrary text.
_ids = st.text(alphabet="abcdefghij", min_size=1, max_size=4)
_corpus = st.dictionaries(_ids, st.text(min_size=1, max_size=8), max_size=10)

# One store per test, emptied at the start of every example; the store fixture
# is function-scoped, which is safe here because each example resets it. No
# deadline: an example is moto or SQLite I/O, which overruns the 200 ms default
# on a loaded machine (the ci profile already drops it; local runs did not).
_per_store = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _reset(store: DocStatusPort, records: dict[str, str]) -> None:
    for record in store.list_all():
        store.delete(record.doc_id)
    store.put_many(
        [_record(doc_id, content_hash) for doc_id, content_hash in records.items()]
    )


@_per_store
@given(stored=_corpus, incoming=_corpus)
def test_diff_partitions_are_disjoint_and_complete(
    store: DocStatusPort, stored: dict[str, str], incoming: dict[str, str]
) -> None:
    _reset(store, stored)
    delta = store.diff(incoming)

    # new + changed + unchanged exactly partitions the incoming keys.
    classified = set(delta.new) | set(delta.changed) | set(delta.unchanged)
    assert classified == set(incoming)
    assert len(delta.new) + len(delta.changed) + len(delta.unchanged) == len(incoming)
    # deleted are exactly the stored ids absent from incoming.
    assert set(delta.deleted) == set(stored) - set(incoming)


@_per_store
@given(corpus=_corpus)
def test_diff_unchanged_when_corpus_identical(
    store: DocStatusPort, corpus: dict[str, str]
) -> None:
    _reset(store, {})
    store.put_many(
        [
            _record(doc_id, content_hash, status=DocStatus.PROCESSED)
            for doc_id, content_hash in corpus.items()
        ]
    )
    delta = store.diff(corpus)

    assert delta.is_empty is True
    assert set(delta.unchanged) == set(corpus)


# --- whole incremental runs --------------------------------------------------

R0: Corpus = {
    "/c/a.txt": [[("Vendor", "Depot", 1)]],
    "/c/b.txt": [[("Vendor", "Carrier", 2)], [("Depot", "Carrier", 1)]],
    "/c/c.txt": [[("Buyer", "Bank", 1)]],
}
R1: Corpus = {
    "/c/a.txt": R0["/c/a.txt"],
    "/c/b.txt": [[("Vendor", "Carrier", 2)], [("Carrier", "Bank", 2)]],
    "/c/e.txt": [[("Vendor", "Bank", 1)]],
}


@pytest.mark.parametrize("point", [None, *POINTS])
def test_interrupted_incremental_run_converges(
    store: DocStatusPort, point: str | None
) -> None:
    harness = IncrementalRun(store)
    harness.run(R0)
    harness.run(R1, interrupt=point)
    harness.run(R1)
    assert harness.state() == expected_state(R1)
    assert harness.registry_problems(R1) == []
    harness.run(R1)
    assert harness.extracted == []


# --- SQLite specifics --------------------------------------------------------


def test_sqlite_registry_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "doc-status.db"
    first = SQLiteDocStatusStore(Config(), path)
    first.put(_record("d1", "h1", status=DocStatus.PROCESSED, scope="s"))
    first.close()

    second = SQLiteDocStatusStore(Config(), path)
    assert second.get("d1") is not None
    assert second.diff({"d1": "h1"}, scope="s").unchanged == ["d1"]
    second.close()


def test_sqlite_locked_registry_waits_store_timeout_seconds(tmp_path: Path) -> None:
    # Another process's write lock is waited for search.store_timeout_seconds
    # (not sqlite3's 5 s default), then named in the error.
    path = tmp_path / "doc-status.db"
    config = Config(search={"store_timeout_seconds": 0.2})
    store = SQLiteDocStatusStore(config, path)
    store.put(_record("d1"))
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        with pytest.raises(DocStatusRegistryError, match="stayed locked for 0.2s"):
            store.put(_record("d2"))
        assert time.monotonic() - started < 3
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    store.put(_record("d2"))
    assert {r.doc_id for r in store.list_all()} == {"d1", "d2"}
    store.close()


def test_sqlite_path_defaults_to_config(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "registry.sqlite3"
    config = Config(
        providers={"doc_status": {"kind": "sqlite", "sqlite_path": str(path)}}
    )
    store = SQLiteDocStatusStore(config)
    store.put(_record("d1"))
    assert path.exists()
    store.close()


def test_sqlite_errors_are_registry_errors(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory", encoding="utf-8")
    store = SQLiteDocStatusStore(Config(), blocker / "doc-status.db")
    with pytest.raises(DocStatusRegistryError, match="sqlite_path"):
        store.get("d1")


def test_sqlite_interrupted_write_is_rolled_back(tmp_path: Path) -> None:
    store = SQLiteDocStatusStore(Config(), tmp_path / "db.sqlite3")
    store.put(_record("kept"))
    with pytest.raises(RuntimeError), store._transaction("write") as connection:
        connection.execute("DELETE FROM documents")
        raise RuntimeError("interrupted")
    assert [r.doc_id for r in store.list_all()] == ["kept"]
    store.close()
