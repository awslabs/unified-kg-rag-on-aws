# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the DynamoDB doc-status adapter (moto-mocked).

These run AWS-free via ``moto`` (no ``aws`` marker needed). They assert the
adapter's behaviour is identical to the in-memory ``FakeDocStatusStore``, which
is the reference the production adapter must match.
"""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from unified_kg_rag.adapters.aws import DynamoDBDocStatusStore
from unified_kg_rag.adapters.aws import dynamodb as dynamodb_module
from unified_kg_rag.domain.models import (
    PENDING_CONTENT_HASH,
    Config,
    DocStatus,
    DocStatusRecord,
)
from unified_kg_rag.ports import DocStatusPort
from unified_kg_rag.shared import DataProcessingError, DocStatusRegistryError

pytestmark = pytest.mark.integration


@pytest.fixture
def ddb_store() -> DynamoDBDocStatusStore:
    with mock_aws():
        config = Config()
        config.aws.dynamodb.enabled = True
        config.aws.dynamodb.table_name = "test-doc-status"
        # moto needs a region/session; profile_name None is fine under mock.
        session = boto3.Session(region_name="us-east-1")
        store = DynamoDBDocStatusStore(config, boto_session=session)
        # Touch the client to trigger lazy table creation under the mock.
        _ = store.client
        yield store


def test_conforms_to_port(ddb_store: DynamoDBDocStatusStore) -> None:
    assert isinstance(ddb_store, DocStatusPort)


def test_table_auto_created(ddb_store: DynamoDBDocStatusStore) -> None:
    # describe_table should now succeed without raising.
    ddb_store.client.describe_table(TableName="test-doc-status")


def test_put_get_roundtrip(ddb_store: DynamoDBDocStatusStore) -> None:
    record = DocStatusRecord(
        doc_id="d1",
        content_hash="h1",
        status=DocStatus.PROCESSED,
        file_path="/tmp/d1.txt",
        content_length=42,
        entity_ids=["e1", "e2"],
        relationship_ids=["r1"],
        text_unit_ids=["t1"],
        community_ids=[],
        updated_at="2026-06-19T00:00:00",
    )
    ddb_store.put(record)
    fetched = ddb_store.get("d1")
    assert fetched is not None
    assert fetched.doc_id == "d1"
    assert fetched.status is DocStatus.PROCESSED
    assert fetched.content_length == 42
    assert set(fetched.entity_ids) == {"e1", "e2"}
    assert fetched.relationship_ids == ["r1"]
    assert fetched.community_ids == []
    assert fetched.file_path == "/tmp/d1.txt"


def test_get_missing_returns_none(ddb_store: DynamoDBDocStatusStore) -> None:
    assert ddb_store.get("nope") is None


def test_delete(ddb_store: DynamoDBDocStatusStore) -> None:
    ddb_store.put(DocStatusRecord(doc_id="d1", content_hash="h1"))
    ddb_store.delete("d1")
    assert ddb_store.get("d1") is None


def test_list_all(ddb_store: DynamoDBDocStatusStore) -> None:
    ddb_store.put(DocStatusRecord(doc_id="d1", content_hash="h1"))
    ddb_store.put(DocStatusRecord(doc_id="d2", content_hash="h2"))
    ids = {r.doc_id for r in ddb_store.list_all()}
    assert ids == {"d1", "d2"}


def test_roundtrip_all_empty_lists_and_none_scalars(
    ddb_store: DynamoDBDocStatusStore,
) -> None:
    record = DocStatusRecord(doc_id="empty", content_hash="h")
    ddb_store.put(record)
    fetched = ddb_store.get("empty")
    assert fetched is not None
    assert fetched.status is DocStatus.PENDING
    assert fetched.entity_ids == []
    assert fetched.relationship_ids == []
    assert fetched.content_length is None
    assert fetched.file_path is None
    assert fetched.suffix == "default"


def test_roundtrip_failed_status_and_zero_length(
    ddb_store: DynamoDBDocStatusStore,
) -> None:
    record = DocStatusRecord(
        doc_id="f1",
        content_hash="h",
        status=DocStatus.FAILED,
        content_length=0,
        error_info="boom",
        suffix="tenant-a",
    )
    ddb_store.put(record)
    fetched = ddb_store.get("f1")
    assert fetched is not None
    assert fetched.status is DocStatus.FAILED
    assert fetched.content_length == 0
    assert fetched.error_info == "boom"
    assert fetched.suffix == "tenant-a"


def test_diff_matches_fake(ddb_store: DynamoDBDocStatusStore) -> None:
    fake = FakeDocStatusStore()
    for doc_id, content_hash, status in [
        ("keep", "h1", DocStatus.PROCESSED),
        ("edit", "old", DocStatus.PROCESSED),
        ("gone", "h3", DocStatus.PROCESSED),
        # Same content, but its last extraction failed: it must be retried.
        ("retry", "h5", DocStatus.FAILED),
    ]:
        record = DocStatusRecord(
            doc_id=doc_id, content_hash=content_hash, status=status
        )
        ddb_store.put(record)
        fake.put(record)

    incoming = {"keep": "h1", "edit": "new", "fresh": "h4", "retry": "h5"}
    ddb_delta = ddb_store.diff(incoming)
    fake_delta = fake.diff(incoming)

    assert sorted(ddb_delta.new) == sorted(fake_delta.new) == ["fresh"]
    assert sorted(ddb_delta.changed) == sorted(fake_delta.changed) == ["edit", "retry"]
    assert sorted(ddb_delta.unchanged) == sorted(fake_delta.unchanged) == ["keep"]
    assert sorted(ddb_delta.deleted) == sorted(fake_delta.deleted) == ["gone"]


def test_scan_fingerprints_returns_doc_id_and_hash_only(
    ddb_store: DynamoDBDocStatusStore,
) -> None:
    # The projection-scan helper backing diff() returns just {doc_id: (hash,
    # scope, failed, pending)}, even for records carrying full artifact-id
    # lineage.
    ddb_store.put(
        DocStatusRecord(
            doc_id="d1",
            content_hash="h1",
            entity_ids=["e1", "e2"],
            relationship_ids=["r1"],
            text_unit_ids=["t1"],
        )
    )
    ddb_store.put(DocStatusRecord(doc_id="d2", content_hash="h2", status="processed"))
    ddb_store.put(DocStatusRecord(doc_id="d3", content_hash="h3", status="failed"))
    assert ddb_store._scan_fingerprints() == {
        "d1": ("h1", None, False, True),
        "d2": ("h2", None, False, False),
        "d3": ("h3", None, True, False),
    }


def test_diff_after_projection_optimization_on_empty_table(
    ddb_store: DynamoDBDocStatusStore,
) -> None:
    # Empty registry -> everything incoming is new, nothing deleted.
    delta = ddb_store.diff({"a": "h1", "b": "h2"})
    assert sorted(delta.new) == ["a", "b"]
    assert delta.changed == [] and delta.unchanged == [] and delta.deleted == []


def test_get_many_reads_more_than_one_batch_and_skips_unknown_ids(
    ddb_store: DynamoDBDocStatusStore,
) -> None:
    ids = [f"d{i}" for i in range(230)]
    for doc_id in ids[:150]:
        ddb_store.put(DocStatusRecord(doc_id=doc_id, content_hash=f"h-{doc_id}"))
    calls: list[int] = []
    batch_get = ddb_store.client.batch_get_item

    def counting(**kwargs):
        calls.append(len(kwargs["RequestItems"]["test-doc-status"]["Keys"]))
        return batch_get(**kwargs)

    ddb_store.client.batch_get_item = counting  # type: ignore[method-assign]

    # Duplicates are requested once (BatchGetItem rejects repeated keys).
    records = ddb_store.get_many([*ids, "d0", "d1"])

    assert calls == [100, 100, 30]
    assert sorted(records) == sorted(ids[:150])
    assert records["d7"] == ddb_store.get("d7")
    assert ddb_store.get_many([]) == {}
    assert calls == [100, 100, 30]


def test_get_many_retries_unprocessed_keys(
    ddb_store: DynamoDBDocStatusStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    for doc_id in ("a", "b", "c"):
        ddb_store.put(DocStatusRecord(doc_id=doc_id, content_hash="h"))
    monkeypatch.setattr(dynamodb_module.time, "sleep", lambda _: None)
    batch_get = ddb_store.client.batch_get_item
    requests: list[list[str]] = []

    def throttled(**kwargs):
        keys = kwargs["RequestItems"]["test-doc-status"]["Keys"]
        requests.append([key["doc_id"]["S"] for key in keys])
        if len(requests) > 1:
            return batch_get(**kwargs)
        # First request: only the first key is served, the rest come back
        # unprocessed, as under throttling.
        served = batch_get(RequestItems={"test-doc-status": {"Keys": keys[:1]}})
        served["UnprocessedKeys"] = {"test-doc-status": {"Keys": keys[1:]}}
        return served

    ddb_store.client.batch_get_item = throttled  # type: ignore[method-assign]

    assert sorted(ddb_store.get_many(["a", "b", "c"])) == ["a", "b", "c"]
    assert requests == [["a", "b", "c"], ["b", "c"]]


def test_get_many_fails_when_keys_stay_unprocessed(
    ddb_store: DynamoDBDocStatusStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    ddb_store.put(DocStatusRecord(doc_id="a", content_hash="h"))
    monkeypatch.setattr(dynamodb_module.time, "sleep", lambda _: None)

    def never(**kwargs):
        return {"Responses": {}, "UnprocessedKeys": kwargs["RequestItems"]}

    ddb_store.client.batch_get_item = never  # type: ignore[method-assign]

    # A key that could not be read must not read as absent (a new document).
    with pytest.raises(DocStatusRegistryError, match="unprocessed"):
        ddb_store.get_many(["a"])


def test_string_sets_read_back_sorted(ddb_store: DynamoDBDocStatusStore) -> None:
    # String sets have no order: read back sorted, a record compares equal on
    # every read whatever order DynamoDB returns the members in.
    ddb_store.put(
        DocStatusRecord(
            doc_id="a",
            content_hash="h",
            entity_ids=["e3", "e1", "e2"],
            text_unit_ids=["t2", "t1"],
        )
    )

    read = ddb_store.get("a")
    assert read.entity_ids == ["e1", "e2", "e3"]
    assert read.text_unit_ids == ["t1", "t2"]
    assert ddb_store.list_all() == [read]
    assert ddb_store.get_many(["a"]) == {"a": read}


def test_diff_reports_stored_scopes_like_the_fake(
    ddb_store: DynamoDBDocStatusStore,
) -> None:
    fake = FakeDocStatusStore()
    for store in (ddb_store, fake):
        for doc_id, scope in (("a", "s1"), ("b", "s2"), ("c", None), ("d", "s1")):
            store.put(DocStatusRecord(doc_id=doc_id, content_hash="h", scope=scope))

    assert ddb_store.diff({"a": "h"}, scope="s1") == fake.diff({"a": "h"}, scope="s1")
    assert ddb_store.diff({}, scope="s2").stored_scopes == ["s1", "s2"]


def test_put_many_writes_in_batches_of_25_and_last_record_wins(
    ddb_store: DynamoDBDocStatusStore,
) -> None:
    records = [DocStatusRecord(doc_id=f"d{i}", content_hash="h") for i in range(60)]
    records.append(DocStatusRecord(doc_id="d0", content_hash="h-last"))
    calls: list[int] = []
    batch_write = ddb_store.client.batch_write_item

    def counting(**kwargs):
        calls.append(len(kwargs["RequestItems"]["test-doc-status"]))
        return batch_write(**kwargs)

    ddb_store.client.batch_write_item = counting  # type: ignore[method-assign]

    # A repeated key is written once (BatchWriteItem rejects repeated keys).
    ddb_store.put_many(records)
    ddb_store.put_many([])

    assert calls == [25, 25, 10]
    assert len(ddb_store.list_all()) == 60
    stored = ddb_store.get("d0")
    assert stored is not None and stored.content_hash == "h-last"


def test_put_many_retries_unprocessed_items(
    ddb_store: DynamoDBDocStatusStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dynamodb_module.time, "sleep", lambda _: None)
    batch_write = ddb_store.client.batch_write_item
    requests: list[list[str]] = []

    def throttled(**kwargs):
        items = kwargs["RequestItems"]["test-doc-status"]
        requests.append([i["PutRequest"]["Item"]["doc_id"]["S"] for i in items])
        if len(requests) > 1:
            return batch_write(**kwargs)
        # First request: only the first item is written, the rest come back
        # unprocessed, as under throttling.
        served = batch_write(RequestItems={"test-doc-status": items[:1]})
        served["UnprocessedItems"] = {"test-doc-status": items[1:]}
        return served

    ddb_store.client.batch_write_item = throttled  # type: ignore[method-assign]

    ddb_store.put_many(
        [DocStatusRecord(doc_id=doc_id, content_hash="h") for doc_id in "abc"]
    )

    assert requests == [["a", "b", "c"], ["b", "c"]]
    assert sorted(r.doc_id for r in ddb_store.list_all()) == ["a", "b", "c"]


def test_put_many_fails_when_items_stay_unprocessed(
    ddb_store: DynamoDBDocStatusStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dynamodb_module.time, "sleep", lambda _: None)

    def never(**kwargs):
        return {"UnprocessedItems": kwargs["RequestItems"]}

    ddb_store.client.batch_write_item = never  # type: ignore[method-assign]

    with pytest.raises(DocStatusRegistryError, match="unprocessed"):
        ddb_store.put_many([DocStatusRecord(doc_id="a", content_hash="h")])


def test_put_many_rejects_an_oversized_record_before_writing_any(
    ddb_store: DynamoDBDocStatusStore,
) -> None:
    huge = DocStatusRecord(
        doc_id="huge",
        content_hash="h",
        entity_ids=[f"entity-{i:08d}-{'x' * 40}" for i in range(9000)],
    )

    with pytest.raises(DataProcessingError, match="400 KB"):
        ddb_store.put_many([DocStatusRecord(doc_id="small", content_hash="h"), huge])

    assert ddb_store.list_all() == []


def test_write_ahead_record_round_trips_and_diffs_as_changed(
    ddb_store: DynamoDBDocStatusStore,
) -> None:
    pending = DocStatusRecord(
        doc_id="d",
        content_hash=PENDING_CONTENT_HASH,
        status=DocStatus.PENDING,
        scope="ns|src",
        entity_ids=["e1"],
    )
    ddb_store.put_many([pending])

    assert ddb_store.get("d") == pending
    assert ddb_store.diff({"d": "any-hash"}, scope="ns|src").changed == ["d"]
    assert ddb_store.diff({}, scope="ns|src").deleted == ["d"]
