# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the DynamoDB doc-status adapter (moto-mocked).

These run AWS-free via ``moto`` (no ``aws`` marker needed). The port behaviour
shared with the other registries lives in ``test_doc_status_conformance.py``;
this file keeps what is DynamoDB-specific: table creation, empty string sets,
projected per-namespace reads, batch sizes, unprocessed keys, the item
size limit and the key schema check.
"""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from unified_kg_rag.adapters.doc_status import dynamodb as dynamodb_module
from unified_kg_rag.adapters.doc_status.dynamodb import DynamoDBDocStatusStore
from unified_kg_rag.domain.models import (
    PENDING_CONTENT_HASH,
    Config,
    DocStatus,
    DocStatusRecord,
)
from unified_kg_rag.shared import (
    ConfigurationError,
    DataProcessingError,
    DocStatusRegistryError,
)

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


def test_table_auto_created(ddb_store: DynamoDBDocStatusStore) -> None:
    # describe_table should now succeed without raising.
    ddb_store.client.describe_table(TableName="test-doc-status")


def test_roundtrip_all_empty_lists_and_none_scalars(
    ddb_store: DynamoDBDocStatusStore,
) -> None:
    record = DocStatusRecord(doc_id="empty", content_hash="h")
    ddb_store.put(record)
    fetched = ddb_store.get("empty")
    assert fetched is not None
    assert fetched.status is DocStatus.PENDING
    assert fetched.text_unit_ids == []
    assert fetched.file_path is None
    assert fetched.suffix == "default"


def test_roundtrip_failed_status(
    ddb_store: DynamoDBDocStatusStore,
) -> None:
    record = DocStatusRecord(
        doc_id="f1",
        content_hash="h",
        status=DocStatus.FAILED,
        error_info="boom",
        suffix="tenant-a",
    )
    ddb_store.put(record)
    fetched = ddb_store.get("f1")
    assert fetched is not None
    assert fetched.status is DocStatus.FAILED
    assert fetched.error_info == "boom"
    assert fetched.suffix == "tenant-a"


def test_stored_states_are_projected_and_read_per_namespace(
    ddb_store: DynamoDBDocStatusStore, mocker
) -> None:
    # The projected read backing diff() returns just {doc_id: (hash, scope,
    # failed, extraction fingerprint)}, even for records listing text units,
    # and Queries the namespace's partition rather than scanning the table.
    ddb_store.put(
        DocStatusRecord(doc_id="ns#d1", content_hash="h1", text_unit_ids=["t1"])
    )
    ddb_store.put(DocStatusRecord(doc_id="ns#d2", content_hash="h2", status="failed"))
    ddb_store.put(
        DocStatusRecord(
            doc_id="ns#d3",
            content_hash="h3",
            status="processed",
            scope="ns|src",
            extraction_fingerprint="x1",
        )
    )
    ddb_store.put(DocStatusRecord(doc_id="other#d1", content_hash="h4"))
    scan = mocker.spy(ddb_store.client, "scan")

    assert ddb_store._stored_states("ns") == {
        "ns#d1": ("h1", None, False, None),
        "ns#d2": ("h2", None, True, None),
        "ns#d3": ("h3", "ns|src", False, "x1"),
    }
    assert ddb_store.diff({}, scope="ns|src").deleted == ["ns#d3"]
    scan.assert_not_called()
    assert set(ddb_store._stored_states(None)) == {
        "ns#d1",
        "ns#d2",
        "ns#d3",
        "other#d1",
    }


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
            text_unit_ids=["t3", "t1", "t2"],
        )
    )

    read = ddb_store.get("a")
    assert read.text_unit_ids == ["t1", "t2", "t3"]
    assert ddb_store.list_all() == [read]
    assert ddb_store.get_many(["a"]) == {"a": read}


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
        text_unit_ids=[f"unit-{i:08d}-{'x' * 40}" for i in range(9000)],
    )

    with pytest.raises(DataProcessingError, match="400 KB"):
        ddb_store.put_many([DocStatusRecord(doc_id="small", content_hash="h"), huge])

    assert ddb_store.list_all() == []


def test_write_ahead_record_round_trips_and_diffs_as_changed(
    ddb_store: DynamoDBDocStatusStore,
) -> None:
    pending = DocStatusRecord(
        doc_id="ns#d",
        content_hash=PENDING_CONTENT_HASH,
        status=DocStatus.PENDING,
        scope="ns|src",
        text_unit_ids=["t1"],
    )
    ddb_store.put_many([pending])

    assert ddb_store.get("ns#d") == pending
    assert ddb_store.diff({"ns#d": "any-hash"}, scope="ns|src").changed == ["ns#d"]
    assert ddb_store.diff({}, scope="ns|src").deleted == ["ns#d"]


@pytest.mark.parametrize("create_table_if_missing", [True, False])
def test_a_table_of_the_earlier_key_schema_is_rejected(
    create_table_if_missing: bool,
) -> None:
    # Keyed by doc_id alone, its records hold per-artifact lineage this
    # release does not read: fail before any read or write, naming the fix.
    with mock_aws():
        session = boto3.Session(region_name="us-east-1")
        session.client("dynamodb").create_table(
            TableName="old-doc-status",
            KeySchema=[{"AttributeName": "doc_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "doc_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        config = Config()
        config.aws.dynamodb.table_name = "old-doc-status"
        config.aws.dynamodb.create_table_if_missing = create_table_if_missing
        store = DynamoDBDocStatusStore(config, boto_session=session)

        with pytest.raises(ConfigurationError, match="indexing.reset"):
            store.get("d")
