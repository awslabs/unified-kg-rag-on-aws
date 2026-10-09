# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write-ahead lineage overflow in the DynamoDB registry (moto-mocked).

A large document whose stored and re-extracted records each fit one 400 KB
item has a write-ahead union that does not. The PENDING record then keeps the
stored lineage and the other planned ids go to overflow items in the same
table, which no record read, diff or scan returns.
"""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from tests.fixtures.incremental_runs import (
    Corpus,
    IncrementalRun,
    document,
    expected_state,
)
from unified_kg_rag.adapters.aws import DynamoDBDocStatusStore
from unified_kg_rag.domain.ingestion.delta_detector import document_doc_id
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    DocStatusRecord,
    DocumentLineage,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def ddb_store() -> DynamoDBDocStatusStore:
    with mock_aws():
        config = Config()
        config.aws.dynamodb.enabled = True
        config.aws.dynamodb.table_name = "test-doc-status"
        session = boto3.Session(region_name="us-east-1")
        store = DynamoDBDocStatusStore(config, boto_session=session)
        _ = store.client
        yield store


def _raw_items(store: DynamoDBDocStatusStore) -> list[dict]:
    paginator = store.client.get_paginator("scan")
    return [
        item
        for page in paginator.paginate(TableName=store.table_name)
        for item in page.get("Items", [])
    ]


def _overflow_keys(store: DynamoDBDocStatusStore) -> list[str]:
    return sorted(
        item["doc_id"]["S"]
        for item in _raw_items(store)
        if item.get("record_kind", {}).get("S") == "lineage_overflow"
    )


def _ids(prefix: str, n: int) -> list[str]:
    return [f"{prefix}-{i:06d}-{'x' * 28}" for i in range(n)]


def test_overflow_round_trips_in_parts_and_appends(
    ddb_store: DynamoDBDocStatusStore,
) -> None:
    # ~440 KB of ids: two parts.
    first = DocumentLineage(doc_id="d", entity_ids=_ids("e", 12000), claim_ids=["c1"])
    ddb_store.add_lineage_overflow([first])
    assert _overflow_keys(ddb_store) == ["d#pending#0", "d#pending#1"]

    # Appending adds only the ids not yet stored, as a new part.
    ddb_store.add_lineage_overflow(
        [DocumentLineage(doc_id="d", entity_ids=["e-new", _ids("e", 1)[0]])]
    )
    assert _overflow_keys(ddb_store) == ["d#pending#0", "d#pending#1", "d#pending#2"]

    overflow = ddb_store.get_lineage_overflow(["d", "other"])
    assert list(overflow) == ["d"]
    assert overflow["d"].entity_ids == sorted([*_ids("e", 12000), "e-new"])
    assert overflow["d"].claim_ids == ["c1"]

    ddb_store.delete_lineage_overflow(["d", "other"])
    assert _overflow_keys(ddb_store) == []
    assert ddb_store.get_lineage_overflow(["d"]) == {}


def test_overflow_items_are_no_records(ddb_store: DynamoDBDocStatusStore) -> None:
    ddb_store.put(
        DocStatusRecord(
            doc_id="d", content_hash="pending", status=DocStatus.PENDING, scope="s|a"
        )
    )
    ddb_store.add_lineage_overflow([DocumentLineage(doc_id="d", entity_ids=["e"])])
    key = "d#pending#0"

    assert [r.doc_id for r in ddb_store.list_all()] == ["d"]
    assert ddb_store.get(key) is None
    assert ddb_store.get_many([key, "d"]).keys() == {"d"}
    # Never new/changed/unchanged/deleted, in any scope or none, and no scope.
    for scope in (None, "s|a"):
        delta = ddb_store.diff({}, scope=scope)
        assert delta.deleted == ["d"]
        assert delta.stored_scopes == ["s|a"]
    assert ddb_store.diff({key: "h"}).new == [key]


def test_record_fits_tracks_the_item_limit(ddb_store: DynamoDBDocStatusStore) -> None:
    small = DocStatusRecord(doc_id="d", content_hash="h", entity_ids=_ids("e", 10))
    huge = small.model_copy(update={"entity_ids": _ids("e", 12000)})
    assert ddb_store.record_fits(small)
    assert not ddb_store.record_fits(huge)


# A document whose old and new versions each produce ~235 KB of artifact ids
# (a distinct entity pair per text unit): the write-ahead union is ~470 KB.
_UNITS = 1850
R0: Corpus = {
    "/c/big.txt": [[(f"Org{i}a", f"Org{i}b", 1)] for i in range(_UNITS)],
    "/c/small.txt": [[("Vendor", "Depot", 1)]],
    "/c/kept.txt": [[("Vendor", "Buyer", 1)]],
}
R1: Corpus = {
    "/c/big.txt": [[(f"Org{i}c", f"Org{i}d", 1)] for i in range(_UNITS)],
    "/c/small.txt": [[("Vendor", "Carrier", 1)]],
    "/c/kept.txt": R0["/c/kept.txt"],
    "/c/new.txt": [[("Carrier", "Bank", 1)]],
}


def _big_id() -> str:
    return document_doc_id(document("/c/big.txt", R0["/c/big.txt"]))


@pytest.mark.timeout(600)
def test_large_document_write_ahead_spills_and_recovers(
    ddb_store: DynamoDBDocStatusStore,
) -> None:
    harness = IncrementalRun(ddb_store)
    harness.run(R0)
    big = ddb_store.get(_big_id())
    assert big is not None and big.status is DocStatus.PROCESSED
    # The premise: the stored record (and so the committed one) fits, at more
    # than half the item limit; old and new share no id.
    size = ddb_store._item_size(ddb_store._serialize(big))
    assert 200 * 1024 < size < 400 * 1024

    # The write-ahead succeeds, and every document of the delta is PENDING.
    assert harness.run(R1, interrupt="after_write_ahead")
    pending = {
        r.file_path for r in ddb_store.list_all() if r.status is DocStatus.PENDING
    }
    assert pending == {"/c/big.txt", "/c/small.txt", "/c/new.txt"}
    assert _overflow_keys(ddb_store)
    assert all(k.startswith(f"{_big_id()}#pending#") for k in _overflow_keys(ddb_store))

    # Interrupted after writing the new version: everything it wrote is
    # listed (record + overflow), so removing big then leaves nothing behind.
    assert harness.run(R1, interrupt="before_record")
    without_big = {p: c for p, c in R1.items() if p != "/c/big.txt"}
    harness.run(without_big)
    assert harness.state() == expected_state(without_big)
    assert harness.registry_problems(without_big) == []
    assert _overflow_keys(ddb_store) == []

    # An uninterrupted change of big commits and deletes its overflow.
    harness.run(R0)
    harness.run(R1)
    assert harness.state() == expected_state(R1)
    assert harness.registry_problems(R1) == []
    assert _overflow_keys(ddb_store) == []
