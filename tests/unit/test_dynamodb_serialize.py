# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""DynamoDB doc-status (de)serialization round-trips every record field.

The DynamoDBDocStatusStore._serialize / _deserialize pair is static (no
AWS/network), so it is tested directly. The item is keyed by the index
namespace (partition key) and the doc id (sort key).
"""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.doc_status.dynamodb import DynamoDBDocStatusStore
from unified_kg_rag.domain.models import DocStatus, DocStatusRecord

pytestmark = pytest.mark.unit


def _full_record() -> DocStatusRecord:
    return DocStatusRecord(
        doc_id="doc-1",
        content_hash="hash-1",
        status=DocStatus.PROCESSED,
        suffix="default",
        file_path="/x/y.txt",
        text_unit_ids=["t1", "t2", "t3"],
        error_info=None,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-02T00:00:00Z",
    )


def test_record_round_trips_with_its_namespace_key() -> None:
    original = _full_record().model_copy(update={"doc_id": "tenant-a#doc-1"})
    item = DynamoDBDocStatusStore._serialize(original)

    assert item["namespace"] == {"S": "tenant-a"}
    assert item["doc_id"] == {"S": "tenant-a#doc-1"}
    assert item["text_unit_ids"] == {"SS": ["t1", "t2", "t3"]}
    assert DynamoDBDocStatusStore._deserialize(item) == original


def test_empty_text_unit_list_round_trips_as_empty() -> None:
    record = DocStatusRecord(doc_id="d", content_hash="h")
    item = DynamoDBDocStatusStore._serialize(record)
    assert item["namespace"] == {"S": "default"}
    assert DynamoDBDocStatusStore._deserialize(item).text_unit_ids == []


def _store(mocker) -> DynamoDBDocStatusStore:
    from unified_kg_rag.domain.models import Config

    store = DynamoDBDocStatusStore(Config(), boto_session=mocker.MagicMock())
    store._client = mocker.MagicMock()
    return store


def test_put_rejects_a_record_over_the_item_size_limit(mocker) -> None:
    from unified_kg_rag.shared import DataProcessingError

    store = _store(mocker)
    # ~12k uuid-length ids is over DynamoDB's 400 KB item limit.
    ids = [f"{i:036d}" for i in range(12_000)]
    record = DocStatusRecord(
        doc_id="big", content_hash="h", file_path="big.pdf", text_unit_ids=ids
    )
    with pytest.raises(DataProcessingError, match="400 KB"):
        store.put(record)
    store._client.put_item.assert_not_called()


def test_put_writes_a_record_within_the_limit(mocker) -> None:
    store = _store(mocker)
    store.put(_full_record())
    store._client.put_item.assert_called_once()


def test_failure_count_round_trips_and_defaults_to_zero() -> None:
    record = _full_record().model_copy(
        update={"status": DocStatus.FAILED, "failure_count": 2}
    )
    item = DynamoDBDocStatusStore._serialize(record)
    assert DynamoDBDocStatusStore._deserialize(item).failure_count == 2

    # A record written before the attribute existed reads as never failed.
    del item["failure_count"]
    assert DynamoDBDocStatusStore._deserialize(item).failure_count == 0


def test_record_with_removed_content_attributes_still_loads() -> None:
    # Records written while DocStatusRecord had content_summary/content_length.
    item = DynamoDBDocStatusStore._serialize(_full_record())
    item["content_summary"] = {"S": "summary"}
    item["content_length"] = {"N": "42"}
    assert DynamoDBDocStatusStore._deserialize(item) == _full_record()
    stored_json = _full_record().model_dump() | {"content_length": 42}
    assert DocStatusRecord.model_validate(stored_json) == _full_record()
