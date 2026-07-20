# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""DynamoDB doc-status (de)serialization must round-trip ALL artifact-id lists.

Regression: _serialize/_deserialize persisted only entity/relationship/text_unit/
community ids and silently dropped claim_ids and community_report_ids, so on an
incremental delete/change those artifacts were never pruned and lingered as
orphaned documents in OpenSearch. The DynamoDBDocStatusStore._serialize /
_deserialize pair is static (no AWS/network), so it is tested directly.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.aws.dynamodb import DynamoDBDocStatusStore
from unified_kg_rag.domain.models import DocStatus, DocStatusRecord

pytestmark = pytest.mark.unit


def _full_record() -> DocStatusRecord:
    return DocStatusRecord(
        doc_id="doc-1",
        content_hash="hash-1",
        status=DocStatus.PROCESSED,
        suffix="default",
        file_path="/x/y.txt",
        content_summary="summary",
        content_length=42,
        entity_ids=["e1", "e2"],
        relationship_ids=["r1"],
        text_unit_ids=["t1", "t2", "t3"],
        community_ids=["c1"],
        claim_ids=["cl1", "cl2"],
        community_report_ids=["cr1"],
        error_info=None,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-02T00:00:00Z",
    )


def test_all_six_artifact_id_lists_round_trip() -> None:
    original = _full_record()
    item = DynamoDBDocStatusStore._serialize(original)
    restored = DynamoDBDocStatusStore._deserialize(item)

    # The two previously-dropped lists must survive the round trip.
    assert restored.claim_ids == ["cl1", "cl2"]
    assert restored.community_report_ids == ["cr1"]
    # And the ones that already worked stay intact.
    assert restored.entity_ids == ["e1", "e2"]
    assert restored.relationship_ids == ["r1"]
    assert restored.text_unit_ids == ["t1", "t2", "t3"]
    assert restored.community_ids == ["c1"]


def test_serialized_item_contains_claim_and_report_attributes() -> None:
    item = DynamoDBDocStatusStore._serialize(_full_record())
    assert item["claim_ids"] == {"SS": ["cl1", "cl2"]}
    assert item["community_report_ids"] == {"SS": ["cr1"]}


def test_empty_id_lists_round_trip_as_empty() -> None:
    record = DocStatusRecord(doc_id="d", content_hash="h")
    item = DynamoDBDocStatusStore._serialize(record)
    restored = DynamoDBDocStatusStore._deserialize(item)
    assert restored.claim_ids == []
    assert restored.community_report_ids == []
