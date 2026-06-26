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
from unified_kg_rag.domain.models import Config, DocStatus, DocStatusRecord
from unified_kg_rag.ports import DocStatusPort

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
    for doc_id, content_hash in [("keep", "h1"), ("edit", "old"), ("gone", "h3")]:
        record = DocStatusRecord(doc_id=doc_id, content_hash=content_hash)
        ddb_store.put(record)
        fake.put(record)

    incoming = {"keep": "h1", "edit": "new", "fresh": "h4"}
    ddb_delta = ddb_store.diff(incoming)
    fake_delta = fake.diff(incoming)

    assert sorted(ddb_delta.new) == sorted(fake_delta.new) == ["fresh"]
    assert sorted(ddb_delta.changed) == sorted(fake_delta.changed) == ["edit"]
    assert sorted(ddb_delta.unchanged) == sorted(fake_delta.unchanged) == ["keep"]
    assert sorted(ddb_delta.deleted) == sorted(fake_delta.deleted) == ["gone"]
