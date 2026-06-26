# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""DynamoDB adapter implementing the document-status registry (DocStatusPort).

Persists, across indexing runs, each document's content hash, processing status,
and the ids of the graph artifacts it produced. An incremental run diffs the
incoming corpus against this registry to compute a :class:`DocumentDelta`
(new / changed / unchanged / deleted) and merges instead of re-indexing wholesale.

The reference diff behaviour matches the in-memory ``FakeDocStatusStore`` used in
tests; both conform structurally to ``unified_kg_rag.ports.DocStatusPort``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import boto3
from botocore.exceptions import ClientError

from unified_kg_rag.domain.models import Config, DocStatusRecord, DocumentDelta
from unified_kg_rag.shared import get_logger

if TYPE_CHECKING:
    from types_boto3_dynamodb import DynamoDBClient

logger = get_logger(__name__)

# DynamoDB stores list attributes as lists; empty lists are allowed. The doc_id
# is the partition key.
_PARTITION_KEY = "doc_id"


class DynamoDBDocStatusStore:
    """DynamoDB-backed implementation of :class:`DocStatusPort`."""

    def __init__(
        self, config: Config, boto_session: boto3.Session | None = None
    ) -> None:
        self.config = config
        self.ddb_config = config.aws.dynamodb
        self.table_name = self.ddb_config.table_name
        self.boto_session = boto_session or boto3.Session(
            profile_name=config.aws.profile_name,
            region_name=config.aws.region_name,
        )
        self._client: DynamoDBClient | None = None

    @property
    def client(self) -> DynamoDBClient:
        if self._client is None:
            self._client = self.boto_session.client("dynamodb")
            if self.ddb_config.create_table_if_missing:
                self._ensure_table()
        return self._client

    def _ensure_table(self) -> None:
        """Create the doc-status table on first use if it does not exist."""
        assert self._client is not None
        try:
            self._client.describe_table(TableName=self.table_name)
            return
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
                raise

        logger.info("Creating DynamoDB doc-status table '%s'", self.table_name)
        self._client.create_table(
            TableName=self.table_name,
            KeySchema=[{"AttributeName": _PARTITION_KEY, "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": _PARTITION_KEY, "AttributeType": "S"}
            ],
            BillingMode=self.ddb_config.billing_mode,  # type: ignore[arg-type]
        )
        self._client.get_waiter("table_exists").wait(TableName=self.table_name)

    def get(self, doc_id: str) -> DocStatusRecord | None:
        response = self.client.get_item(
            TableName=self.table_name, Key={_PARTITION_KEY: {"S": doc_id}}
        )
        item = response.get("Item")
        if not item:
            return None
        return self._deserialize(item)

    def put(self, record: DocStatusRecord) -> None:
        self.client.put_item(TableName=self.table_name, Item=self._serialize(record))

    def delete(self, doc_id: str) -> None:
        self.client.delete_item(
            TableName=self.table_name, Key={_PARTITION_KEY: {"S": doc_id}}
        )

    def list_all(self) -> list[DocStatusRecord]:
        records: list[DocStatusRecord] = []
        paginator = self.client.get_paginator("scan")
        for page in paginator.paginate(TableName=self.table_name):
            for item in page.get("Items", []):
                records.append(self._deserialize(item))
        return records

    def diff(self, incoming: dict[str, str]) -> DocumentDelta:
        """Classify ``{doc_id: content_hash}`` against persisted state.

        Mirrors ``FakeDocStatusStore.diff`` exactly so the production and test
        implementations stay behaviourally identical.
        """
        stored = {r.doc_id: r.content_hash for r in self.list_all()}
        delta = DocumentDelta()
        for doc_id, content_hash in incoming.items():
            if doc_id not in stored:
                delta.new.append(doc_id)
            elif stored[doc_id] != content_hash:
                delta.changed.append(doc_id)
            else:
                delta.unchanged.append(doc_id)
        incoming_ids = set(incoming)
        delta.deleted = [doc_id for doc_id in stored if doc_id not in incoming_ids]
        return delta

    @staticmethod
    def _serialize(record: DocStatusRecord) -> dict[str, Any]:
        """Convert a record into a DynamoDB item (low-level attribute format)."""
        item: dict[str, Any] = {
            _PARTITION_KEY: {"S": record.doc_id},
            "content_hash": {"S": record.content_hash},
            "status": {"S": record.status.value},
            "suffix": {"S": record.suffix},
            "entity_ids": (
                {"SS": record.entity_ids} if record.entity_ids else {"NULL": True}
            ),
            "relationship_ids": (
                {"SS": record.relationship_ids}
                if record.relationship_ids
                else {"NULL": True}
            ),
            "text_unit_ids": (
                {"SS": record.text_unit_ids} if record.text_unit_ids else {"NULL": True}
            ),
            "community_ids": (
                {"SS": record.community_ids} if record.community_ids else {"NULL": True}
            ),
        }
        # Optional scalar string/int attributes.
        for attr in (
            "file_path",
            "content_summary",
            "error_info",
            "created_at",
            "updated_at",
        ):
            value = getattr(record, attr)
            item[attr] = {"S": value} if value is not None else {"NULL": True}
        if record.content_length is not None:
            item["content_length"] = {"N": str(record.content_length)}
        return item

    @staticmethod
    def _deserialize(item: dict[str, Any]) -> DocStatusRecord:
        def _str(attr: str) -> str | None:
            cell = item.get(attr)
            return cell["S"] if cell and "S" in cell else None

        def _str_set(attr: str) -> list[str]:
            cell = item.get(attr)
            return list(cell["SS"]) if cell and "SS" in cell else []

        content_length_cell = item.get("content_length")
        content_length = (
            int(content_length_cell["N"])
            if content_length_cell and "N" in content_length_cell
            else None
        )

        return DocStatusRecord(
            doc_id=item[_PARTITION_KEY]["S"],
            content_hash=item["content_hash"]["S"],
            status=item["status"]["S"],
            suffix=_str("suffix") or "default",
            file_path=_str("file_path"),
            content_summary=_str("content_summary"),
            content_length=content_length,
            entity_ids=_str_set("entity_ids"),
            relationship_ids=_str_set("relationship_ids"),
            text_unit_ids=_str_set("text_unit_ids"),
            community_ids=_str_set("community_ids"),
            error_info=_str("error_info"),
            created_at=_str("created_at"),
            updated_at=_str("updated_at"),
        )
