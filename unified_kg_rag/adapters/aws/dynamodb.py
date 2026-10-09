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

import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    DocStatusRecord,
    DocumentDelta,
)
from unified_kg_rag.shared import (
    DataProcessingError,
    DocStatusRegistryError,
    get_logger,
)

if TYPE_CHECKING:
    from types_boto3_dynamodb import DynamoDBClient

logger = get_logger(__name__)

# DynamoDB stores list attributes as lists; empty lists are allowed. The doc_id
# is the partition key.
_PARTITION_KEY = "doc_id"
_SCOPE_ATTRIBUTE = "registry_scope"
# DynamoDB rejects an item over 400 KB (attribute names plus values).
_MAX_ITEM_BYTES = 400 * 1024
# botocore's standard retry mode backs off and retries throttling
# (ThrottlingException, ProvisionedThroughputExceededException,
# RequestLimitExceeded) and transient 5xx/connection errors. Ten attempts ride
# out a throttling burst; an error that outlasts them fails the run.
_RETRY_CONFIG = BotoConfig(retries={"mode": "standard", "total_max_attempts": 10})
# BatchGetItem reads at most 100 keys per request. Keys it returns unprocessed
# (throttling, or the 16 MB response limit) are re-requested with exponential
# backoff, up to this many requests per batch.
_BATCH_GET_SIZE = 100
_BATCH_GET_MAX_ATTEMPTS = 8
_BATCH_GET_BASE_DELAY = 0.05
_BATCH_GET_MAX_DELAY = 2.0

_MISSING_TABLE_CODES = frozenset({"ResourceNotFoundException"})
_ACCESS_CODES = frozenset(
    {
        "AccessDeniedException",
        "UnrecognizedClientException",
        "ExpiredTokenException",
        "KMSAccessDeniedException",
    }
)
_THROTTLING_CODES = frozenset(
    {
        "ThrottlingException",
        "ProvisionedThroughputExceededException",
        "RequestLimitExceeded",
    }
)


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
            client = self.boto_session.client("dynamodb", config=_RETRY_CONFIG)
            if self.ddb_config.create_table_if_missing:
                with self._registry_errors("create the table"):
                    self._ensure_table(client)
            self._client = client
        return self._client

    @contextmanager
    def _registry_errors(self, operation: str) -> Iterator[None]:
        """Re-raise AWS errors as :class:`DocStatusRegistryError` with a fix hint."""
        try:
            yield
        except (ClientError, BotoCoreError) as e:
            raise DocStatusRegistryError(self._describe_error(operation, e)) from e

    def _describe_error(self, operation: str, error: Exception) -> str:
        code = (
            str(error.response.get("Error", {}).get("Code", ""))
            if isinstance(error, ClientError)
            else type(error).__name__
        )
        if code in _MISSING_TABLE_CODES:
            hint = (
                "The table does not exist in this account and region. Create it "
                "(the CDK stack does), point aws.dynamodb.table_name (env "
                "GRAPHRAG_DOC_STATUS_TABLE) at the existing table, or set "
                "aws.dynamodb.create_table_if_missing: true."
            )
        elif code in _ACCESS_CODES:
            hint = (
                "Refresh the credentials, or grant the caller dynamodb:Scan, "
                "GetItem, BatchGetItem, PutItem, DeleteItem and DescribeTable on "
                "the table "
                "(and the use of the table's KMS key, if it has one)."
            )
        elif code in _THROTTLING_CODES:
            hint = (
                "Still throttled after the client's retries. Raise the table's "
                "capacity or switch it to on-demand billing, then re-run."
            )
        else:
            hint = "Check the table and the connection to DynamoDB, then re-run."
        region = self.boto_session.region_name or "unset"
        return (
            f"Doc-status registry: could not {operation} on DynamoDB table "
            f"'{self.table_name}' (region {region}): {code}: {error}. {hint} "
            "Incremental indexing needs the registry; set aws.dynamodb.enabled: "
            "false to run without it (every run then rebuilds the stores from "
            "its own documents)."
        )

    def _ensure_table(self, client: DynamoDBClient) -> None:
        """Create the doc-status table on first use if it does not exist."""
        try:
            client.describe_table(TableName=self.table_name)
            return
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
                raise

        logger.info("Creating DynamoDB doc-status table '%s'", self.table_name)
        client.create_table(
            TableName=self.table_name,
            KeySchema=[{"AttributeName": _PARTITION_KEY, "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": _PARTITION_KEY, "AttributeType": "S"}
            ],
            BillingMode=self.ddb_config.billing_mode,  # type: ignore[arg-type]
        )
        client.get_waiter("table_exists").wait(TableName=self.table_name)

    def get(self, doc_id: str) -> DocStatusRecord | None:
        with self._registry_errors("read a record"):
            response = self.client.get_item(
                TableName=self.table_name, Key={_PARTITION_KEY: {"S": doc_id}}
            )
        item = response.get("Item")
        if not item:
            return None
        return self._deserialize(item)

    def get_many(self, doc_ids: Iterable[str]) -> dict[str, DocStatusRecord]:
        """Read the stored ``doc_ids`` with ``BatchGetItem``, 100 keys per call.

        Keys DynamoDB returns as ``UnprocessedKeys`` are re-requested with
        exponential backoff; keys still unprocessed after
        ``_BATCH_GET_MAX_ATTEMPTS`` requests raise
        :class:`DocStatusRegistryError` rather than reading as absent (an
        absent record means a new document).
        """
        unique = list(dict.fromkeys(doc_ids))
        records: dict[str, DocStatusRecord] = {}
        for start in range(0, len(unique), _BATCH_GET_SIZE):
            batch = unique[start : start + _BATCH_GET_SIZE]
            for item in self._batch_get(batch):
                record = self._deserialize(item)
                records[record.doc_id] = record
        return records

    def _batch_get(self, doc_ids: list[str]) -> list[dict[str, Any]]:
        request: dict[str, Any] = {
            self.table_name: {
                "Keys": [{_PARTITION_KEY: {"S": doc_id}} for doc_id in doc_ids]
            }
        }
        items: list[dict[str, Any]] = []
        for attempt in range(_BATCH_GET_MAX_ATTEMPTS):
            if attempt:
                time.sleep(
                    min(_BATCH_GET_MAX_DELAY, _BATCH_GET_BASE_DELAY * 2**attempt)
                )
            with self._registry_errors("read records"):
                response = self.client.batch_get_item(RequestItems=request)
            items.extend(response.get("Responses", {}).get(self.table_name, []))
            request = dict(response.get("UnprocessedKeys") or {})
            if not request.get(self.table_name, {}).get("Keys"):
                return items
        remaining = len(request[self.table_name]["Keys"])
        raise DocStatusRegistryError(
            f"Doc-status registry: DynamoDB table '{self.table_name}' left "
            f"{remaining} of {len(doc_ids)} keys unprocessed after "
            f"{_BATCH_GET_MAX_ATTEMPTS} BatchGetItem requests. Raise the table's "
            "capacity or switch it to on-demand billing, then re-run."
        )

    def put(self, record: DocStatusRecord) -> None:
        """Write ``record``; raise ``DataProcessingError`` if it is over 400 KB.

        The record holds every artifact id of the document (~36 bytes each),
        so one item fits roughly 10,000 ids; a larger document must be split
        into smaller files.
        """
        item = self._serialize(record)
        size = self._item_size(item)
        if size > _MAX_ITEM_BYTES:
            raise DataProcessingError(
                f"Doc-status record for '{record.file_path or record.doc_id}' is "
                f"{size} bytes, over the DynamoDB 400 KB item limit; split the "
                "document into smaller files"
            )
        with self._registry_errors("write a record"):
            self.client.put_item(TableName=self.table_name, Item=item)

    @staticmethod
    def _item_size(item: dict[str, Any]) -> int:
        """Item size as DynamoDB counts it: names plus UTF-8 values."""
        size = 0
        for name, cell in item.items():
            size += len(name.encode())
            if "S" in cell:
                size += len(cell["S"].encode())
            elif "SS" in cell:
                size += sum(len(value.encode()) for value in cell["SS"])
            elif "N" in cell:
                size += len(cell["N"])
            else:  # NULL
                size += 1
        return size

    def delete(self, doc_id: str) -> None:
        with self._registry_errors("delete a record"):
            self.client.delete_item(
                TableName=self.table_name, Key={_PARTITION_KEY: {"S": doc_id}}
            )

    def list_all(self) -> list[DocStatusRecord]:
        records: list[DocStatusRecord] = []
        with self._registry_errors("scan the records"):
            paginator = self.client.get_paginator("scan")
            for page in paginator.paginate(TableName=self.table_name):
                for item in page.get("Items", []):
                    records.append(self._deserialize(item))
        return records

    def _scan_fingerprints(self) -> dict[str, tuple[str, str | None, bool]]:
        """Scan only ``{doc_id: (content_hash, scope, failed)}`` for diffing.

        ``diff`` needs just the partition key, the content hash, the scope and
        whether the last run failed on the document,
        so this uses a ``ProjectionExpression`` to fetch those attributes instead
        of deserializing the full ``DocStatusRecord`` (content hash + six
        artifact-id lists) for every row. (A full ``scan`` is still required
        because deletion detection needs every stored doc_id of the scope.)
        """
        fingerprints: dict[str, tuple[str, str | None, bool]] = {}
        paginator = self.client.get_paginator("scan")
        for page in paginator.paginate(
            TableName=self.table_name,
            # Attribute-name placeholders keep the projection clear of
            # DynamoDB reserved words.
            ProjectionExpression="#pk, #hash, #scope, #status",
            ExpressionAttributeNames={
                "#pk": _PARTITION_KEY,
                "#hash": "content_hash",
                "#scope": _SCOPE_ATTRIBUTE,
                "#status": "status",
            },
        ):
            for item in page.get("Items", []):
                doc_id = item.get(_PARTITION_KEY, {}).get("S")
                content_hash = item.get("content_hash", {}).get("S", "")
                scope = item.get(_SCOPE_ATTRIBUTE, {}).get("S")
                failed = item.get("status", {}).get("S") == DocStatus.FAILED.value
                if doc_id is not None:
                    fingerprints[doc_id] = (content_hash, scope, failed)
        return fingerprints

    def diff(self, incoming: dict[str, str], scope: str | None = None) -> DocumentDelta:
        """Classify ``{doc_id: content_hash}`` against persisted state.

        Mirrors ``FakeDocStatusStore.diff`` exactly so the production and test
        implementations stay behaviourally identical: with ``scope``, only
        stored records of that scope can be classified deleted. A FAILED
        record is ``changed`` even with an unchanged hash, so it is retried.
        """
        with self._registry_errors("scan the records"):
            stored = self._scan_fingerprints()
        delta = DocumentDelta()
        for doc_id, content_hash in incoming.items():
            if doc_id not in stored:
                delta.new.append(doc_id)
            elif stored[doc_id][0] != content_hash or stored[doc_id][2]:
                delta.changed.append(doc_id)
            else:
                delta.unchanged.append(doc_id)
        incoming_ids = set(incoming)
        delta.deleted = [
            doc_id
            for doc_id, (_, stored_scope, _) in stored.items()
            if doc_id not in incoming_ids and (scope is None or stored_scope == scope)
        ]
        return delta

    @staticmethod
    def _serialize(record: DocStatusRecord) -> dict[str, Any]:
        """Convert a record into a DynamoDB item (low-level attribute format)."""
        item: dict[str, Any] = {
            _PARTITION_KEY: {"S": record.doc_id},
            "content_hash": {"S": record.content_hash},
            "status": {"S": record.status.value},
            "suffix": {"S": record.suffix},
            _SCOPE_ATTRIBUTE: (
                {"S": record.scope} if record.scope is not None else {"NULL": True}
            ),
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
            "claim_ids": (
                {"SS": record.claim_ids} if record.claim_ids else {"NULL": True}
            ),
            "community_report_ids": (
                {"SS": record.community_report_ids}
                if record.community_report_ids
                else {"NULL": True}
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
        item["failure_count"] = {"N": str(record.failure_count)}
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

        failure_count_cell = item.get("failure_count")
        return DocStatusRecord(
            doc_id=item[_PARTITION_KEY]["S"],
            content_hash=item["content_hash"]["S"],
            status=item["status"]["S"],
            suffix=_str("suffix") or "default",
            scope=_str(_SCOPE_ATTRIBUTE),
            file_path=_str("file_path"),
            content_summary=_str("content_summary"),
            content_length=content_length,
            entity_ids=_str_set("entity_ids"),
            relationship_ids=_str_set("relationship_ids"),
            text_unit_ids=_str_set("text_unit_ids"),
            community_ids=_str_set("community_ids"),
            claim_ids=_str_set("claim_ids"),
            community_report_ids=_str_set("community_report_ids"),
            error_info=_str("error_info"),
            failure_count=(
                int(failure_count_cell["N"])
                if failure_count_cell and "N" in failure_count_cell
                else 0
            ),
            created_at=_str("created_at"),
            updated_at=_str("updated_at"),
        )
