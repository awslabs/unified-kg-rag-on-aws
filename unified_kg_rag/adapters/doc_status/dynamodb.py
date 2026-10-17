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

from unified_kg_rag.domain.ingestion.delta_detector import scope_namespace
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    DocStatusRecord,
    DocumentDelta,
    ExtractionSettings,
    StoredDocument,
)
from unified_kg_rag.shared import (
    ConfigurationError,
    DataProcessingError,
    DocStatusRegistryError,
    get_logger,
)
from unified_kg_rag.shared.utils.document_identity import doc_id_namespace

if TYPE_CHECKING:
    from types_boto3_dynamodb import DynamoDBClient

logger = get_logger(__name__)

# The index namespace (the doc_id's prefix, doc_id_namespace) is the partition
# key and the doc_id the sort key, so a run reads its own namespace with one
# Query instead of scanning every tenant's records.
_PARTITION_KEY = "namespace"
_SORT_KEY = "doc_id"
_SCOPE_ATTRIBUTE = "registry_scope"
# The namespace's write lease is an item of its partition under this sort key
# (no doc_id is "#lease": registry keys are "<namespace>#<hash>"). Its expiry
# is the table's TTL attribute, so DynamoDB drops a lease its run abandoned.
_LEASE_KEY = "#lease"
# A scope's extraction settings are an item of its namespace's partition
# under "#settings|<scope>". Items whose sort key starts with "#" are never
# listed as documents.
_SETTINGS_KEY_PREFIX = "#settings|"
LEASE_EXPIRY_ATTRIBUTE = "lease_expires_at"
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
# BatchWriteItem writes at most 25 items per request; unprocessed items are
# re-sent with the same backoff and attempt limit as BatchGetItem.
_BATCH_WRITE_SIZE = 25

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
            with self._registry_errors("describe the table"):
                self._check_key_schema(client)
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
                "UKR__AWS__DYNAMODB__TABLE_NAME) at the existing table, or set "
                "aws.dynamodb.create_table_if_missing: true."
            )
        elif code in _ACCESS_CODES:
            hint = (
                "Refresh the credentials, or grant the caller dynamodb:Query, "
                "Scan, GetItem, BatchGetItem, PutItem, BatchWriteItem, DeleteItem and "
                "DescribeTable on "
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
            "Incremental indexing needs the registry; set "
            "providers.doc_status.enabled: false to run without it (every run then rebuilds the stores from "
            "its own documents)."
        )

    def _check_key_schema(self, client: DynamoDBClient) -> None:
        """Raise :class:`ConfigurationError` for a table of the earlier
        registry format (keyed by ``doc_id`` alone): its records hold
        per-artifact lineage this release does not read. A missing table is
        left to the first read, which names the fix."""
        try:
            table = client.describe_table(TableName=self.table_name)["Table"]
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in _MISSING_TABLE_CODES:
                return
            raise
        keys = {k["AttributeName"]: k["KeyType"] for k in table.get("KeySchema", [])}
        if keys != {_PARTITION_KEY: "HASH", _SORT_KEY: "RANGE"}:
            raise ConfigurationError(
                f"Doc-status registry: DynamoDB table '{self.table_name}' has "
                f"the key schema {keys} of an earlier release; this release "
                f"keys records by '{_PARTITION_KEY}' (partition) and "
                f"'{_SORT_KEY}' (sort). Point aws.dynamodb.table_name (env "
                "UKR__AWS__DYNAMODB__TABLE_NAME) at a new table, or deploy the CDK "
                "stack's new table, and re-run with indexing.reset: true."
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
            KeySchema=[
                {"AttributeName": _PARTITION_KEY, "KeyType": "HASH"},
                {"AttributeName": _SORT_KEY, "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": _PARTITION_KEY, "AttributeType": "S"},
                {"AttributeName": _SORT_KEY, "AttributeType": "S"},
            ],
            BillingMode=self.ddb_config.billing_mode,  # type: ignore[arg-type]
        )
        client.get_waiter("table_exists").wait(TableName=self.table_name)
        # An abandoned namespace lease expires by itself (a live one is
        # renewed long before this matters; see try_acquire_lease).
        client.update_time_to_live(
            TableName=self.table_name,
            TimeToLiveSpecification={
                "Enabled": True,
                "AttributeName": LEASE_EXPIRY_ATTRIBUTE,
            },
        )

    def get(self, doc_id: str) -> DocStatusRecord | None:
        with self._registry_errors("read a record"):
            response = self.client.get_item(TableName=self.table_name, Key=_key(doc_id))
        item = response.get("Item")
        return self._deserialize(item) if item else None

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

    def _batch_get(
        self, keys: list[str], consistent: bool = False
    ) -> list[dict[str, Any]]:
        table: dict[str, Any] = {"Keys": [_key(k) for k in keys]}
        if consistent:
            table["ConsistentRead"] = True
        request: dict[str, Any] = {self.table_name: table}
        items: list[dict[str, Any]] = []
        for attempt in range(_BATCH_GET_MAX_ATTEMPTS):
            _backoff_before(attempt)
            with self._registry_errors("read records"):
                response = self.client.batch_get_item(RequestItems=request)
            items.extend(response.get("Responses", {}).get(self.table_name, []))
            request = dict(response.get("UnprocessedKeys") or {})
            if not request.get(self.table_name, {}).get("Keys"):
                return items
        remaining = len(request[self.table_name]["Keys"])
        raise DocStatusRegistryError(
            f"Doc-status registry: DynamoDB table '{self.table_name}' left "
            f"{remaining} of {len(keys)} keys unprocessed after "
            f"{_BATCH_GET_MAX_ATTEMPTS} BatchGetItem requests. Raise the table's "
            "capacity or switch it to on-demand billing, then re-run."
        )

    def put(self, record: DocStatusRecord) -> None:
        """Write ``record``; raise ``DataProcessingError`` if it is over 400 KB.

        The record holds the document's text-unit ids (~36 bytes each), so one
        item fits roughly 10,000 of them; a larger document must be split
        into smaller files.
        """
        item = self._sized_item(record)
        with self._registry_errors("write a record"):
            self.client.put_item(TableName=self.table_name, Item=item)

    def put_many(self, records: Iterable[DocStatusRecord]) -> None:
        """Write ``records`` with ``BatchWriteItem``, 25 items per call.

        The last record per ``doc_id`` wins (a batch must not repeat a key).
        Every record is size-checked before the first request, so an
        oversized one (see :meth:`put`) fails the call without writing any.
        Items returned as ``UnprocessedItems`` are re-sent with exponential
        backoff; items still unprocessed after ``_BATCH_GET_MAX_ATTEMPTS``
        requests raise :class:`DocStatusRegistryError`.
        """
        unique = {record.doc_id: record for record in records}
        items = [self._sized_item(record) for record in unique.values()]
        for start in range(0, len(items), _BATCH_WRITE_SIZE):
            self._batch_write(items[start : start + _BATCH_WRITE_SIZE])

    def _batch_write(self, items: list[dict[str, Any]]) -> None:
        request: dict[str, Any] = {
            self.table_name: [{"PutRequest": {"Item": item}} for item in items]
        }
        for attempt in range(_BATCH_GET_MAX_ATTEMPTS):
            _backoff_before(attempt)
            with self._registry_errors("write records"):
                response = self.client.batch_write_item(RequestItems=request)
            request = dict(response.get("UnprocessedItems") or {})
            if not request.get(self.table_name):
                return
        remaining = len(request[self.table_name])
        raise DocStatusRegistryError(
            f"Doc-status registry: DynamoDB table '{self.table_name}' left "
            f"{remaining} of {len(items)} items unprocessed after "
            f"{_BATCH_GET_MAX_ATTEMPTS} BatchWriteItem requests. Raise the "
            "table's capacity or switch it to on-demand billing, then re-run."
        )

    def _sized_item(self, record: DocStatusRecord) -> dict[str, Any]:
        item = self._serialize(record)
        size = self._item_size(item)
        if size > _MAX_ITEM_BYTES:
            raise DataProcessingError(
                f"Doc-status record for '{record.file_path or record.doc_id}' "
                f"lists {len(record.text_unit_ids)} text units ({size} bytes), "
                "over the DynamoDB 400 KB item limit: the document has more "
                "chunks than one registry item holds (roughly 10,000). Split "
                "the document into smaller files"
            )
        return item

    def record_fits(self, record: DocStatusRecord) -> bool:
        """Whether ``record`` is within the DynamoDB 400 KB item limit."""
        return self._item_size(self._serialize(record)) <= _MAX_ITEM_BYTES

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
            self.client.delete_item(TableName=self.table_name, Key=_key(doc_id))

    def _items(self, namespace: str | None, **kwargs: Any) -> Iterator[dict[str, Any]]:
        """The items of ``namespace`` (one paginated Query), or of the whole
        table (a Scan) for ``None``."""
        if namespace is None:
            paginator = self.client.get_paginator("scan")
            pages = paginator.paginate(TableName=self.table_name, **kwargs)
        else:
            names = {
                **kwargs.pop("ExpressionAttributeNames", {}),
                "#ns": _PARTITION_KEY,
            }
            pages = self.client.get_paginator("query").paginate(
                TableName=self.table_name,
                KeyConditionExpression="#ns = :ns",
                ExpressionAttributeNames=names,
                ExpressionAttributeValues={":ns": {"S": namespace}},
                **kwargs,
            )
        for page in pages:
            for item in page.get("Items", []):
                if not item.get(_SORT_KEY, {}).get("S", "").startswith("#"):
                    yield item

    # --- the namespace write lease -------------------------------------------

    @staticmethod
    def _lease_key(namespace: str) -> dict[str, Any]:
        return {_PARTITION_KEY: {"S": namespace}, _SORT_KEY: {"S": _LEASE_KEY}}

    def try_acquire_lease(self, namespace: str, owner: str, seconds: float) -> bool:
        """A conditional put: it succeeds only when no lease is held, the held
        one expired, or ``owner`` holds it."""
        now = time.time()
        with self._registry_errors("take the namespace lease"):
            try:
                self.client.put_item(
                    TableName=self.table_name,
                    Item={
                        **self._lease_key(namespace),
                        "lease_owner": {"S": owner},
                        LEASE_EXPIRY_ATTRIBUTE: {"N": f"{now + seconds:.3f}"},
                    },
                    ConditionExpression=(
                        "attribute_not_exists(#id) OR #expiry < :now "
                        "OR #owner = :owner"
                    ),
                    ExpressionAttributeNames={
                        "#id": _SORT_KEY,
                        "#expiry": LEASE_EXPIRY_ATTRIBUTE,
                        "#owner": "lease_owner",
                    },
                    ExpressionAttributeValues={
                        ":now": {"N": f"{now:.3f}"},
                        ":owner": {"S": owner},
                    },
                )
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code")
                if code == "ConditionalCheckFailedException":
                    return False
                raise
        return True

    def lease_holder(self, namespace: str) -> tuple[str, float] | None:
        with self._registry_errors("read the namespace lease"):
            item = self.client.get_item(
                TableName=self.table_name,
                Key=self._lease_key(namespace),
                ConsistentRead=True,
            ).get("Item")
        if not item:
            return None
        return (
            item.get("lease_owner", {}).get("S", ""),
            float(item.get(LEASE_EXPIRY_ATTRIBUTE, {}).get("N", "0")),
        )

    def release_lease(self, namespace: str, owner: str) -> None:
        with self._registry_errors("release the namespace lease"):
            try:
                self.client.delete_item(
                    TableName=self.table_name,
                    Key=self._lease_key(namespace),
                    ConditionExpression="#owner = :owner",
                    ExpressionAttributeNames={"#owner": "lease_owner"},
                    ExpressionAttributeValues={":owner": {"S": owner}},
                )
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code")
                if code != "ConditionalCheckFailedException":
                    raise

    # --- the extraction settings of a scope's last commit ---------------------

    @staticmethod
    def _settings_key(scope: str) -> dict[str, Any]:
        return {
            _PARTITION_KEY: {"S": scope_namespace(scope) or scope},
            _SORT_KEY: {"S": f"{_SETTINGS_KEY_PREFIX}{scope}"},
        }

    def get_extraction_settings(
        self, scope: str
    ) -> tuple[str, ExtractionSettings] | None:
        with self._registry_errors("read the extraction settings"):
            item = self.client.get_item(
                TableName=self.table_name,
                Key=self._settings_key(scope),
                ConsistentRead=True,
            ).get("Item")
        if not item:
            return None
        return (
            item["extraction_fingerprint"]["S"],
            ExtractionSettings.model_validate_json(item["extraction_settings"]["S"]),
        )

    def put_extraction_settings(
        self, scope: str, fingerprint: str, settings: ExtractionSettings
    ) -> None:
        with self._registry_errors("record the extraction settings"):
            self.client.put_item(
                TableName=self.table_name,
                Item={
                    **self._settings_key(scope),
                    "extraction_fingerprint": {"S": fingerprint},
                    "extraction_settings": {"S": settings.model_dump_json()},
                },
            )

    def list_namespace(self, namespace: str) -> list[DocStatusRecord]:
        with self._registry_errors("query the records"):
            return [self._deserialize(item) for item in self._items(namespace)]

    def list_all(self) -> list[DocStatusRecord]:
        with self._registry_errors("scan the records"):
            return [self._deserialize(item) for item in self._items(None)]

    def _stored_states(self, namespace: str | None) -> dict[str, StoredDocument]:
        """The records of ``namespace`` (every namespace for ``None``) as the
        delta rule reads them, projected: ``diff`` needs no text units."""
        states: dict[str, StoredDocument] = {}
        for item in self._items(
            namespace,
            # Attribute-name placeholders keep the projection clear of
            # DynamoDB reserved words.
            ProjectionExpression="#id, #hash, #scope, #status, #extraction",
            ExpressionAttributeNames={
                "#id": _SORT_KEY,
                "#hash": "content_hash",
                "#extraction": "extraction_fingerprint",
                "#scope": _SCOPE_ATTRIBUTE,
                "#status": "status",
            },
        ):
            doc_id = item.get(_SORT_KEY, {}).get("S")
            if doc_id is not None:
                states[doc_id] = StoredDocument(
                    item.get("content_hash", {}).get("S", ""),
                    item.get(_SCOPE_ATTRIBUTE, {}).get("S"),
                    item.get("status", {}).get("S") == DocStatus.FAILED.value,
                    item.get("extraction_fingerprint", {}).get("S"),
                )
        return states

    def diff(
        self,
        incoming: dict[str, str],
        scope: str | None = None,
        extraction_fingerprint: str | None = None,
    ) -> DocumentDelta:
        """Classify ``{doc_id: content_hash}`` against persisted state.

        With ``scope``, only the scope's namespace is queried; the partition
        is ``DocumentDelta.classify``, as for every registry.
        """
        namespace = scope_namespace(scope) if scope is not None else None
        operation = "scan the records" if namespace is None else "query the records"
        with self._registry_errors(operation):
            stored = self._stored_states(namespace)
        return DocumentDelta.classify(incoming, stored, scope, extraction_fingerprint)

    @staticmethod
    def _serialize(record: DocStatusRecord) -> dict[str, Any]:
        """Convert a record into a DynamoDB item (low-level attribute format)."""
        item: dict[str, Any] = {
            _PARTITION_KEY: {"S": doc_id_namespace(record.doc_id)},
            _SORT_KEY: {"S": record.doc_id},
            "content_hash": {"S": record.content_hash},
            "status": {"S": record.status.value},
            "suffix": {"S": record.suffix},
            _SCOPE_ATTRIBUTE: (
                {"S": record.scope} if record.scope is not None else {"NULL": True}
            ),
            "text_unit_ids": (
                {"SS": record.text_unit_ids} if record.text_unit_ids else {"NULL": True}
            ),
        }
        # Optional scalar string attributes.
        for attr in (
            "file_path",
            "error_info",
            "attempted_hash",
            "extraction_fingerprint",
            "created_at",
            "updated_at",
        ):
            value = getattr(record, attr)
            item[attr] = {"S": value} if value is not None else {"NULL": True}
        item["failure_count"] = {"N": str(record.failure_count)}
        item["attempt_count"] = {"N": str(record.attempt_count)}
        item["crash_batch"] = {"N": str(record.crash_batch)}
        item["attempt_batch"] = (
            {"N": str(record.attempt_batch)}
            if record.attempt_batch is not None
            else {"NULL": True}
        )
        return item

    @staticmethod
    def _deserialize(item: dict[str, Any]) -> DocStatusRecord:
        def _str(attr: str) -> str | None:
            cell = item.get(attr)
            return cell["S"] if cell and "S" in cell else None

        def _str_set(attr: str) -> list[str]:
            cell = item.get(attr)
            # A string set has no order: sort it so a record reads back the
            # same on every scan (and compares equal to its written copy).
            return sorted(cell["SS"]) if cell and "SS" in cell else []

        def _int(attr: str) -> int:
            cell = item.get(attr)
            return int(cell["N"]) if cell and "N" in cell else 0

        return DocStatusRecord(
            doc_id=item[_SORT_KEY]["S"],
            content_hash=item["content_hash"]["S"],
            status=item["status"]["S"],
            suffix=_str("suffix") or "default",
            scope=_str(_SCOPE_ATTRIBUTE),
            file_path=_str("file_path"),
            text_unit_ids=_str_set("text_unit_ids"),
            error_info=_str("error_info"),
            extraction_fingerprint=_str("extraction_fingerprint"),
            failure_count=_int("failure_count"),
            attempt_count=_int("attempt_count"),
            attempted_hash=_str("attempted_hash"),
            crash_batch=_int("crash_batch"),
            attempt_batch=_int("attempt_batch") or None,
            created_at=_str("created_at"),
            updated_at=_str("updated_at"),
        )


def _backoff_before(attempt: int) -> None:
    """Wait before re-sending a batch's unprocessed items (none before the
    first request)."""
    if attempt:
        time.sleep(min(_BATCH_GET_MAX_DELAY, _BATCH_GET_BASE_DELAY * 2**attempt))


def _key(doc_id: str) -> dict[str, Any]:
    """The item key of ``doc_id``: its namespace and itself."""
    return {
        _PARTITION_KEY: {"S": doc_id_namespace(doc_id)},
        _SORT_KEY: {"S": doc_id},
    }
