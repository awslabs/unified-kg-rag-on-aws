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
    DocumentLineage,
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
# Lineage overflow (see add_lineage_overflow) lives in the same table, one
# item per part under the key "<doc_id>#pending#<n>", n = 0, 1, ... with no
# gap. The record-kind attribute tells a part from a registry record: every
# read of records (get, get_many, list_all, diff) skips the parts.
_RECORD_KIND_ATTRIBUTE = "record_kind"
_OVERFLOW_KIND = "lineage_overflow"
_OVERFLOW_OWNER_ATTRIBUTE = "owner_doc_id"
# Ids per part, well under the item limit: room for the key, kind and owner
# attributes, and for a size count slightly off DynamoDB's (parts are cheap).
_OVERFLOW_PART_BYTES = 380 * 1024
# Parts probed per document in each BatchGetItem round after the first, which
# probes part 0 only (most documents have no overflow).
_OVERFLOW_PROBE = 8
_LINEAGE_FIELDS = (
    "entity_ids",
    "relationship_ids",
    "text_unit_ids",
    "community_ids",
    "claim_ids",
    "community_report_ids",
)
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
        # Overflow part keys by owner doc_id, as the last full scan (diff or
        # list_all) found them.
        self._scanned_overflow: dict[str, set[str]] = {}

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
                "GetItem, BatchGetItem, PutItem, BatchWriteItem, DeleteItem and "
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
        if not item or _is_overflow(item):
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
                if _is_overflow(item):
                    continue
                record = self._deserialize(item)
                records[record.doc_id] = record
        return records

    def _batch_get(
        self, keys: list[str], consistent: bool = False
    ) -> list[dict[str, Any]]:
        table: dict[str, Any] = {"Keys": [{_PARTITION_KEY: {"S": k}} for k in keys]}
        if consistent:
            table["ConsistentRead"] = True
        request: dict[str, Any] = {self.table_name: table}
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
            f"{remaining} of {len(keys)} keys unprocessed after "
            f"{_BATCH_GET_MAX_ATTEMPTS} BatchGetItem requests. Raise the table's "
            "capacity or switch it to on-demand billing, then re-run."
        )

    def put(self, record: DocStatusRecord) -> None:
        """Write ``record``; raise ``DataProcessingError`` if it is over 400 KB.

        The record holds every artifact id of the document (~36 bytes each),
        so one item fits roughly 10,000 ids; a larger document must be split
        into smaller files. (A write-ahead record that does not fit keeps its
        extra ids in lineage overflow instead, see
        :meth:`add_lineage_overflow`.)
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
            if attempt:
                time.sleep(
                    min(_BATCH_GET_MAX_DELAY, _BATCH_GET_BASE_DELAY * 2**attempt)
                )
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
                f"lists {_lineage_size(record)} artifact ids ({size} bytes), "
                "over the DynamoDB 400 KB item limit: the document produces "
                "more artifacts than one registry item holds (roughly 10,000 "
                "ids). Split the document into smaller files"
            )
        return item

    def record_fits(self, record: DocStatusRecord) -> bool:
        """Whether ``record`` is within the DynamoDB 400 KB item limit."""
        return self._item_size(self._serialize(record)) <= _MAX_ITEM_BYTES

    def add_lineage_overflow(self, lineages: Iterable[DocumentLineage]) -> None:
        """Append each lineage's ids not yet in its overflow, as new parts.

        Parts are only appended (from the first free index) until
        :meth:`delete_lineage_overflow` removes them all, so an interrupted
        append can lose only the ids it was adding, never earlier ones.
        """
        wanted: dict[str, dict[str, set[str]]] = {}
        for lineage in lineages:
            fields = wanted.setdefault(lineage.doc_id, {})
            for name in _LINEAGE_FIELDS:
                fields.setdefault(name, set()).update(getattr(lineage, name))
        if not wanted:
            return
        existing = self._read_overflow(list(wanted))
        items: list[dict[str, Any]] = []
        for doc_id, fields in wanted.items():
            parts = existing.get(doc_id, [])
            stored = _merge_parts(parts)
            new = {
                name: sorted(fields[name] - set(stored[name]))
                for name in _LINEAGE_FIELDS
            }
            for offset, chunk in enumerate(_chunk_lineage(new)):
                items.append(_overflow_item(doc_id, len(parts) + offset, chunk))
        for start in range(0, len(items), _BATCH_WRITE_SIZE):
            self._batch_write(items[start : start + _BATCH_WRITE_SIZE])

    def get_lineage_overflow(
        self, doc_ids: Iterable[str]
    ) -> dict[str, DocumentLineage]:
        """Read the overflow of ``doc_ids`` (batched; documents without any
        cost one key each)."""
        return {
            doc_id: DocumentLineage.model_validate(
                {"doc_id": doc_id, **_merge_parts(parts)}
            )
            for doc_id, parts in self._read_overflow(list(doc_ids)).items()
        }

    def delete_lineage_overflow(self, doc_ids: Iterable[str]) -> None:
        """Delete every overflow part of ``doc_ids``, the last part first, so
        an interrupted delete leaves a gap-free prefix that still reads back."""
        keys = [
            _overflow_key(doc_id, n)
            for doc_id, parts in self._read_overflow(list(doc_ids)).items()
            for n in reversed(range(len(parts)))
        ]
        with self._registry_errors("delete lineage overflow"):
            for key in keys:
                self.client.delete_item(
                    TableName=self.table_name, Key={_PARTITION_KEY: {"S": key}}
                )

    def _read_overflow(self, doc_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """``{doc_id: [part 0, part 1, ...]}`` for the doc_ids with overflow.

        Reads up to the first missing part: part 0 of every document first,
        then ``_OVERFLOW_PROBE`` parts per round for the documents whose parts
        have not ended yet.
        """
        found: dict[str, list[dict[str, Any]]] = {}
        probe = dict.fromkeys(doc_ids, 0)
        window = 1
        while probe:
            keys = [
                _overflow_key(doc_id, n)
                for doc_id, start in probe.items()
                for n in range(start, start + window)
            ]
            items: dict[str, dict[str, Any]] = {}
            for start in range(0, len(keys), _BATCH_GET_SIZE):
                # Strongly consistent: an append numbers its parts after the
                # ones read, so it must see every part written so far.
                batch = keys[start : start + _BATCH_GET_SIZE]
                for item in self._batch_get(batch, consistent=True):
                    if _is_overflow(item):
                        items[item[_PARTITION_KEY]["S"]] = item
            following: dict[str, int] = {}
            for doc_id, start in probe.items():
                n = start
                while n < start + window and _overflow_key(doc_id, n) in items:
                    found.setdefault(doc_id, []).append(items[_overflow_key(doc_id, n)])
                    n += 1
                if n == start + window:
                    following[doc_id] = n
            probe = following
            window = _OVERFLOW_PROBE
        return found

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
                    if not _is_overflow(item):
                        records.append(self._deserialize(item))
        return records

    def _scan_fingerprints(
        self,
    ) -> dict[str, tuple[str, str | None, bool, bool]]:
        """Scan only ``{doc_id: (content_hash, scope, failed, pending)}``.

        ``diff`` needs just the partition key, the content hash, the scope and
        whether the record is FAILED or PENDING,
        so this uses a ``ProjectionExpression`` to fetch those attributes instead
        of deserializing the full ``DocStatusRecord`` (content hash + six
        artifact-id lists) for every row. (A full ``scan`` is still required
        because deletion detection needs every stored doc_id of the scope.)
        The lineage-overflow parts it passes are noted by owner
        (``_scanned_overflow``).
        """
        fingerprints: dict[str, tuple[str, str | None, bool, bool]] = {}
        overflow: dict[str, set[str]] = {}
        paginator = self.client.get_paginator("scan")
        for page in paginator.paginate(
            TableName=self.table_name,
            # Attribute-name placeholders keep the projection clear of
            # DynamoDB reserved words.
            ProjectionExpression="#pk, #hash, #scope, #status, #kind, #owner",
            ExpressionAttributeNames={
                "#pk": _PARTITION_KEY,
                "#hash": "content_hash",
                "#scope": _SCOPE_ATTRIBUTE,
                "#status": "status",
                "#kind": _RECORD_KIND_ATTRIBUTE,
                "#owner": _OVERFLOW_OWNER_ATTRIBUTE,
            },
        ):
            for item in page.get("Items", []):
                if _is_overflow(item):
                    # Lineage overflow is no document: never new, changed,
                    # deleted or a stored scope.
                    _note_overflow(overflow, item)
                    continue
                doc_id = item.get(_PARTITION_KEY, {}).get("S")
                content_hash = item.get("content_hash", {}).get("S", "")
                scope = item.get(_SCOPE_ATTRIBUTE, {}).get("S")
                status = item.get("status", {}).get("S")
                if doc_id is not None:
                    fingerprints[doc_id] = (
                        content_hash,
                        scope,
                        status == DocStatus.FAILED.value,
                        status == DocStatus.PENDING.value,
                    )
        self._scanned_overflow = overflow
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
            for doc_id, (_, stored_scope, _, _) in stored.items()
            if doc_id not in incoming_ids and (scope is None or stored_scope == scope)
        ]
        # Already projected for the deletion filter: reporting it is free.
        delta.stored_scopes = sorted(
            {s for _, s, _, _ in stored.values() if s is not None}
        )
        # Overflow is only read with a PENDING record: next to any other
        # record, or none, it is left over (see DocumentDelta).
        delta.orphan_overflow = sorted(
            owner
            for owner in self._scanned_overflow
            if owner not in stored or not stored[owner][3]
        )
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
            # A string set has no order: sort it so a record reads back the
            # same on every scan (and compares equal to its written copy).
            return sorted(cell["SS"]) if cell and "SS" in cell else []

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


def _is_overflow(item: dict[str, Any]) -> bool:
    """Whether ``item`` is a lineage-overflow part, not a registry record."""
    return bool(item.get(_RECORD_KIND_ATTRIBUTE, {}).get("S") == _OVERFLOW_KIND)


def _overflow_key(doc_id: str, part: int) -> str:
    return f"{doc_id}#pending#{part}"


def _note_overflow(found: dict[str, set[str]], item: dict[str, Any]) -> None:
    """Add overflow part ``item``'s key under its owner doc_id to ``found``."""
    owner = item.get(_OVERFLOW_OWNER_ATTRIBUTE, {}).get("S")
    key = item.get(_PARTITION_KEY, {}).get("S")
    if owner is not None and key is not None:
        found.setdefault(owner, set()).add(key)


def _lineage_size(record: DocStatusRecord) -> int:
    return sum(len(getattr(record, name)) for name in _LINEAGE_FIELDS)


def _merge_parts(parts: list[dict[str, Any]]) -> dict[str, list[str]]:
    """``{lineage field: sorted ids}`` over overflow ``parts``."""
    return {
        name: sorted(
            {value for part in parts for value in part.get(name, {}).get("SS", [])}
        )
        for name in _LINEAGE_FIELDS
    }


def _chunk_lineage(lineage: dict[str, list[str]]) -> list[dict[str, list[str]]]:
    """Split ``{lineage field: ids}`` into parts that each fit one item."""
    parts: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] = {}
    size = 0
    for name in _LINEAGE_FIELDS:
        for value in lineage.get(name, []):
            cost = len(value.encode()) + (0 if name in current else len(name))
            if current and size + cost > _OVERFLOW_PART_BYTES:
                parts.append(current)
                current, size = {}, 0
                cost = len(value.encode()) + len(name)
            current.setdefault(name, []).append(value)
            size += cost
    if current:
        parts.append(current)
    return parts


def _overflow_item(
    doc_id: str, part: int, lineage: dict[str, list[str]]
) -> dict[str, Any]:
    item: dict[str, Any] = {
        _PARTITION_KEY: {"S": _overflow_key(doc_id, part)},
        _RECORD_KIND_ATTRIBUTE: {"S": _OVERFLOW_KIND},
        _OVERFLOW_OWNER_ATTRIBUTE: {"S": doc_id},
    }
    for name, ids in lineage.items():
        if ids:
            item[name] = {"SS": ids}
    return item
