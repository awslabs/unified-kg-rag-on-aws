# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every record a doc-status registry holds, read from its storage directly.

``DocStatusPort`` lists one namespace at a time (``list_namespace``, or
``diff`` for a scope): no run reads the whole registry. A test asserting on
what a run left in it, across namespaces, reads the store's own storage
through :func:`all_records` instead.
"""

from __future__ import annotations

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from unified_kg_rag.adapters.doc_status.dynamodb import DynamoDBDocStatusStore
from unified_kg_rag.adapters.doc_status.sqlite import SQLiteDocStatusStore
from unified_kg_rag.domain.models import DocStatusRecord


def all_records(store: object) -> list[DocStatusRecord]:
    """Every document record ``store`` holds, of every namespace, in the
    order it stores them (the annotation, lease and settings items are not
    document records)."""
    if isinstance(store, FakeDocStatusStore):
        return list(store._records.values())
    if isinstance(store, SQLiteDocStatusStore):
        with store._lock:
            rows = store.connection.execute(
                "SELECT record FROM documents ORDER BY rowid"
            ).fetchall()
        return [DocStatusRecord.model_validate_json(row[0]) for row in rows]
    if isinstance(store, DynamoDBDocStatusStore):
        return [store._deserialize(item) for item in store._items(None)]
    raise TypeError(f"no record reader for {type(store).__name__}")
