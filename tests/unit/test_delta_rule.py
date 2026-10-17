# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one delta rule every doc-status registry applies (``DocumentDelta.classify``).

The DynamoDB, SQLite and in-memory registries used to each carry a copy of
it; they now only read their records, so a rule change cannot reach one of
them and not the others.
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from unified_kg_rag.adapters.doc_status.sqlite import SQLiteDocStatusStore
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    DocStatusRecord,
    DocumentDelta,
    StoredDocument,
)

pytestmark = pytest.mark.unit

_STORED = {
    "same": StoredDocument("h1", "s", False, "fp"),
    "edited": StoredDocument("old", "s", False, "fp"),
    "failed": StoredDocument("h3", "s", True, "fp"),
    "resettled": StoredDocument("h4", "s", False, "other-fp"),
    "unknown-fp": StoredDocument("h5", "s", False, None),
    "gone": StoredDocument("h6", "s", False, None),
    "elsewhere": StoredDocument("h7", "t", False, None),
}
_INCOMING = {
    "same": "h1",
    "edited": "new",
    "failed": "h3",
    "resettled": "h4",
    "unknown-fp": "h5",
    "fresh": "h8",
}


def test_classify_partitions_the_corpus() -> None:
    delta = DocumentDelta.classify(_INCOMING, _STORED, "s", "fp")
    assert delta.new == ["fresh"]
    assert delta.changed == ["edited", "failed", "resettled"]
    assert delta.settings_changed == ["resettled"]
    assert delta.unchanged == ["same", "unknown-fp"]
    assert delta.deleted == ["gone"]  # "elsewhere" is another scope's
    assert delta.stored_scopes == ["s", "t"]


def test_without_a_scope_or_fingerprint() -> None:
    delta = DocumentDelta.classify(_INCOMING, _STORED)
    assert delta.deleted == ["gone", "elsewhere"]
    assert "resettled" in delta.unchanged and not delta.settings_changed


@pytest.mark.parametrize("store_kind", ["fake", "sqlite"])
def test_registries_apply_the_rule(store_kind: str, tmp_path) -> None:
    store = (
        FakeDocStatusStore()
        if store_kind == "fake"
        else SQLiteDocStatusStore(Config(), tmp_path / "registry.sqlite3")
    )
    for doc_id, stored in _STORED.items():
        store.put(
            DocStatusRecord(
                doc_id=doc_id,
                content_hash=stored.content_hash,
                scope=stored.scope,
                status=DocStatus.FAILED if stored.failed else DocStatus.PROCESSED,
                extraction_fingerprint=stored.extraction_fingerprint,
            )
        )
    assert store.diff(_INCOMING, None, "fp") == DocumentDelta.classify(
        _INCOMING, _STORED, None, "fp"
    )
