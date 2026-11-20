# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The path filters ingestion stamps on every corpus document (``file``,
``folder``), and how a build record and a query tell a namespace built
without them (``IndexBuildRecord.document_filter_keys``)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from unified_kg_rag.application.retrieval.filter_validation import (
    check_path_filter_keys,
)
from unified_kg_rag.domain.ingestion.delta_detector import assign_document_identity
from unified_kg_rag.domain.ingestion.document_identity import (
    PATH_FILTER_KEYS,
    path_filters,
)
from unified_kg_rag.domain.ingestion.methodologies import namespace_holdings
from unified_kg_rag.domain.models import Config, Document, IndexBuildRecord
from unified_kg_rag.shared import InvalidFilterError

pytestmark = pytest.mark.unit

_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _document(path: str, **metadata) -> Document:
    return Document(
        document_id="x",
        file_name=Path(path).name,
        file_path=f"/corpus/{path}",
        page_content="Vendor supplies widgets to Buyer.",
        metadata=metadata,
    )


def test_path_filters_name_the_file_and_every_folder_holding_it() -> None:
    assert path_filters("a/b/c.txt") == {"file": "a/b/c.txt", "folder": ["a", "a/b"]}
    assert path_filters("c.txt") == {"file": "c.txt"}


def test_identity_stamps_the_path_filters_keeping_the_documents_own() -> None:
    plain = _document("contracts/2024/acme.txt")
    assign_document_identity(plain, "/corpus")
    assert plain.metadata["filters"] == {
        "file": "contracts/2024/acme.txt",
        "folder": ["contracts", "contracts/2024"],
    }
    shared = {"folder": "legal", "region": "emea"}
    given = _document("contracts/acme.txt", filters=shared)
    assign_document_identity(given, "/corpus")
    # The document's own value wins; its dict is not changed in place.
    assert given.metadata["filters"] == {
        "file": "contracts/acme.txt",
        "folder": "legal",
        "region": "emea",
    }
    assert shared == {"folder": "legal", "region": "emea"}


def test_a_path_outside_the_corpus_root_is_not_stamped() -> None:
    # A document parsed outside the pipeline keeps its absolute path: a
    # filter never names where the corpus lived on the indexing host.
    outside = _document("acme.txt")
    outside.file_path = "/elsewhere/acme.txt"
    assign_document_identity(outside, "/corpus")
    assert "filters" not in outside.metadata


def test_a_seed_graph_shard_gets_no_path_filters() -> None:
    shard = _document(
        "seed_graphs/products/1-of-1.jsonl",
        source_kind="seed_graph",
        relative_path="seed_graphs/products/1-of-1.jsonl",
        filters={"seed_graph": "products"},
    )
    assign_document_identity(shard, "/corpus")
    assert shard.metadata["filters"] == {"seed_graph": "products"}


def _record(mode: str, keys: list[str] | None) -> IndexBuildRecord:
    return IndexBuildRecord(
        pipeline_id=mode,
        mode=mode,
        built_at=_AT,
        documents=1,
        extraction_fingerprint="fp",
        document_filter_keys=keys,
    )


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (["file", "folder"], ["file", "folder"]),
        (["file"], ["file"]),
        # A record an earlier version wrote: its documents carry none.
        (None, None),
    ],
)
def test_a_delta_record_keeps_only_the_keys_the_stored_record_names(
    stored: list[str] | None, expected: list[str] | None
) -> None:
    previous = _record("full", stored).model_dump(mode="json")
    delta = _record("incremental", list(PATH_FILTER_KEYS)).over(previous)
    assert delta.document_filter_keys == expected


def test_a_full_build_names_its_own_keys_over_an_earlier_record() -> None:
    previous = _record("full", None).model_dump(mode="json")
    full = _record("full", list(PATH_FILTER_KEYS)).over(previous)
    assert full.document_filter_keys == ["file", "folder"]


@pytest.mark.parametrize(
    ("record", "refused"),
    [
        (None, False),
        ({"document_filter_keys": ["file", "folder"]}, False),
        ({"document_filter_keys": ["file"]}, True),
        ({"mode": "full"}, True),
    ],
)
def test_a_path_filter_is_refused_on_a_namespace_without_it(
    record: dict | None, refused: bool
) -> None:
    holdings = namespace_holdings(record, Config())
    filters = {"attr_folder": "contracts"}
    if refused:
        with pytest.raises(InvalidFilterError, match="attr_folder.*'tenant'"):
            check_path_filter_keys(filters, holdings, "tenant")
    else:
        check_path_filter_keys(filters, holdings, "tenant")
    # Another document key and an item's own attribute are not path filters.
    check_path_filter_keys(
        {"attr_region": "emea", "attributes.folder": "x"}, holdings, "tenant"
    )
