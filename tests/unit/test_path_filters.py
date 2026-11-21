# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The path filters ingestion stamps on every corpus document (``file``,
``folder``), and how a build record and a query tell a namespace built
without them (``IndexBuildRecord.document_filter_keys``)."""

from __future__ import annotations

import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import pytest

from unified_kg_rag.application.ingestion.pipeline_stages import _unstamped_documents
from unified_kg_rag.application.retrieval.filter_validation import (
    check_path_filter_keys,
)
from unified_kg_rag.domain.ingestion.delta_detector import assign_document_identity
from unified_kg_rag.domain.ingestion.document_identity import (
    PATH_FILTER_KEYS,
    path_filters,
)
from unified_kg_rag.domain.ingestion.methodologies import namespace_holdings
from unified_kg_rag.domain.ingestion.seed_graph import SEED_UNIT_PREFIX
from unified_kg_rag.domain.models import Config, Document, IndexBuildRecord, TextUnit
from unified_kg_rag.domain.retrieval.filters import typed_filter_values
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


_NFD_FOLDER = unicodedata.normalize("NFD", "서류")


def test_the_stamped_path_filters_are_nfc_and_the_identity_path_as_given() -> None:
    # macOS (and many archive tools) give a Hangul name decomposed; a user
    # types it composed, and the stores compare filter values as written.
    assert _NFD_FOLDER != "서류"
    document = _document(f"{_NFD_FOLDER}/memo.txt")
    assign_document_identity(document, "/corpus")
    assert document.metadata["filters"] == {"file": "서류/memo.txt", "folder": ["서류"]}
    # The document's identity (its relative path, so its ids) is unchanged.
    assert document.metadata["relative_path"] == f"{_NFD_FOLDER}/memo.txt"


@pytest.mark.parametrize("key", ["attr_folder", "attributes.filters.folder"])
def test_a_path_filter_value_matches_its_nfc_form(key: str) -> None:
    assert typed_filter_values({key: _NFD_FOLDER}) == {key: [_NFD_FOLDER, "서류"]}
    assert typed_filter_values({key: [_NFD_FOLDER, "a"]}) == {
        key: [_NFD_FOLDER, "서류", "a"]
    }
    # Already NFC: as given.
    assert typed_filter_values({key: "서류"}) == {key: "서류"}
    # Another document key is compared as written.
    assert typed_filter_values({"attr_region": _NFD_FOLDER}) == {
        "attr_region": _NFD_FOLDER
    }


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("notes 10:30.txt", {"file": "notes 10:30.txt"}),
        ("2024:Q1/r.txt", {"file": "2024:Q1/r.txt", "folder": ["2024:Q1"]}),
        ("C:/corpus/r.txt", None),
        ("C:", None),
        ("x:/y", None),
        # A letter and a colon begin a name, not a drive (R2B-3).
        ("a:b.txt", {"file": "a:b.txt"}),
        ("Q:1 report.txt", {"file": "Q:1 report.txt"}),
        ("B:side/x.txt", {"file": "B:side/x.txt", "folder": ["B:side"]}),
        ("s3://bucket/r.txt", None),
        ("https://host/r.txt", None),
    ],
)
def test_only_a_drive_or_a_url_scheme_is_not_a_relative_path(
    path: str, expected: dict | None
) -> None:
    document = _document("x.txt")
    document.file_path = f"/elsewhere/{path}"
    document.metadata["relative_path"] = path
    assign_document_identity(document, "/corpus")
    assert document.metadata.get("filters") == expected


def _unit(unit_id: str, filters: dict | None) -> TextUnit:
    attributes = None if filters is None else {"filters": filters}
    return TextUnit(id=unit_id, text="Vendor supplies widgets.", attributes=attributes)


def test_a_document_without_path_filters_is_named_not_the_namespace_refused() -> None:
    # One document whose path is not relative to the corpus root used to
    # record no keys, so every attr_file/attr_folder query on the namespace
    # was refused, and a reset recorded none again (R2B-3). It now matches
    # no path filter itself, and ingestion names it.
    stamped = _unit("t1", {"file": "a/b.txt", "folder": ["a"]})
    at_root = _unit("t2", {"file": "b.txt"})
    seed = _unit(f"{SEED_UNIT_PREFIX}products", {"seed_graph": "products"})
    assert _unstamped_documents([stamped, at_root, seed]) == []
    assert _unstamped_documents([]) == []
    outside = TextUnit(
        id="t3",
        text="Vendor supplies widgets.",
        attributes={"file_name": "acme.txt"},
        document_ids=["d3"],
    )
    tagged = TextUnit(
        id="t4",
        text="Vendor supplies widgets.",
        attributes={"filters": {"region": "emea"}},
        document_ids=["d4"],
    )
    assert _unstamped_documents([stamped, outside, tagged, outside]) == [
        "acme.txt",
        "d4",
    ]


def test_a_documents_own_path_filters_are_nfc_too() -> None:
    # A key the document's own filters give was stored as given, so an NFD
    # folder never matched the NFC value a query types (R2B-5).
    given = _document(
        "b.txt",
        filters={
            "folder": _NFD_FOLDER,
            "file": [_NFD_FOLDER, 3],
            "region": _NFD_FOLDER,
        },
    )
    assign_document_identity(given, "/corpus")
    assert given.metadata["filters"] == {
        "file": ["서류", 3],
        "folder": "서류",
        # Another document key is kept as written.
        "region": _NFD_FOLDER,
    }
    # Also where no path is stamped.
    outside = _document("b.txt", filters={"folder": _NFD_FOLDER})
    outside.file_path = "/elsewhere/b.txt"
    outside.metadata["relative_path"] = "C:/elsewhere/b.txt"
    assign_document_identity(outside, "/corpus")
    assert outside.metadata["filters"] == {"folder": "서류"}
