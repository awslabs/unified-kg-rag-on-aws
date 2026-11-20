# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vector store's ports: ``VectorIndexer`` and the concern ``Protocol``s
it inherits, with the defaults a third-party store relies on."""

from __future__ import annotations

import inspect

import pytest

from tests.fixtures.fakes.stores import FakeVectorStore
from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.adapters.storage.sqlite_graph_indexer import SQLiteGraphIndexer
from unified_kg_rag.domain.models import Config
from unified_kg_rag.ports.indexer import (
    BuildRecordStore,
    ExtractionRecordStore,
    GraphIndexer,
    SchemaAnnotationStore,
    StoreInspection,
    SummaryTreeStore,
    VectorIndexer,
    keeps_build_records,
    keeps_extraction_records,
    rewrites_schema_annotations,
)

pytestmark = pytest.mark.unit

# The vector store's concerns besides writing, each its own port.
CONCERNS = (
    SchemaAnnotationStore,
    ExtractionRecordStore,
    SummaryTreeStore,
    BuildRecordStore,
    StoreInspection,
)


# The writes both indexers declare, each for its own store's items (the
# vector delete takes an index prefix too): one name, two contracts.
_ONE_NAME_TWO_CONTRACTS = {
    "index_entities",
    "index_relationships",
    "upsert_entities",
    "upsert_relationships",
    "delete_by_id",
}


class _Bare(VectorIndexer):
    """A store implementing none of the optional reads."""

    def __init__(self) -> None:
        super().__init__(Config())


_Bare.__abstractmethods__ = frozenset()


def _own(port: type) -> set[str]:
    return {
        name
        for name, member in vars(port).items()
        if inspect.isfunction(member) and not name.startswith("_")
    }


def test_the_vector_port_inherits_every_concern() -> None:
    # The composition root builds one vector store and the callers take
    # the concern they use: every store that is a VectorIndexer is each.
    for concern in CONCERNS:
        assert concern in VectorIndexer.__mro__, concern.__name__
        assert concern in OpenSearchIndexer.__mro__, concern.__name__
        assert getattr(concern, "_is_protocol", False), concern.__name__
    for graph in (GraphIndexer, NeptuneIndexer, SQLiteGraphIndexer):
        assert SchemaAnnotationStore in graph.__mro__, graph.__name__


def test_each_method_belongs_to_one_port() -> None:
    # A method declared on two of them would leave two defaults that the
    # MRO picks between.
    owners: dict[str, list[str]] = {}
    for port in (*CONCERNS, VectorIndexer, GraphIndexer):
        for name in _own(port) - _ONE_NAME_TWO_CONTRACTS:
            owners.setdefault(name, []).append(port.__name__)
    shared = {name: names for name, names in owners.items() if len(names) > 1}
    assert not shared, shared


def test_the_whole_list_of_text_units_is_gone() -> None:
    # One way to read every stored text unit: a page at a time.
    for store in (VectorIndexer, OpenSearchIndexer, FakeVectorStore):
        assert not hasattr(store, "scan_text_units"), store.__name__


def test_a_bare_store_inherits_every_cannot_default() -> None:
    store = _Bare()
    assert store.scan_text_unit_pages("x", fields=["text"]) is None
    assert store.read_text_unit_fields(["u"], "x", fields=["text"]) is None
    assert store.extraction_record_keys(["fp:"], suffix="x") is None
    assert store.read_extraction_records(["k"], suffix="x") == {}
    assert store.stored_extraction_record_keys(["k"], suffix="x") == set()
    assert store.read_build_record("x") is None
    assert store.describe_indices("x") is None
    assert store.ingested_suffixes() is None
    assert store.scan_items("entities", "x") is None
    assert store.holds_any_text_unit("x") is None
    assert store.read_schema_annotations("entities", "x") is None
    assert store.holds_any_item("x") is None
    assert store.write_schema_annotations("entities", {}, "x").total_items == 0


def test_the_implements_checks_tell_a_bare_store_from_a_backend() -> None:
    bare = _Bare()
    assert not keeps_extraction_records(bare)
    assert not keeps_build_records(bare)
    assert not rewrites_schema_annotations(bare)
    for check in (
        keeps_extraction_records,
        keeps_build_records,
        rewrites_schema_annotations,
    ):
        assert check(OpenSearchIndexer), check.__name__
        assert check(FakeVectorStore), check.__name__
    assert rewrites_schema_annotations(NeptuneIndexer)
