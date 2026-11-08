# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One filter, one answer: the same caller filter on every store family.

A caller filter is applied by the OpenSearch retriever (on OpenSearch and on
the embedded engine) and by graph expansion (on Gremlin Server/Neptune and
on the SQLite graph). Each family either applies a key exactly as the
others do, or does not declare it (the strategy then checks the graph's
entities against the entities index, ``_graph_unapplied_filters``), or
refuses it (``InvalidFilterError``) where no store can apply it. The
harness kinds of ``tests/fixtures/store_backends.py`` give the families:
``sqlite`` the embedded engine and the SQLite graph, ``container``
(``LOCAL_STORES=1``) OpenSearch and Gremlin Server, ``aws`` OpenSearch
Service and Neptune.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.store_backends import (
    RETRIEVERS,
    StoreHarness,
    backends,
    open_harness,
)
from unified_kg_rag.domain.models import (
    Entity,
    Relationship,
    RetrieverRole,
    SearchQuery,
    SearchType,
)
from unified_kg_rag.shared import InvalidFilterError

pytestmark = pytest.mark.integration

_SUFFIX = "default"
_ALL = ("e1", "e2", "e3", "e4")

# Known divergences, by (case, family): the test fails strictly once fixed.
_KNOWN: dict[tuple[str, str], str] = {}

_REFUSED = "refused"


def _entities() -> list[Entity]:
    # Each entity's ``filters`` attribute holds its documents' metadata
    # (attr_<key>); the other attributes are the entity's own (LLM
    # extracted: attributes.<key>). attr_category and attributes.category
    # disagree on e1 on purpose.
    return [
        Entity(
            id="e1",
            name="Acme",
            type="org",
            description="Acme makes widgets",
            rank=3,
            in_schema=True,
            text_unit_ids=["t1"],
            attributes={
                "category": "hardware",
                "founded": 1999,
                "filters": {"category": "finance", "year": 2024},
            },
        ),
        Entity(
            id="e2",
            name="가나다 상사",
            type="org",
            description="가나다 상사는 위젯을 만든다",
            rank=1,
            in_schema=False,
            text_unit_ids=["t2"],
            attributes={
                "founded": 200,
                "filters": {"category": "finance", "year": "2024"},
            },
        ),
        Entity(
            id="e3",
            name="Beta Labs",
            type="person",
            description="Beta Labs writes software",
            rank=2,
            text_unit_ids=["t3"],
            source_kinds=["seed_graph"],
            attributes={"category": "software", "filters": {"category": "legal"}},
        ),
        Entity(
            id="e4",
            name="ガンマ",
            type="location",
            description="ガンマは倉庫です",
            rank=0,
            text_unit_ids=["t4"],
            source_kinds=["document", "seed_graph"],
            attributes={"seed_properties": {"color": "red"}},
        ),
    ]


def _relationships() -> list[Relationship]:
    return [
        Relationship(
            id=f"r{i}",
            source_id=source,
            target_id=target,
            source_name=source,
            target_name=target,
            description=f"{source} relates to {target}",
        )
        for i, (source, target) in enumerate([("e1", "e2"), ("e2", "e3"), ("e3", "e4")])
    ]


# (case id, filters, expected entity ids or _REFUSED, whether the graph
# applies the keys; a graph that does not leaves them to the entities index)
_CASES: list[tuple[str, dict[str, Any], Any, bool]] = [
    ("document-filter", {"attr_category": "finance"}, {"e1", "e2"}, False),
    ("document-filter-number", {"attr_year": 2024}, {"e1", "e2"}, False),
    ("document-filter-string", {"attr_year": "2024"}, {"e1", "e2"}, False),
    (
        "document-filter-path",
        {"attributes.filters.category": "legal"},
        {"e3"},
        False,
    ),
    ("own-attribute", {"attributes.category": "hardware"}, {"e1"}, True),
    ("own-attribute-list", {"attributes.category": ["software"]}, {"e3"}, True),
    ("own-attribute-range", {"attributes.founded": {"gte": 1000}}, {"e1"}, True),
    ("own-attribute-number-text", {"attributes.founded": "1999"}, {"e1"}, True),
    ("type", {"type": "org"}, {"e1", "e2"}, True),
    ("type-list", {"type": ["org", "person"]}, {"e1", "e2", "e3"}, True),
    ("rank-range", {"rank": {"gte": 2}}, {"e1", "e3"}, True),
    ("rank-string", {"rank": "3"}, {"e1"}, True),
    ("in-schema-string", {"in_schema": "false"}, {"e2"}, True),
    ("source-kinds-seed", {"source_kinds": "seed_graph"}, {"e3", "e4"}, True),
    ("source-kinds-document", {"source_kinds": "document"}, {"e1", "e2", "e4"}, True),
    ("name", {"name": "Acme"}, {"e1"}, True),
    ("name-korean", {"name": "가나다 상사"}, {"e2"}, True),
    ("name-japanese", {"name": "ガンマ"}, {"e4"}, True),
    ("name-is-exact", {"name": "가나다"}, set(), True),
    ("name-case", {"name": "acme"}, set(), True),
    ("name-keyword", {"name.keyword": "Acme"}, {"e1"}, True),
    ("text-field", {"description": "widgets"}, _REFUSED, False),
    (
        "unindexed-path",
        {"attributes.seed_properties.color": "red"},
        _REFUSED,
        False,
    ),
]
_CASE_PARAMS = [pytest.param(*case, id=case[0]) for case in _CASES]


@pytest.fixture(scope="module", params=backends(RETRIEVERS))
def harness(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[StoreHarness]:
    tmp_path: Path = tmp_path_factory.mktemp("filters")
    for h in open_harness(request.param, tmp_path):
        h.initialize()
        assert h.vector_indexer.index_entities(_entities()).failed_items == 0
        assert h.graph_indexer.index_entities(_entities()).failed_items == 0
        assert h.graph_indexer.index_relationships(_relationships()).failed_items == 0
        yield h


def _families(h: StoreHarness) -> tuple[str, str]:
    """The (vector, graph) family names of ``h``'s stores."""
    return (
        ("embedded", "sqlite-graph")
        if h.kind == "sqlite"
        else ("opensearch", "gremlin")
    )


def _expect_known(request: pytest.FixtureRequest, case: str, family: str) -> None:
    reason = _KNOWN.get((case, family))
    if reason:
        request.applymarker(pytest.mark.xfail(reason=reason, strict=True))


@pytest.mark.parametrize("case, filters, expected, graph_applies", _CASE_PARAMS)
def test_the_document_retriever_applies_a_filter_as_documented(
    request: pytest.FixtureRequest,
    harness: StoreHarness,
    case: str,
    filters: dict[str, Any],
    expected: Any,
    graph_applies: bool,
) -> None:
    _expect_known(request, case, _families(harness)[0])
    query = SearchQuery(
        query="",
        suffix=_SUFFIX,
        search_type=SearchType.LEXICAL,
        index_prefixes=[harness.config.indexing.opensearch.entities_index_prefix],
        top_k=10,
        retrieval_multiplier=1,
        filters=filters,
    )
    if expected == _REFUSED:
        with pytest.raises(InvalidFilterError):
            harness.retrieve(RetrieverRole.DOCUMENT, query)
        return
    (hits,) = harness.retrieve(RetrieverRole.DOCUMENT, query)
    assert {r.source for r in hits} == expected


@pytest.mark.parametrize("case, filters, expected, graph_applies", _CASE_PARAMS)
def test_graph_expansion_applies_a_filter_or_leaves_it_to_the_entities_index(
    request: pytest.FixtureRequest,
    harness: StoreHarness,
    case: str,
    filters: dict[str, Any],
    expected: Any,
    graph_applies: bool,
) -> None:
    _expect_known(request, case, _families(harness)[1])
    query = SearchQuery(
        query="",
        suffix=_SUFFIX,
        top_k=10,
        retrieval_multiplier=1,
        max_hops=1,
        filters={"id": list(_ALL), **filters},
    )
    retriever = harness.graph_retriever
    assert retriever is not None
    declared = retriever().filter_fields()
    assert all(declared.declares(key) for key in filters) is graph_applies
    if expected == _REFUSED:
        with pytest.raises(InvalidFilterError):
            harness.retrieve(RetrieverRole.GRAPH, query)
        return
    (hits,) = harness.retrieve(RetrieverRole.GRAPH, query)
    # Every seed comes back when the graph leaves the keys to the index.
    assert {r.source for r in hits} == (expected if graph_applies else set(_ALL))


# A filter on every kind of key, under one namespace, reads nothing of
# another: the suffix, not a filter, separates tenants, and every compiler
# keeps the store name or label the suffix gives.
_TENANT_FILTERS: list[dict[str, Any]] = [
    {},
    {"name": "Acme"},
    {"name.keyword": "Acme"},
    {"type": ["org", "person"]},
    {"attributes.category": "hardware"},
    {"attr_category": "finance"},
    {"source_kinds": "document"},
    {"rank": {"gte": 0}},
]


def test_a_filter_never_reads_another_namespace(harness: StoreHarness) -> None:
    other = harness.suffix("tenant-b")
    # The same ids, names and filter values under another suffix.
    foreign = [
        e.model_copy(
            update={
                "description": "tenant b only",
                "attributes": {**e.attributes, "index": other},
            }
        )
        for e in _entities()
    ]
    assert harness.vector_indexer.index_entities(foreign).failed_items == 0
    assert harness.graph_indexer.index_entities(foreign).failed_items == 0
    prefix = harness.config.indexing.opensearch.entities_index_prefix
    for suffix, own in ((_SUFFIX, False), (other, True)):
        document = [
            SearchQuery(
                query="",
                suffix=suffix,
                search_type=SearchType.LEXICAL,
                index_prefixes=[prefix],
                top_k=10,
                retrieval_multiplier=1,
                filters=filters or None,
            )
            for filters in _TENANT_FILTERS
        ]
        graph = [
            SearchQuery(
                query="",
                suffix=suffix,
                top_k=10,
                retrieval_multiplier=1,
                max_hops=1,
                filters={"id": list(_ALL), **filters},
            )
            for filters in _TENANT_FILTERS
        ]
        for role, queries in (
            (RetrieverRole.DOCUMENT, document),
            (RetrieverRole.GRAPH, graph),
        ):
            for filters, hits in zip(
                _TENANT_FILTERS, harness.retrieve(role, *queries), strict=True
            ):
                assert hits, (suffix, role, filters)
                assert {"tenant b only" in r.content for r in hits} == {own}, (
                    suffix,
                    role,
                    filters,
                )
