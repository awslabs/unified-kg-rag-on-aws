# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed facts in retrieval: seed units never reach a TEXT section, and a
source a seed graph states names it.

A seed graph's text units are stored for their lineage only. The retriever
leaves them out of every text-units search (one ``must_not`` beside the
caller's exclusions, in the kNN filter too), local search and the LightRAG
chunk follow drop them from the ids they rank, and ``_source_entry`` adds a
``provenance`` to the sources a seed graph states. Pure logic: the retriever
is built with ``__new__`` over a recording client, the strategies over fake
retrievers.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401
from tests.fixtures.fakes.retrievers import FakeRetriever
from unified_kg_rag.adapters.retrieval.base import query_scope
from unified_kg_rag.adapters.retrieval.strategy_registry import get_strategy_spec
from unified_kg_rag.adapters.retrievers.opensearch_retriever import OpenSearchRetriever
from unified_kg_rag.adapters.search_strategies.local_search import (
    LocalSearchStrategy,
)
from unified_kg_rag.adapters.storage.embedding_settings import (
    configured_embedding_model,
)
from unified_kg_rag.adapters.storage.filter_schema import IndexLimits
from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    seed_provenance,
)
from unified_kg_rag.domain.ingestion.seed_graph import seed_unit_id
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
    SearchType,
)
from unified_kg_rag.shared.exceptions import (
    StoreAccessDeniedError,
    StoreRequestRejectedError,
)

pytestmark = pytest.mark.unit

SEED = seed_unit_id("doc-seed", 0)
EXCLUSION = {"term": {"source_kind": "seed_graph"}}
DIM = 4


@pytest.fixture
def retriever(config: Config) -> OpenSearchRetriever:
    inst = OpenSearchRetriever.__new__(OpenSearchRetriever)
    for name, value in {
        "_config": config,
        "_opensearch_config": config.indexing.opensearch,
        "_max_size": config.indexing.opensearch.max_query_size,
        "_settings_denied_logged": False,
        "_window_cut_warned": set(),
        "_configured_limits": IndexLimits(),
        "_index_limits": {},
        "_unit_kind_mapped": set(),
        "_embedding_model_id": configured_embedding_model(config),
        "_embedding_dimension": DIM,
        "_embedding_checked_aliases": set(),
        "_embedding_check_denied_logged": False,
        "_query_embedding_cache": OrderedDict(),
        "_pending_embeddings": {},
    }.items():
        object.__setattr__(inst, name, value)
    object.__setattr__(inst, "_field_mappings", inst._initialize_field_mappings())
    object.__setattr__(
        inst, "_index_filter_fields", inst._initialize_index_filter_fields()
    )
    return inst


class _Client:
    """Answers each search with one hit; a kNN filter on ``source_kind``
    over an index that does not map it draws OpenSearch 2.13's 400.

    ``mapped``: whether the text-units index maps ``source_kind`` (its
    mappings say so); ``seeded``: whether a unit holds it (the probe finds
    it); ``denied``: the mappings may not be read, ``unread``: their read
    fails; ``probe_fails``: the probe fails.
    """

    def __init__(
        self,
        *,
        mapped: bool | list[bool] = True,
        seeded: bool = False,
        denied: bool = False,
        unread: bool = False,
        probe_fails: bool = False,
    ) -> None:
        self.mapped, self.seeded = mapped, seeded
        self.denied, self.unread, self.probe_fails = denied, unread, probe_fails
        self.bodies: list[dict[str, Any]] = []
        self.probes: list[dict[str, Any]] = []

    async def aget_mappings(self, index: str) -> dict[str, dict[str, Any]]:
        if self.denied:
            raise StoreAccessDeniedError("denied")
        if self.unread:
            raise RuntimeError("connection reset")
        # A list: the alias's indexes, each mapping the field or not.
        found = {}
        for n, mapped in enumerate(
            self.mapped if isinstance(self.mapped, list) else [self.mapped], 1
        ):
            properties: dict[str, Any] = {
                "text_embedding": {"type": "knn_vector", "dimension": DIM}
            }
            if mapped:
                properties["source_kind"] = {"type": "keyword"}
            found[f"{index}-{n}"] = {"properties": properties}
        return found

    async def asearch(self, **request: Any) -> dict[str, Any]:
        body = request["body"]
        if "exists" in str(body["query"]):
            self.probes.append(body)
            if self.probe_fails:
                raise RuntimeError("connection reset")  # transient
            return {"hits": {"hits": [{"_id": SEED}] if self.seeded else []}}
        self.bodies.append(body)
        query = str(body["query"])
        if not self.mapped and "knn" in query and "source_kind" in query:
            raise StoreRequestRejectedError("status=400", reason="Rewrite first")
        return {
            "hits": {
                "hits": [
                    {"_index": f"{request['index']}-1", "_id": "u1", "_score": 1.0}
                ]
            }
        }


def _use(retriever: OpenSearchRetriever, client: _Client, mocker) -> None:
    object.__setattr__(retriever, "_opensearch_client", client)
    mocker.patch.object(
        OpenSearchRetriever,
        "_get_query_vector",
        mocker.AsyncMock(side_effect=lambda text, search_type, _: [0.1] * DIM),
    )


def _excluded(body: dict[str, Any]) -> bool:
    return EXCLUSION in _must_not(body["query"])


def _must_not(clause: Any) -> list[Any]:
    """Every ``must_not`` entry anywhere in ``clause``."""
    found: list[Any] = []
    if isinstance(clause, dict):
        for key, value in clause.items():
            if key == "must_not" and isinstance(value, list):
                found.extend(value)
            found.extend(_must_not(value))
    elif isinstance(clause, list):
        for value in clause:
            found.extend(_must_not(value))
    return found


@pytest.mark.parametrize(
    "search_type", [SearchType.HYBRID, SearchType.VECTOR, SearchType.LEXICAL]
)
async def test_every_text_units_search_leaves_the_seed_units_out(
    retriever, config, mocker, search_type
) -> None:
    client = _Client()
    _use(retriever, client, mocker)
    o = config.indexing.opensearch
    await retriever.aretrieve(
        SearchQuery(
            query="who ships widgets",
            search_type=search_type,
            index_prefixes=[o.text_units_index_prefix, o.entities_index_prefix],
        )
    )
    text = [b for b in client.bodies if "text" in str(b["query"])]
    entities = [b for b in client.bodies if "name" in str(b["query"])]
    assert text and entities
    # In the lexical bool and in the kNN filter alike (efficient filtering:
    # no seed unit takes one of the k); no other index is touched.
    assert all(_excluded(b) for b in text)
    assert not any(_excluded(b) for b in entities)
    assert client.probes == []


async def test_a_fetch_by_id_leaves_them_out_beside_the_callers_exclusions(
    retriever, config, mocker
) -> None:
    client = _Client()
    _use(retriever, client, mocker)
    prefix = config.indexing.opensearch.text_units_index_prefix
    await retriever.aretrieve(
        SearchQuery(
            query="",
            search_type=SearchType.LEXICAL,
            top_k=2,
            index_prefixes=[prefix],
            filters={"id": ["u1", SEED]},
            exclude={"community_ids": "c9"},
        )
    )
    (body,) = client.bodies
    assert _must_not(body["query"]) == [{"term": {"community_ids": "c9"}}, EXCLUSION]


async def test_an_ids_only_check_is_left_as_is(retriever, config, mocker) -> None:
    # A filter check of the units an item cites returns no text: a seed
    # unit passes or fails the caller's filters like any other.
    client = _Client()
    _use(retriever, client, mocker)
    prefix = config.indexing.opensearch.text_units_index_prefix
    await retriever.aretrieve(
        SearchQuery(
            query="",
            search_type=SearchType.LEXICAL,
            top_k=1,
            index_prefixes=[prefix],
            filters={"id": [SEED], "attr_department": "legal"},
            source_fields=[],
        )
    )
    (body,) = client.bodies
    assert not _excluded(body)


@pytest.mark.parametrize(
    ("client", "excluded", "probed"),
    [
        # Mapped (every index this release builds): excluded, no probe.
        (_Client(), True, False),
        # An index built before the field, holding no seed unit: nothing to
        # exclude, and the kNN search is not refused.
        (_Client(mapped=False), False, True),
        # Mappings unreadable: the probe decides.
        (_Client(denied=True), False, True),
        (_Client(mapped=False, unread=True), False, True),
        (_Client(denied=True, seeded=True), True, True),
        # A failed probe: excluded (never a seed unit as a chunk).
        (_Client(denied=True, probe_fails=True), True, True),
    ],
)
async def test_an_index_without_the_field_is_probed_for_seed_units(
    retriever, config, mocker, client, excluded, probed
) -> None:
    _use(retriever, client, mocker)
    prefix = config.indexing.opensearch.text_units_index_prefix
    with query_scope():
        results = await retriever.aretrieve(
            SearchQuery(query="widgets", index_prefixes=[prefix])
        )
    assert results
    assert {_excluded(b) for b in client.bodies} == {excluded}
    assert bool(client.probes) == probed
    if probed:
        assert client.probes[0]["query"] == {"exists": {"field": "source_kind"}}


async def test_the_probe_is_asked_once_per_query(retriever, config, mocker) -> None:
    client = _Client(mapped=False)
    _use(retriever, client, mocker)
    prefix = config.indexing.opensearch.text_units_index_prefix
    query = SearchQuery(query="widgets", index_prefixes=[prefix])
    with query_scope():
        await retriever.aretrieve(query)
        await retriever.aretrieve(query)
    assert len(client.probes) == 1
    # The next query sees what an ingestion in between mapped and wrote.
    client.mapped = client.seeded = True
    with query_scope():
        await retriever.aretrieve(query)
    assert len(client.probes) == 2
    assert _excluded(client.bodies[-1])


@pytest.mark.parametrize(
    ("client", "known"),
    [
        (_Client(), True),
        (_Client(mapped=False), False),
        (_Client(denied=True), False),
        (_Client(unread=True), False),
        # An alias over two indexes is known only when both map it.
        (_Client(mapped=[True, False]), False),
        (_Client(mapped=[True, True]), True),
    ],
)
async def test_a_mapped_index_is_known_from_the_embedding_check(
    retriever, config, mocker, client, known
) -> None:
    _use(retriever, client, mocker)
    o = config.indexing.opensearch
    await retriever._check_embedding_settings(
        [o.text_units_index_prefix, o.entities_index_prefix], None
    )
    text_units = retriever._get_name(o.text_units_index_prefix, None)
    assert retriever._unit_kind_mapped == ({text_units} if known else set())


# --------------------------------------------------------------------------- #
# local search and the LightRAG chunk follow
# --------------------------------------------------------------------------- #


def _hit(name: str, units: list[str], score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        content=name,
        score=score,
        source=name,
        retriever_type="entity",
        metadata={"id": name, "text_unit_ids": units},
    )


async def test_local_search_ranks_document_units_only() -> None:
    nodes = [_hit("Vendor", ["u2", SEED, "u1"], 0.9), _hit("Port", [SEED], 0.8)]
    nodes.append(_hit("Buyer", ["u1"], 0.5))
    assert await LocalSearchStrategy._rank_text_unit_ids(nodes) == ["u1", "u2"]


class _SeedLineageGraph(FakeRetriever):
    """Graph expansion: an entity documents and a seed state, and one only
    the seed states."""

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        return [_hit("Vendor", ["u1", SEED]), _hit("Harbor Port", [SEED])]


def _strategy(config: Config, mode: SearchStrategy) -> tuple[Any, FakeRetriever]:
    spec = get_strategy_spec(mode)
    document = FakeRetriever("document")
    graph = _SeedLineageGraph("graph")
    strategy = spec.strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: document,
            RetrieverRole.GRAPH.value: graph,
        },
    )
    strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
        lambda results_dict, top_k, retrieval_multiplier=1, query=None, **_kw: [
            r for results in results_dict.values() for r in results
        ]
    )
    return strategy, document


async def test_local_search_fetches_no_seed_unit(config: Config) -> None:
    strategy, document = _strategy(config, SearchStrategy.LOCAL)
    result = await strategy.asearch(
        SearchQuery(query="who ships", entity_focus=["Vendor"])
    )
    # The seed-only entity is still in the context; its unit is not fetched.
    entities = {r.source for r in result.results if r.retriever_type == "entity"}
    assert "Harbor Port" in entities
    (fetch,) = [q for q in document.calls if q.filters and "id" in q.filters]
    assert fetch.filters is not None and "u1" in fetch.filters["id"]
    assert SEED not in fetch.filters["id"]
    # The PageRank's passage hit is a document unit; the seed unit is none.
    assert set(fetch.filters["id"]) <= {"u1", "graphrag-text-units-1"}


def _mix(config: Config) -> Any:
    strategy, _ = _strategy(config, SearchStrategy.MIX)
    return strategy


def test_the_chunk_follow_drops_seed_units(config: Config) -> None:
    strategy = _mix(config)
    entities = [_hit("Vendor", [SEED, "u1"]), _hit("Harbor Port", [SEED])]
    relationships = [_hit("Vendor->Port", [SEED, "u2"])]
    assert strategy._lineages(entities) == [["u1"]]
    assert strategy._collect_linked_chunk_ids(entities, relationships, limit=10) == (
        ["u1"],
        ["u2"],
    )


class _SeedLineageDocuments(FakeRetriever):
    """Entity and relationship hits citing a seed unit; chunks by id."""

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        ids = (query.filters or {}).get("id")
        if ids:
            return [_hit(unit, []) for unit in ids]
        return [_hit("Harbor Port", [SEED]), _hit("Vendor", ["u1", SEED])]


@pytest.mark.parametrize("mode", [SearchStrategy.MIX, SearchStrategy.HYBRID])
async def test_lightrag_follows_no_seed_unit(config: Config, mode) -> None:
    strategy, _ = _strategy(config, mode)
    documents = _SeedLineageDocuments("document")
    strategy.retrievers[RetrieverRole.DOCUMENT.value] = documents
    await strategy.asearch(
        SearchQuery(
            query="who ships",
            ll_keywords=["Vendor"],
            hl_keywords=["shipping"],
            metadata={"search_strategy": mode.value},
        )
    )
    prefix = config.indexing.opensearch.text_units_index_prefix
    fetched = [
        unit
        for q in documents.calls
        if q.index_prefixes == [prefix] and q.filters and "id" in q.filters
        for unit in q.filters["id"]
    ]
    assert fetched == ["u1"]


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


def _result(metadata: dict[str, Any], section: str = "entity") -> RetrievalResult:
    return RetrievalResult(
        content="Harbor Port",
        score=0.5,
        source="e1",
        retriever_type=section,
        metadata=metadata,
    )


def test_a_source_without_seed_evidence_is_reported_as_before() -> None:
    metadata = {"id": "e1", "text_unit_ids": ["u1"], "document_ids": ["d1"]}
    entry = GraphRAGChain._source_entry(_result(metadata), "x", truncated=False)
    assert entry == {
        "content": "x",
        "source": "e1",
        "score": 0.5,
        "metadata": {
            **metadata,
            "source_id": "e1",
            "chunk_id": None,
            "section_type": "entity",
            "score": 0.5,
            "truncated": False,
        },
    }


@pytest.mark.parametrize(
    ("metadata", "provenance"),
    [
        # Only a seed graph states it: the graph's name from its filters.
        (
            {
                "source_kinds": ["seed_graph"],
                "iri": "urn:x:port",
                "attributes": {"filters": {"seed_graph": "procure"}},
            },
            {"kind": "seed_graph", "seed_graphs": ["procure"], "iri": "urn:x:port"},
        ),
        # Documents too; two graphs, as the filter union keeps them.
        (
            {
                "source_kinds": ["document", "seed_graph"],
                "attr_seed_graph": ["b", "a"],
            },
            {"kind": "document+seed_graph", "seed_graphs": ["a", "b"], "iri": None},
        ),
        # A graph vertex's one-element list read back as its value, and a
        # graph edge's JSON string: no filters there.
        (
            {"source_kinds": "seed_graph"},
            {"kind": "seed_graph", "seed_graphs": [], "iri": None},
        ),
        (
            {"source_kinds": '["document", "seed_graph"]'},
            {"kind": "document+seed_graph", "seed_graphs": [], "iri": None},
        ),
    ],
)
def test_a_seed_fact_names_its_seed_graph(
    metadata: dict[str, Any], provenance: dict[str, Any]
) -> None:
    entry = GraphRAGChain._source_entry(_result(metadata), "x", truncated=False)
    assert entry["metadata"]["provenance"] == provenance
    # A seed fact has no chunk.
    assert entry["metadata"]["chunk_id"] is None


def test_provenance_needs_seed_evidence() -> None:
    assert seed_provenance({"source_kinds": ["document"], "iri": "urn:x"}) is None
    assert seed_provenance({"iri": "urn:x"}) is None
    assert seed_provenance({"source_kinds": "not json ["}) is None


def test_a_retrievers_own_provenance_is_kept() -> None:
    metadata = {"source_kinds": ["seed_graph"], "provenance": "custom"}
    entry = GraphRAGChain._source_entry(_result(metadata), "x", truncated=False)
    assert entry["metadata"]["provenance"] == "custom"
