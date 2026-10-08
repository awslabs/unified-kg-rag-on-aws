# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test of the graph and vector adapters against the local stores.

Runs the Neptune and OpenSearch indexers and retrievers against the Gremlin
Server and OpenSearch containers of ``docker/compose.local.yaml``, with a
hashing embedding provider instead of Bedrock, so no AWS call is made. Skipped
unless ``LOCAL_STORES=1``::

    docker compose -f docker/compose.local.yaml up -d --wait
    LOCAL_STORES=1 uv run pytest tests/integration/test_local_stores.py -v
    docker compose -f docker/compose.local.yaml down -v

Every run writes under its own ``indexing.additional_suffix`` and clears it
afterwards, so it does not touch other data in the containers.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from unified_kg_rag.adapters.aws import NeptuneClient, OpenSearchClient
from unified_kg_rag.adapters.retrievers import NeptuneRetriever, OpenSearchRetriever
from unified_kg_rag.adapters.storage import NeptuneIndexer, OpenSearchIndexer
from unified_kg_rag.domain.models import (
    Community,
    CommunityReport,
    Config,
    Entity,
    Relationship,
    RetrievalResult,
    SearchQuery,
    SearchType,
    TextUnit,
)
from unified_kg_rag.domain.models.config import LanguageCode
from unified_kg_rag.shared import get_config

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("LOCAL_STORES") != "1",
        reason="set LOCAL_STORES=1 with docker/compose.local.yaml running",
    ),
]

_LOCAL_CONFIG = Path(__file__).parents[2] / "docker" / "config.local.yaml"
_SUFFIX = "default"


@pytest.fixture(scope="module")
def local_config() -> Config:
    config = get_config(str(_LOCAL_CONFIG))
    config.indexing.additional_suffix = f"smoke{uuid.uuid4().hex[:8]}"
    # Korean target language: the text mappings then use the nori analyzer.
    config.processing.translation.target_language = LanguageCode.KO
    return config


def _entities() -> list[Entity]:
    return [
        Entity(
            id="e-vendor",
            name="Vendor",
            type="ORG",
            description="Vendor supplies widgets",
            text_unit_ids=["t1", "t2"],
            community_ids=["c1"],
            rank=3,
        ),
        Entity(
            id="e-buyer",
            name="Buyer",
            type="ORG",
            description="Buyer orders widgets",
            text_unit_ids=["t1"],
            rank=2,
        ),
        Entity(
            id="e-depot",
            name="Depot",
            type="LOCATION",
            description="Depot stores widgets",
            text_unit_ids=["t2"],
            rank=1,
        ),
    ]


def _relationships() -> list[Relationship]:
    return [
        Relationship(
            id="r-supplies",
            source_id="e-vendor",
            target_id="e-buyer",
            source_name="Vendor",
            target_name="Buyer",
            type="SUPPLIES",
            description="Vendor supplies widgets to Buyer",
            weight=1.0,
            text_unit_ids=["t1"],
        ),
        Relationship(
            id="r-ships",
            source_id="e-vendor",
            target_id="e-depot",
            source_name="Vendor",
            target_name="Depot",
            description="Vendor ships widgets from Depot",
            weight=0.5,
            text_unit_ids=["t2"],
        ),
    ]


def _communities() -> list[Community]:
    return [
        Community(
            id="c1",
            name="Widget supply",
            level="0",
            parent="",
            children=[],
            entity_ids=["e-vendor", "e-buyer", "e-depot"],
            size=3,
        )
    ]


def _text_units() -> list[TextUnit]:
    return [
        TextUnit(
            id="t1",
            text="Vendor supplies widgets to Buyer under a purchase order.",
            translated_texts={"ko": "공급업체가 구매자에게 위젯을 공급한다."},
        ),
        TextUnit(
            id="t2",
            text="Vendor ships widgets from the Depot every week.",
            translated_texts={"ko": "공급업체는 매주 창고에서 위젯을 출하한다."},
        ),
    ]


@pytest.fixture(scope="module")
def graph_indexer(local_config: Config) -> Iterator[NeptuneIndexer]:
    indexer = NeptuneIndexer(local_config)
    try:
        yield indexer
    finally:
        indexer.clear([_SUFFIX])
        indexer.close()


@pytest.fixture(scope="module")
def vector_indexer(local_config: Config) -> Iterator[OpenSearchIndexer]:
    indexer = OpenSearchIndexer(
        local_config, embedding_factory=HashingEmbeddingFactory()
    )
    try:
        yield indexer
    finally:
        indexer.clear([_SUFFIX])
        indexer.close()


async def _graph_retrieve(
    config: Config, *queries: SearchQuery
) -> list[list[RetrievalResult]]:
    retriever = NeptuneRetriever(config, neptune_client=NeptuneClient(config))
    try:
        return [await retriever.aretrieve(query) for query in queries]
    finally:
        await retriever.aclose()


async def _vector_retrieve(
    config: Config, *queries: SearchQuery
) -> list[list[RetrievalResult]]:
    retriever = OpenSearchRetriever(
        config,
        opensearch_client=OpenSearchClient(config),
        embedding_factory=HashingEmbeddingFactory(),
    )
    try:
        return [await retriever.aretrieve(query) for query in queries]
    finally:
        await retriever.aclose()


def test_graph_round_trip(local_config: Config, graph_indexer) -> None:
    entities, relationships = _entities(), _relationships()
    assert graph_indexer.index_entities(entities).failed_items == 0
    assert graph_indexer.index_relationships(relationships).failed_items == 0
    assert graph_indexer.index_communities(_communities()).failed_items == 0
    assert graph_indexer.get_entity_count([_SUFFIX]) == 3

    # Multi-valued vertex properties keep every value (set cardinality).
    by_id = {e.id: e for e in graph_indexer.read_entities(["e-vendor", "e-buyer"])}
    assert sorted(by_id["e-vendor"].text_unit_ids or []) == ["t1", "t2"]
    assert by_id["e-buyer"].text_unit_ids == ["t1"]
    rels = {r.id: r for r in graph_indexer.read_relationships(["r-supplies"])}
    assert (rels["r-supplies"].source_id, rels["r-supplies"].target_id) == (
        "e-vendor",
        "e-buyer",
    )
    assert sorted(graph_indexer.read_entity_names(_SUFFIX)) == [
        ("e-buyer", "Buyer"),
        ("e-depot", "Depot"),
        ("e-vendor", "Vendor"),
    ]

    # Upserts update in place: no duplicate vertices, scalars overwritten.
    updated = entities[0].model_copy(update={"description": "Vendor sells widgets"})
    assert graph_indexer.upsert_entities([updated]).failed_items == 0
    assert graph_indexer.upsert_relationships(relationships[:1]).failed_items == 0
    assert graph_indexer.upsert_communities(_communities()).failed_items == 0
    assert graph_indexer.get_entity_count([_SUFFIX]) == 3
    (vendor,) = graph_indexer.read_entities(["e-vendor"])
    assert vendor.description == "Vendor sells widgets"
    assert sorted(vendor.text_unit_ids or []) == ["t1", "t2"]

    assert len(graph_indexer.read_relationships(["r-supplies"])) == 1

    # Seeded by id, as every search strategy calls it. The indexers are
    # synchronous; retrieval runs on its own event loop.
    labels = local_config.indexing.neptune
    by_seed_id, community = asyncio.run(
        _graph_retrieve(
            local_config,
            SearchQuery(
                query="",
                suffix=_SUFFIX,
                label_prefixes=labels.entity_label_prefix,
                filters={"id": ["e-vendor"]},
            ),
            SearchQuery(
                query="",
                suffix=_SUFFIX,
                label_prefixes=labels.community_label_prefix,
                filters={"id": ["c1"]},
            ),
        )
    )
    assert {r.source for r in by_seed_id} == {"e-vendor", "e-buyer", "e-depot"}
    assert {"c1", "e-vendor"} <= {r.source for r in community}

    assert graph_indexer.find_incident_relationship_ids(["e-depot"], _SUFFIX) == [
        "r-ships"
    ]
    assert graph_indexer.delete_by_id(["e-depot"], _SUFFIX).failed_items == 0
    assert graph_indexer.get_entity_count([_SUFFIX]) == 2


def test_graph_expansion_returns_every_seed_and_its_neighbourhood(
    local_config: Config, graph_indexer
) -> None:
    # Each seed has its own neighbours. One traversal-wide limit used to let
    # the first seeds' neighbourhoods consume it, and the seeds themselves
    # were never emitted.
    seeds = [f"e-seed{i}" for i in range(4)]
    neighbours = {seed: [f"{seed}-n{j}" for j in range(4)] for seed in seeds}
    ids = seeds + [n for group in neighbours.values() for n in group]
    entities = [
        Entity(
            id=entity_id,
            name=entity_id,
            type="ORG",
            description=f"{entity_id} description",
            text_unit_ids=["t1"],
            rank=1,
        )
        for entity_id in ids
    ]
    relationships = [
        Relationship(
            id=f"r-{seed}-{neighbour}",
            source_id=seed,
            target_id=neighbour,
            source_name=seed,
            target_name=neighbour,
            description=f"{seed} works with {neighbour}",
            weight=1.0,
            text_unit_ids=["t1"],
        )
        for seed, group in neighbours.items()
        for neighbour in group
    ]
    assert graph_indexer.index_entities(entities).failed_items == 0
    assert graph_indexer.index_relationships(relationships).failed_items == 0
    try:
        config = local_config.model_copy(deep=True)
        # Keep everything the traversal returns (no rank-then-cut).
        config.indexing.neptune.traversal_fetch_multiplier = 1
        (results,) = asyncio.run(
            _graph_retrieve(
                config,
                SearchQuery(
                    query="",
                    suffix=_SUFFIX,
                    top_k=8,
                    label_prefixes=config.indexing.neptune.entity_label_prefix,
                    filters={"id": seeds},
                ),
            )
        )
        scores = {r.source: r.score for r in results}
        # Every seed comes back at proximity 1.0 (equal ranks -> importance 1.0).
        assert all(scores.get(seed) == 1.0 for seed in seeds), scores
        # 8 new entities split over 4 seeds: 2 neighbours each, none starved.
        for seed, group in neighbours.items():
            found = [n for n in group if n in scores]
            assert len(found) == 2, (seed, sorted(scores))
            assert all(scores[n] == 0.75 for n in found)
    finally:
        graph_indexer.delete_by_id(ids, _SUFFIX)


def test_vector_round_trip(local_config: Config, vector_indexer) -> None:
    text_units_prefix = local_config.indexing.opensearch.text_units_index_prefix
    assert vector_indexer.initialize()
    assert vector_indexer.index_text_units(_text_units()).failed_items == 0
    assert vector_indexer.index_entities(_entities()).failed_items == 0
    assert vector_indexer.index_relationships(_relationships()).failed_items == 0
    assert (
        vector_indexer.index_community_reports(
            [
                CommunityReport(
                    id="cr1",
                    community_id="c1",
                    name="Widget supply",
                    summary="Vendor supplies widgets to Buyer through the Depot.",
                    full_content="Vendor, Buyer and Depot form the widget supply.",
                )
            ]
        ).failed_items
        == 0
    )
    assert vector_indexer.get_entity_count([_SUFFIX]) == 3

    queries = [
        SearchQuery(
            query="Which widgets does Vendor supply?",
            search_type=search_type,
            suffix=_SUFFIX,
        )
        for search_type in SearchType
    ]
    # nori tokenizes the Korean text-unit field.
    korean_query = SearchQuery(
        query="창고",
        search_type=SearchType.LEXICAL,
        suffix=_SUFFIX,
        index_prefixes=text_units_prefix,
    )
    *hit_lists, korean = asyncio.run(
        _vector_retrieve(local_config, *queries, korean_query)
    )
    results = dict(zip(SearchType, hit_lists, strict=True))
    for search_type, hits in results.items():
        assert hits, f"no {search_type.value} results"
        assert {"t1", "e-vendor"} <= {r.source for r in hits}, search_type
    assert [r.source for r in korean] == ["t2"]

    entities_prefix = local_config.indexing.opensearch.entities_index_prefix
    stats = vector_indexer.delete_by_id(["e-depot"], entities_prefix, _SUFFIX)
    assert stats.failed_items == 0
    assert vector_indexer.get_entity_count([_SUFFIX]) == 2
