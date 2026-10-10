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
afterwards, so it does not touch other data in the containers. The incremental
pipeline test keeps the doc-status registry in memory (no DynamoDB).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.integration.test_incremental_source_scopes import (
    A_TEXT,
    B_TEXT,
    ScopeStack,
    scope_test_config,
    write_corpus,
)
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


def _graph_entity_count(graph_indexer) -> int:
    label = graph_indexer._get_label(
        graph_indexer.neptune_config.entity_label_prefix, _SUFFIX
    )
    return int(graph_indexer.neptune_client.g.V().hasLabel(label).count().next())


def _vector_entity_count(vector_indexer) -> int:
    alias = vector_indexer._get_name(
        vector_indexer.opensearch_config.entities_index_prefix, _SUFFIX
    )
    return int(vector_indexer.opensearch_client.client.count(index=alias)["count"])


def test_graph_round_trip(local_config: Config, graph_indexer) -> None:
    entities, relationships = _entities(), _relationships()
    assert graph_indexer.index_entities(entities).failed_items == 0
    assert graph_indexer.index_relationships(relationships).failed_items == 0
    assert graph_indexer.index_communities(_communities()).failed_items == 0
    assert _graph_entity_count(graph_indexer) == 3

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
    assert _graph_entity_count(graph_indexer) == 3
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
    assert _graph_entity_count(graph_indexer) == 2


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


def test_community_expansion_expands_every_seed_community(
    local_config: Config, graph_indexer
) -> None:
    # Community seeds share the entity expansion's budget shape: one
    # traversal-wide limit used to let the first communities' members use it.
    communities = [f"c-seed{i}" for i in range(4)]
    members = {c: [f"{c}-m{j}" for j in range(2)] for c in communities}
    neighbours = {
        m: [f"{m}-n{k}" for k in range(2)] for g in members.values() for m in g
    }
    ids = (
        communities
        + [m for group in members.values() for m in group]
        + [n for group in neighbours.values() for n in group]
    )
    entity_ids = [i for i in ids if i not in communities]
    entities = [
        Entity(
            id=entity_id,
            name=entity_id,
            type="ORG",
            description=f"{entity_id} description",
            text_unit_ids=["t1"],
            rank=1,
        )
        for entity_id in entity_ids
    ]
    relationships = [
        Relationship(
            id=f"r-{member}-{neighbour}",
            source_id=member,
            target_id=neighbour,
            source_name=member,
            target_name=neighbour,
            description=f"{member} works with {neighbour}",
            weight=1.0,
            text_unit_ids=["t1"],
        )
        for member, group in neighbours.items()
        for neighbour in group
    ]
    seed_communities = [
        Community(
            id=c,
            name=c,
            level="0",
            parent="",
            children=[],
            entity_ids=members[c],
            size=2,
        )
        for c in communities
    ]
    assert graph_indexer.index_entities(entities).failed_items == 0
    assert graph_indexer.index_relationships(relationships).failed_items == 0
    assert graph_indexer.index_communities(seed_communities).failed_items == 0
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
                    label_prefixes=config.indexing.neptune.community_label_prefix,
                    filters={"id": communities},
                ),
            )
        )
        found = {r.source for r in results}
        # Width 8 over 4 seeds: each community, its 2 members and 2 of their
        # neighbours, none starved.
        for community, group in members.items():
            assert community in found, sorted(found)
            assert set(group) <= found, (community, sorted(found))
            reached = [n for m in group for n in neighbours[m] if n in found]
            assert len(reached) == 2, (community, sorted(found))
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
    assert _vector_entity_count(vector_indexer) == 3

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
    assert _vector_entity_count(vector_indexer) == 2


def _korean_lexical_hits(config: Config, *texts: str) -> list[set[str]]:
    prefix = config.indexing.opensearch.text_units_index_prefix
    queries = [
        SearchQuery(
            query=text,
            search_type=SearchType.LEXICAL,
            suffix=_SUFFIX,
            index_prefixes=prefix,
        )
        for text in texts
    ]
    hit_lists = asyncio.run(_vector_retrieve(config, *queries))
    return [{r.source for r in hits} for hits in hit_lists]


def test_untranslated_korean_corpus_has_lexical_recall(local_config: Config) -> None:
    # A Korean corpus with a Korean target is never translated, so only the
    # ``text`` field carries it. That field must use the source-language
    # analyzer: the standard analyzer keeps "홍길동으로부터" and "보증기간은"
    # as single tokens, so "홍길동" and "보증 기간" never matched them.
    config = local_config.model_copy(deep=True)
    config.indexing.additional_suffix = f"ko{uuid.uuid4().hex[:8]}"
    config.processing.translation.source_language = LanguageCode.KO
    config.processing.translation.target_language = LanguageCode.KO
    indexer = OpenSearchIndexer(config, embedding_factory=HashingEmbeddingFactory())
    try:
        units = [
            TextUnit(id="k1", text="가나다상사는 홍길동으로부터 공급계약서를 받았다."),
            TextUnit(id="k2", text="보증기간은 납품일로부터 2년이다."),
            TextUnit(id="k3", text="김철수는 가나다상사의 직원이다."),
        ]
        assert indexer.index_text_units(units).failed_items == 0
        person, warranty, contract, other = _korean_lexical_hits(
            config, "홍길동", "보증 기간", "공급 계약", "김철민"
        )
        assert "k1" in person
        assert "k2" in warranty
        assert "k1" in contract
        # No fuzzy edit on Hangul: 김철민 is a different person from 김철수.
        assert other == set()
    finally:
        indexer.clear([_SUFFIX])
        indexer.close()


def test_additional_target_language_is_searchable(local_config: Config) -> None:
    # The pipeline translates into every additional target language; each
    # translation is indexed with its own analyzer and searched lexically.
    config = local_config.model_copy(deep=True)
    config.indexing.additional_suffix = f"al{uuid.uuid4().hex[:8]}"
    config.processing.translation.target_language = LanguageCode.EN
    config.processing.translation.additional_target_languages = [LanguageCode.KO]
    indexer = OpenSearchIndexer(config, embedding_factory=HashingEmbeddingFactory())
    try:
        unit = TextUnit(
            id="a1",
            text="The warranty period is two years.",
            translated_texts={
                "en": "The warranty period is two years.",
                "ko": "보증기간은 2년이다.",
            },
        )
        assert indexer.index_text_units([unit]).failed_items == 0
        (hits,) = _korean_lexical_hits(config, "보증 기간")
        assert hits == {"a1"}
    finally:
        indexer.clear([_SUFFIX])
        indexer.close()


def test_full_reindex_keeps_one_index_per_alias(
    local_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each full run builds a new timestamped index and swaps the alias onto
    # it; the previous ones must be deleted, or every run leaks an index until
    # the domain's shard limit blocks index creation.
    start = datetime(2026, 1, 1)
    ticks = iter(range(10_000))

    class _Clock:
        @staticmethod
        def now() -> datetime:
            return start + timedelta(seconds=next(ticks))

    # Second-resolution names: advance the clock so runs never share a name.
    monkeypatch.setattr("unified_kg_rag.shared.utils.store_names.datetime", _Clock)
    config = local_config.model_copy(deep=True)
    config.indexing.additional_suffix = f"bg{uuid.uuid4().hex[:8]}"
    indexer = OpenSearchIndexer(config, embedding_factory=HashingEmbeddingFactory())
    os_client = indexer.opensearch_client
    alias = indexer._get_name(config.indexing.opensearch.entities_index_prefix, None)
    # A sibling store whose index names also match ``<alias>-*``.
    sibling = f"{alias}-2-20250101000000"
    try:
        os_client.client.indices.create(index=sibling)
        for _ in range(3):
            assert indexer.index_entities(_entities()).failed_items == 0

        indices = os_client.get_aliases_by_index(f"{alias}-*")
        live = os_client.get_indices_by_alias(alias)
        assert len(live) == 1
        assert indices == {live[0]: [alias], sibling: []}
        assert _vector_entity_count(indexer) == 3
    finally:
        indexer.clear([_SUFFIX])
        indexer.close()


def test_clear_of_a_namespace_never_indexed_succeeds(local_config: Config) -> None:
    # A reset before a namespace's first run has no alias to delete.
    config = local_config.model_copy(deep=True)
    config.indexing.additional_suffix = f"new{uuid.uuid4().hex[:8]}"
    indexer = OpenSearchIndexer(config, embedding_factory=HashingEmbeddingFactory())
    try:
        assert indexer.clear([_SUFFIX])
    finally:
        indexer.close()


def _stored_texts(indexer: OpenSearchIndexer, config: Config) -> set[str]:
    client = indexer.opensearch_client.client
    alias = indexer._get_name(config.indexing.opensearch.text_units_index_prefix, None)
    if not client.indices.exists_alias(name=alias):
        return set()
    client.indices.refresh(index=alias)
    hits = client.search(
        index=alias,
        body={"size": 100, "query": {"match_all": {}}},
        _source_excludes=["*_embedding"],
    )["hits"]["hits"]
    return {hit["_source"]["text"].strip() for hit in hits}


def test_two_source_scopes_with_the_same_relative_path_keep_their_content(
    local_config: Config, tmp_path: Path
) -> None:
    # Two corpora on one index suffix both hold contract.txt. Each run used to
    # read the other's record as its own changed document and prune it.
    config = scope_test_config(local_config)
    config.indexing.additional_suffix = f"scope{uuid.uuid4().hex[:8]}"
    graph = NeptuneIndexer(config)
    vectors = OpenSearchIndexer(config, embedding_factory=HashingEmbeddingFactory())
    stack = ScopeStack(
        FakeDocStatusStore(), tmp_path, config=config, graph=graph, vectors=vectors
    )
    depot = "Depot stores widgets."
    source_a = write_corpus(
        tmp_path / "src-a", {"contract.txt": A_TEXT, "depot.txt": depot}
    )
    source_b = write_corpus(tmp_path / "src-b", {"contract.txt": B_TEXT})
    try:
        stack.run(source_a)
        stack.run(source_b)
        for source in (source_a, source_b, source_a):
            delta = stack.run(source).incremental_delta
            assert delta.unchanged
            assert delta.new == delta.changed == delta.deleted == []
            assert not stack.model.extractions
            assert _stored_texts(vectors, config) == {A_TEXT, B_TEXT, depot}

        (source_a / "contract.txt").unlink()
        assert len(stack.run(source_a).incremental_delta.deleted) == 1
        assert _stored_texts(vectors, config) == {B_TEXT, depot}
        assert sorted(r.file_path for r in stack.registry.list_all()) == [
            "contract.txt",
            "depot.txt",
        ]
    finally:
        graph.clear([_SUFFIX])
        vectors.clear([_SUFFIX])
        graph.close()
        vectors.close()
