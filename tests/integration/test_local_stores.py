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
pipeline test keeps the doc-status registry in memory (no DynamoDB). The
local-mode test runs ``backend: local`` end to end (registry-built stores,
SQLite registry, the Ollama adapters against a fake server), so it needs no
Ollama container and downloads no model.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from gremlin_python.driver.client import Client
from gremlin_python.process.graph_traversal import __
from opensearchpy.exceptions import NotFoundError

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.integration.test_incremental_source_scopes import (
    A_TEXT,
    B_TEXT,
    ScopeStack,
    scope_test_config,
    write_corpus,
)
from unified_kg_rag.adapters.aws.neptune import NeptuneClient
from unified_kg_rag.adapters.aws.opensearch import OpenSearchClient, is_index_not_found
from unified_kg_rag.adapters.retrievers import NeptuneRetriever, OpenSearchRetriever
from unified_kg_rag.adapters.storage import NeptuneIndexer, OpenSearchIndexer
from unified_kg_rag.domain.models import (
    Claim,
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
from unified_kg_rag.shared import ConfigurationError, IndexNotFoundError, get_config

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


def test_gremlin_server_indexes_the_id_property(local_config: Config) -> None:
    # Every indexer write looks its vertex or edge up by "id" first; without
    # an index TinkerGraph scans the whole graph for each lookup.
    neptune = local_config.aws.neptune
    gremlin = Client(f"ws://{neptune.endpoint}:{neptune.port}/gremlin", "g")
    try:
        keys = (
            gremlin.submit(
                "[graph.getIndexedKeys(Vertex.class), graph.getIndexedKeys(Edge.class)]"
            )
            .all()
            .result()
        )
    finally:
        gremlin.close()
    assert [set(k) for k in keys] == [{"id"}, {"id"}]


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
    # A list property is replaced, not merged: an upsert that cites fewer
    # text units drops the others.
    shrunk = updated.model_copy(update={"text_unit_ids": ["t1"]})
    assert graph_indexer.upsert_entities([shrunk]).failed_items == 0
    (vendor,) = graph_indexer.read_entities(["e-vendor"])
    assert vendor.text_unit_ids == ["t1"]
    assert vendor.description == "Vendor sells widgets"

    assert len(graph_indexer.read_relationships(["r-supplies"])) == 1

    # Seeded by id, as every search strategy calls it. The indexers are
    # synchronous; retrieval runs on its own event loop.
    (by_seed_id,) = asyncio.run(
        _graph_retrieve(
            local_config,
            SearchQuery(
                query="",
                suffix=_SUFFIX,
                filters={"id": ["e-vendor"]},
            ),
        )
    )
    assert {r.source for r in by_seed_id} == {"e-vendor", "e-buyer", "e-depot"}

    assert graph_indexer.find_incident_relationship_ids(["e-depot"], _SUFFIX) == [
        "r-ships"
    ]
    assert graph_indexer.delete_by_id(["e-depot"], _SUFFIX).failed_items == 0
    assert _graph_entity_count(graph_indexer) == 2


def test_graph_writes_retried_after_a_lost_response_add_nothing_twice(
    graph_indexer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every request is applied, then its response is lost once, so each one is
    # retried (over a new connection) after it took effect. Regression: the
    # full build's addV and the unconditional addE('MemberOf') then wrote
    # every vertex and membership edge twice, reporting 0 failed.
    from unified_kg_rag.adapters.storage import neptune_indexer as module
    from unified_kg_rag.shared import StoreConnectionError

    real_run = module.run_traversal

    def lose_the_first_response(client, build, *, results=False):
        out = real_run(client, build, results=results)
        if not getattr(build, "lost", False):  # the retries pass the same build
            build.lost = True
            raise StoreConnectionError("response lost")
        return out

    monkeypatch.setattr(module, "run_traversal", lose_the_first_response)
    monkeypatch.setattr(module, "_sleep", lambda _seconds: None)

    def counts() -> tuple[int, int, int, int]:
        g = graph_indexer.neptune_client.g
        entity = graph_indexer._get_label("Entity", _SUFFIX)
        community = graph_indexer._get_label("Community", _SUFFIX)
        edges = g.V().hasLabel(entity).outE()
        return (
            int(g.V().hasLabel(entity).count().next()),
            int(edges.not_(__.hasLabel("MemberOf")).count().next()),
            int(g.V().hasLabel(entity).outE("MemberOf").count().next()),
            int(g.V().hasLabel(community).count().next()),
        )

    assert graph_indexer.index_entities(_entities()).failed_items == 0
    assert graph_indexer.index_relationships(_relationships()).failed_items == 0
    assert graph_indexer.index_communities(_communities()).failed_items == 0
    assert counts() == (3, 2, 3, 1)
    (vendor,) = graph_indexer.read_entities(["e-vendor"], _SUFFIX)
    assert sorted(vendor.text_unit_ids or []) == ["t1", "t2"]

    assert graph_indexer.upsert_entities(_entities()).failed_items == 0
    assert graph_indexer.upsert_relationships(_relationships()).failed_items == 0
    assert graph_indexer.upsert_communities(_communities()).failed_items == 0
    assert counts() == (3, 2, 3, 1)

    assert graph_indexer.delete_by_id(["e-depot"], _SUFFIX).failed_items == 0
    assert counts() == (2, 1, 2, 1)


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
            # One hop out always ranks below every seed (distance orders first).
            assert all(0 < scores[n] < min(scores[x] for x in seeds) for n in found)
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


def test_vector_reads_and_updates_stay_in_a_non_default_suffix(
    local_config: Config,
) -> None:
    # document_parsing.index_value puts a run's artifacts under its own
    # suffix. A claim read back from there (the stored document holds no
    # attributes) must be updated there, not in the default suffix's index
    # where another claim may have the same id.
    config = local_config.model_copy(deep=True)
    config.indexing.additional_suffix = f"sfx{uuid.uuid4().hex[:8]}"
    tenant = "tenant-a"
    indexer = OpenSearchIndexer(config, embedding_factory=HashingEmbeddingFactory())
    prefixes = config.indexing.opensearch
    pipeline_id = f"{config.indexing.additional_suffix}-pipeline"

    def claim(suffix: str, text_unit_ids: list[str]) -> Claim:
        return Claim(
            id="c-supply",
            subject_id="e-vendor",
            subject_name="Vendor",
            object_id="e-buyer",
            object_name="Buyer",
            type="SUPPLY",
            status="TRUE",
            description=f"Vendor supplies Buyer ({suffix})",
            text_unit_ids=text_unit_ids,
            attributes={"index": suffix},
        )

    def stored(prefix: str, suffix: str, doc_id: str) -> dict:
        alias = indexer._get_name(prefix, suffix)
        return indexer.opensearch_client.client.get(index=alias, id=doc_id)["_source"]

    try:
        assert indexer.upsert_claims([claim(tenant, ["t1", "t2"])]).failed_items == 0
        assert indexer.upsert_claims([claim(_SUFFIX, ["t9"])]).failed_items == 0

        (read,) = indexer.read_claims(["c-supply"], suffix=tenant)
        assert read.text_unit_ids == ["t1", "t2"]
        stripped = read.model_copy(update={"text_unit_ids": ["t2"]})
        stats = indexer.update_claim_lineage([stripped])
        assert (stats.successful_items, stats.failed_items) == (1, 0)
        claims_prefix = prefixes.claims_index_prefix
        assert stored(claims_prefix, tenant, "c-supply")["text_unit_ids"] == ["t2"]
        assert stored(claims_prefix, _SUFFIX, "c-supply")["text_unit_ids"] == ["t9"]

        relationships = [
            r.model_copy(update={"attributes": {"index": tenant}})
            for r in _relationships()
        ]
        assert indexer.upsert_relationships(relationships).failed_items == 0
        assert indexer.read_relationship_descriptions(
            ["r-supplies", "r-missing"], suffix=tenant
        ) == {"r-supplies": "Vendor supplies widgets to Buyer"}
        # Another suffix holds no such relationship (its alias does not exist).
        assert indexer.read_relationship_descriptions(["r-supplies"]) == {}

        report = CommunityReport(
            id="cr1",
            community_id="c1",
            name="Widget supply",
            summary="Vendor supplies widgets to Buyer.",
            full_content="Vendor and Buyer form the widget supply.",
            attributes={"index": tenant},
        )
        revised = report.model_copy(update={"summary": "Vendor ships to Buyer."})
        for written in (report, revised):
            assert indexer.upsert_community_reports([written]).failed_items == 0
        reports_prefix = prefixes.community_reports_index_prefix
        alias = indexer._get_name(reports_prefix, tenant)
        os_client = indexer.opensearch_client.client
        os_client.indices.refresh(index=alias)
        assert os_client.count(index=alias)["count"] == 1
        assert stored(reports_prefix, tenant, "cr1")["summary"] == (
            "Vendor ships to Buyer."
        )

        weights = [0.3, 0.7]
        indexer.opensearch_client.put_search_pipeline(
            pipeline_id,
            {
                "phase_results_processors": [
                    {
                        "normalization-processor": {
                            "normalization": {"technique": "min_max"},
                            "combination": {
                                "technique": "arithmetic_mean",
                                "parameters": {"weights": weights},
                            },
                        }
                    }
                ]
            },
        )
        written_pipeline = os_client.search_pipeline.get(id=pipeline_id)[pipeline_id]
        (processor,) = written_pipeline["phase_results_processors"]
        combination = processor["normalization-processor"]["combination"]
        assert combination["parameters"]["weights"] == weights
    finally:
        indexer.clear([tenant, _SUFFIX])
        try:
            indexer.opensearch_client.client.search_pipeline.delete(id=pipeline_id)
        except NotFoundError:
            pass
        indexer.close()


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
        # Another store's index: clear leaves it alone.
        os_client.delete_indices([sibling])
        indexer.close()


def test_index_admin_requests_resent_after_a_lost_response_succeed(
    local_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # opensearch-py resends a request whose response was lost; the resend
    # of a create or delete that already took effect gets the server's
    # "already exists" / "no such index" answer, which is the result asked for.
    os_client = OpenSearchClient(local_config)
    name = f"resend-{uuid.uuid4().hex[:8]}"
    try:
        os_client.client.indices.create(index=name)  # the create that applied
        # create_index saw no index before its request went out.
        monkeypatch.setattr(os_client.client.indices, "exists", lambda index: False)
        os_client.create_index(name, {}, exist_ok=False)
        monkeypatch.undo()
        os_client.client.indices.delete(index=name)  # the delete that applied
        os_client.delete_indices([name])
        assert not os_client.client.indices.exists(index=name)
    finally:
        os_client.client.indices.delete(index=name, ignore_unavailable=True)
        os_client.close()


def test_clear_keeps_sibling_namespaces(local_config: Config) -> None:
    # A reset of one namespace clears its stores only, though the index
    # names of its suffix under an additional suffix, and of a suffix
    # extending it, also match ``<alias>-*``.
    config = local_config.model_copy(deep=True)
    config.indexing.additional_suffix = None
    with_x = config.model_copy(deep=True)
    with_x.indexing.additional_suffix = "x"
    tag = f"clr{uuid.uuid4().hex[:8]}"
    # (item suffix, indexer): the cleared namespace, a suffix extending it,
    # and the same suffix under an additional suffix.
    stores = [
        (suffix, OpenSearchIndexer(c, embedding_factory=HashingEmbeddingFactory()))
        for suffix, c in ((tag, config), (f"{tag}-eu", config), (tag, with_x))
    ]
    cleared = stores[0][1]
    try:
        for suffix, indexer in stores:
            entities = [
                e.model_copy(update={"attributes": {"index": suffix}})
                for e in _entities()
            ]
            assert indexer.index_entities(entities).failed_items == 0
        assert cleared.clear([tag])

        os_client = cleared.opensearch_client
        prefix = config.indexing.opensearch.entities_index_prefix
        assert not os_client.get_indices_by_alias(f"{prefix}-{tag}")
        assert not os_client.get_aliases_by_index(f"{prefix}-{tag}-2*")
        assert os_client.get_indices_by_alias(f"{prefix}-{tag}-eu")
        assert os_client.get_indices_by_alias(f"{prefix}-{tag}-x")
    finally:
        for suffix, indexer in stores:
            indexer.clear([suffix])
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


def test_incremental_write_refuses_another_embedding_model(
    local_config: Config,
) -> None:
    # The index records the model and dimension in its mapping's _meta; a
    # run embedding with another one must not write into it.
    config = local_config.model_copy(deep=True)
    config.indexing.additional_suffix = f"emb{uuid.uuid4().hex[:8]}"
    other_model = config.model_copy(deep=True)
    other_model.indexing.opensearch.embedding_model_id = "synthetic-embed-2"
    indexer = OpenSearchIndexer(config, embedding_factory=HashingEmbeddingFactory())
    changed = [
        OpenSearchIndexer(other_model, embedding_factory=HashingEmbeddingFactory()),
        OpenSearchIndexer(config, embedding_factory=HashingEmbeddingFactory(32)),
    ]
    try:
        assert indexer.upsert_entities(_entities()).failed_items == 0
        indexer.check_index_settings(_SUFFIX)
        for other in changed:
            with pytest.raises(ConfigurationError, match="indexing.reset: true"):
                other.check_index_settings(_SUFFIX)
    finally:
        indexer.clear([_SUFFIX])
        indexer.close()
        for other in changed:
            other.close()


def test_incremental_write_refuses_other_languages_or_an_untracked_build(
    local_config: Config,
) -> None:
    # The real server's mapping echoes each text field's analyzer, and a
    # field it does not map yet would be mapped as a keyword on first write.
    config = local_config.model_copy(deep=True)
    config.indexing.additional_suffix = f"lang{uuid.uuid4().hex[:8]}"
    with_ja = config.model_copy(deep=True)
    with_ja.processing.translation.additional_target_languages = [LanguageCode.JA]
    korean = config.model_copy(deep=True)
    korean.processing.translation.source_language = LanguageCode.KO
    indexer = OpenSearchIndexer(config, embedding_factory=HashingEmbeddingFactory())
    others = [
        OpenSearchIndexer(c, embedding_factory=HashingEmbeddingFactory())
        for c in (with_ja, korean)
    ]
    try:
        assert indexer.upsert_text_units(_text_units()).failed_items == 0
        indexer.check_index_settings(_SUFFIX)
        for other, message in zip(
            others,
            ["'translated_text_ja' is not mapped", "'text' uses 'english'"],
            strict=True,
        ):
            with pytest.raises(ConfigurationError, match=message):
                other.check_index_settings(_SUFFIX)

        # A full build by a run with the registry off is not updated
        # incrementally; a pre-#231 index (no _meta) is given the dimension
        # it verifiably holds, not this run's model.
        indexer.set_doc_status_tracking(False)
        assert indexer.index_text_units(_text_units()).failed_items == 0
        with pytest.raises(ConfigurationError, match="registry off"):
            indexer.check_index_settings(_SUFFIX)
        client = indexer.opensearch_client
        index_name = client.get_index_name_by_alias(
            indexer._get_name(
                config.indexing.opensearch.text_units_index_prefix, _SUFFIX
            )
        )
        assert index_name
        client.put_index_meta(index_name, {})
        indexer.check_index_settings(_SUFFIX)
        assert client.get_index_mappings(index_name)["_meta"] == {
            "embedding_dimension": indexer._embedding_meta()["embedding_dimension"]
        }
    finally:
        indexer.clear([_SUFFIX])
        indexer.close()
        for other in others:
            other.close()


def test_query_of_a_never_ingested_suffix_raises_index_not_found(
    local_config: Config,
) -> None:
    # The real server's 404 body, through the real sync and async connection
    # classes, must reach the "No indices found for suffix" error rather than
    # a search that quietly returns nothing.
    config = local_config.model_copy(deep=True)
    config.indexing.additional_suffix = f"none{uuid.uuid4().hex[:8]}"
    prefix = config.indexing.opensearch.entities_index_prefix

    client = OpenSearchClient(config)
    try:
        with pytest.raises(NotFoundError) as sync_exc:
            client.client.search(index=f"{prefix}-nosuch-{uuid.uuid4().hex[:8]}")
    finally:
        client.close()
    assert is_index_not_found(sync_exc.value)

    with pytest.raises(IndexNotFoundError, match="No indices found for suffix"):
        asyncio.run(
            _vector_retrieve(
                config,
                SearchQuery(query="contract", index_prefixes=[prefix], suffix="nosuch"),
            )
        )


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


def test_local_mode_ingests_and_answers_through_the_registry(
    tmp_path: Path,
) -> None:
    # backend: local end to end on the real stores: the registry builds the
    # Gremlin/OpenSearch indexers and retrievers and the SQLite registry from
    # docker/config.local.yaml; the models are the local Ollama adapters
    # against a fake server (no model download).
    import httpx

    from tests.fixtures.fakes.model_servers import FakeOllama
    from tests.integration.test_ingestion_stages import _respond
    from unified_kg_rag.adapters.local.embeddings import OllamaEmbeddingFactory
    from unified_kg_rag.adapters.local.models import OllamaLanguageModelFactory
    from unified_kg_rag.adapters.local.sqlite_doc_status import SQLiteDocStatusStore
    from unified_kg_rag.application.bootstrap import build_providers
    from unified_kg_rag.application.ingestion import pipeline_stages as ps
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
    from unified_kg_rag.application.retrieval.rag_chain import (
        GraphRAGChain,
        RAGInput,
        RAGOutput,
    )
    from unified_kg_rag.domain.models import (
        Backend,
        PipelineConfig,
        PipelineStageStatus,
        PipelineStageType,
        SearchStrategy,
    )
    from unified_kg_rag.domain.models.config import ChunkingStrategy

    answer = "Depot stores widgets for Vendor."
    config = get_config(str(_LOCAL_CONFIG))
    assert config.backend is Backend.LOCAL
    config.indexing.additional_suffix = f"local{uuid.uuid4().hex[:8]}"
    config.providers.doc_status.sqlite_path = str(tmp_path / "doc-status.sqlite3")
    config.providers.rerank.kind = "none"
    chunking = config.processing.chunking
    chunking.chunker_type = ChunkingStrategy.SIMPLE
    chunking.fallback_chunk_size = 60
    chunking.chunk_overlap = 0
    chunking.min_chunk_size = 1
    config.processing.translation.enabled = False
    config.processing.gleaning.enabled = False
    config.processing.claim_extraction.enabled = False
    config.fixing.enabled = False
    config.graph.visualization.enabled = False
    server = FakeOllama(_respond, answer, config.providers.embedding.dimension or 64)
    transport = httpx.MockTransport(server)
    providers = build_providers(
        config,
        llm_factory=OllamaLanguageModelFactory(config, transport=transport),
        embedding_factory=OllamaEmbeddingFactory(config, transport=transport),
    )
    source = write_corpus(
        tmp_path / "corpus",
        {
            "contract.txt": "Vendor supplies widgets to Buyer.",
            "logistics.txt": "Depot stores widgets for Vendor.",
        },
    )
    pipeline = DataIngestionPipeline(
        config,
        PipelineConfig(
            stages_enabled=dict.fromkeys(PipelineStageType, True),
            local_directory=tmp_path / "cache",
        ),
        source_directory=source,
        providers=providers,
    )
    for stage in pipeline.stages:
        if isinstance(stage, ps.GraphResolutionStage):
            stage.resolver.entity_resolver.use_process_pool = False
            stage.resolver.relationship_resolver.use_process_pool = False
    chain = GraphRAGChain(config, providers=providers)
    try:
        context = pipeline.run(source, pipeline_id="local-mode")
        assert context.status is PipelineStageStatus.COMPLETED
        registry = providers.build_doc_status()
        assert isinstance(registry, SQLiteDocStatusStore)
        assert len(registry.list_all()) == 2

        out = chain.invoke(
            RAGInput(
                query="Who stores the widgets?",
                search_strategy=SearchStrategy.LOCAL,
                enable_query_processing=False,
            )
        )
        assert isinstance(out, RAGOutput)
        assert out.answer == answer
        assert out.sources
        assert server.answers == 1
        assert set(server.embed_models) == {config.providers.embedding.model_id}
    finally:
        chain.close()
        pipeline.close()
        graph = providers.build_graph_indexer()
        vectors = providers.build_vector_indexer()
        graph.clear([_SUFFIX])
        vectors.clear([_SUFFIX])
        graph.close()
        vectors.close()


def test_query_with_another_embedding_model_is_refused(local_config: Config) -> None:
    # An index built with one embedding model must not answer queries embedded
    # with another as a successful empty (or wrongly ranked) search.
    config = local_config.model_copy(deep=True)
    config.indexing.additional_suffix = f"em{uuid.uuid4().hex[:8]}"
    indexer = OpenSearchIndexer(config, embedding_factory=HashingEmbeddingFactory())
    query = SearchQuery(
        query="Vendor widgets",
        suffix=_SUFFIX,
        index_prefixes=[config.indexing.opensearch.text_units_index_prefix],
    )

    async def retrieve(query_config: Config, factory, *, skip_check: bool = False):
        retriever = OpenSearchRetriever(
            query_config,
            opensearch_client=OpenSearchClient(query_config),
            embedding_factory=factory,
        )
        if skip_check:  # As if the index recorded nothing: OpenSearch's 400.
            retriever._embedding_checked_aliases.add(
                retriever._get_name(query.index_prefixes[0], _SUFFIX)
            )
        try:
            return await retriever.aretrieve(query)
        finally:
            await retriever.aclose()

    try:
        assert indexer.index_text_units(_text_units()).failed_items == 0
        assert asyncio.run(retrieve(config, HashingEmbeddingFactory()))

        other_model = config.model_copy(deep=True)
        other_model.indexing.opensearch.embedding_model_id = "another-model"
        with pytest.raises(ConfigurationError, match="'another-model'"):
            asyncio.run(retrieve(other_model, HashingEmbeddingFactory()))

        with pytest.raises(ConfigurationError, match=r"dimension 64.*dimension 32"):
            asyncio.run(retrieve(config, HashingEmbeddingFactory(dimensions=32)))

        with pytest.raises(ConfigurationError, match="invalid dimension: 32"):
            asyncio.run(
                retrieve(
                    config, HashingEmbeddingFactory(dimensions=32), skip_check=True
                )
            )
    finally:
        indexer.clear([_SUFFIX])
        indexer.close()
