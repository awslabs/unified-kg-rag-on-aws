# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gremlin Server and OpenSearch specifics, against the local stores.

The store contracts every backend shares run in the conformance suites
(``test_*_conformance.py``, ``test_store_parity.py``); this module keeps what
only the servers of ``docker/compose.local.yaml`` show: the Gremlin Server's
id-property index, writes and admin requests resent after a lost response, the
real 404 body, and two pipelines on the real stores. Hashing embeddings stand
in for Bedrock, so no AWS call is made. Skipped unless ``LOCAL_STORES=1``::

    docker compose -f docker/compose.local.yaml up -d --wait
    LOCAL_STORES=1 uv run pytest tests/integration/test_local_stores.py -v
    docker compose -f docker/compose.local.yaml down -v

``LOCAL_STORES_GREMLIN_PORT`` / ``LOCAL_STORES_OPENSEARCH_PORT`` point the tests
at a stack published on other ports. Every run writes under its own
``indexing.additional_suffix`` and clears it afterwards. The incremental
pipeline test keeps the doc-status registry in memory (no DynamoDB). The
local-mode test runs ``backend: local`` end to end (registry-built stores,
SQLite registry, the Ollama adapters against a fake server), so it needs no
Ollama container and downloads no model.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from gremlin_python.driver.client import Client
from opensearchpy.exceptions import NotFoundError

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.store_backends import local_store_config
from tests.integration.test_incremental_source_scopes import (
    A_TEXT,
    B_TEXT,
    ScopeStack,
    scope_test_config,
    write_corpus,
)
from unified_kg_rag.adapters.aws.opensearch import OpenSearchClient, is_index_not_found
from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.domain.models import (
    Community,
    Config,
    Entity,
    Relationship,
)
from unified_kg_rag.domain.models.config import LanguageCode

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("LOCAL_STORES") != "1",
        reason="set LOCAL_STORES=1 with docker/compose.local.yaml running",
    ),
]

_SUFFIX = "default"


@pytest.fixture(scope="module")
def local_config() -> Config:
    config = local_store_config()
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


@pytest.fixture(scope="module")
def graph_indexer(local_config: Config) -> Iterator[NeptuneIndexer]:
    indexer = NeptuneIndexer(local_config)
    try:
        yield indexer
    finally:
        indexer.clear([_SUFFIX])
        indexer.close()


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


def test_graph_writes_retried_after_a_lost_response_add_nothing_twice(
    graph_indexer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every request is applied, then its response is lost once, so each one is
    # retried (over a new connection) after it took effect. Regression: the
    # full build's addV then wrote every vertex twice, reporting 0 failed.
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

    def counts() -> tuple[int, int, int]:
        g = graph_indexer.neptune_client.g
        entity = graph_indexer._get_label("Entity", _SUFFIX)
        community = graph_indexer._get_label("Community", _SUFFIX)
        return (
            int(g.V().hasLabel(entity).count().next()),
            int(g.V().hasLabel(entity).outE().count().next()),
            int(g.V().hasLabel(community).count().next()),
        )

    assert graph_indexer.index_entities(_entities()).failed_items == 0
    assert graph_indexer.index_relationships(_relationships()).failed_items == 0
    assert graph_indexer.index_communities(_communities()).failed_items == 0
    assert counts() == (3, 2, 1)
    (vendor,) = graph_indexer.read_entities(["e-vendor"], _SUFFIX)
    assert sorted(vendor.text_unit_ids or []) == ["t1", "t2"]

    assert graph_indexer.upsert_entities(_entities()).failed_items == 0
    assert graph_indexer.upsert_relationships(_relationships()).failed_items == 0
    assert graph_indexer.upsert_communities(_communities()).failed_items == 0
    assert counts() == (3, 2, 1)

    assert graph_indexer.delete_by_id(["e-depot"], _SUFFIX).failed_items == 0
    assert counts() == (2, 1, 1)


def _stored_edges(graph_indexer: NeptuneIndexer) -> list[tuple[str, str, str]]:
    g = graph_indexer.neptune_client.g
    entity = graph_indexer._get_label("Entity", _SUFFIX)
    from gremlin_python.process.graph_traversal import __

    rows = (
        g.V()
        .hasLabel(entity)
        .outE()
        .project("id", "source", "target")
        .by("id")
        .by(__.outV().values("id"))
        .by(__.inV().values("id"))
        .toList()
    )
    return sorted((r["id"], r["source"], r["target"]) for r in rows)


def test_an_interrupted_edge_rewrite_keeps_every_stored_edge(
    graph_indexer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: the stored edges of a batch were dropped by one request and
    # re-added one request each, so a run killed (or stopped, or whose lease
    # was lost) after the first re-add lost the others for good.
    from unified_kg_rag.adapters.storage import neptune_indexer as module

    graph_indexer.clear([_SUFFIX])
    graph_indexer.index_entities(_entities())
    graph_indexer.index_relationships(_relationships())
    before = _stored_edges(graph_indexer)
    assert len(before) == 2

    class _Killed(BaseException):
        pass

    real = module.NeptuneIndexer._execute_with_retries
    writes = {"n": 0}

    def killed_after_one_write(self, build, operation_name, **kwargs):  # type: ignore[no-untyped-def]
        if operation_name == "Relationship indexing":
            writes["n"] += 1
            if writes["n"] == 2:
                raise _Killed
        return real(self, build, operation_name, **kwargs)

    monkeypatch.setattr(
        module.NeptuneIndexer, "_execute_with_retries", killed_after_one_write
    )
    monkeypatch.setattr(graph_indexer.neptune_config, "index_concurrency", 1)
    with pytest.raises(_Killed):
        graph_indexer.upsert_relationships(
            [r.model_copy(update={"rank": 7}) for r in _relationships()]
        )
    assert _stored_edges(graph_indexer) == before


def test_an_edge_rewrite_replaces_its_stored_copy_once(graph_indexer) -> None:
    # The drop of the stored copy and the add of the new one are one request:
    # re-pointed or resent, the edge is stored once; with an endpoint gone,
    # nothing is dropped.
    from unified_kg_rag.adapters.storage.neptune_indexer import run_traversal

    graph_indexer.clear([_SUFFIX])
    graph_indexer.index_entities(_entities())
    graph_indexer.index_relationships(_relationships()[:1])
    label = graph_indexer._get_label("Entity", _SUFFIX)
    supplies = _relationships()[0]

    def rewrite(rel: Relationship) -> list:
        return run_traversal(
            graph_indexer.neptune_client,
            lambda g: graph_indexer._build_add_edge_traversal(g, rel, label),
            results=True,
        )

    repointed = supplies.model_copy(update={"target_id": "e-depot"})
    assert rewrite(repointed)
    assert rewrite(repointed)  # a resend after a lost response
    assert _stored_edges(graph_indexer) == [("r-supplies", "e-vendor", "e-depot")]

    assert rewrite(supplies.model_copy(update={"target_id": "gone"})) == []
    assert _stored_edges(graph_indexer) == [("r-supplies", "e-vendor", "e-depot")]


def test_a_rebuild_in_place_on_gremlin_sweeps_only_what_it_did_not_write(
    graph_indexer,
) -> None:
    graph_indexer.clear([_SUFFIX])
    graph_indexer.index_entities(_entities())
    graph_indexer.index_relationships(_relationships())
    graph_indexer.index_communities(_communities())

    kept = [e for e in _entities() if e.id != "e-depot"]
    assert graph_indexer.begin_rebuild([_SUFFIX])
    graph_indexer.index_entities(kept, suffixes=[_SUFFIX])
    # The previous graph is still served while the build writes.
    assert len(graph_indexer.read_entities(["e-depot"], _SUFFIX)) == 1
    graph_indexer.index_relationships(_relationships()[:1])
    graph_indexer.index_communities(
        [_communities()[0].model_copy(update={"entity_ids": ["e-vendor"]})],
        suffixes=[_SUFFIX],
    )
    graph_indexer.end_rebuild([_SUFFIX])

    g = graph_indexer.neptune_client.g
    community = graph_indexer._get_label("Community", _SUFFIX)
    assert {
        e.id
        for e in graph_indexer.read_entities(
            ["e-vendor", "e-buyer", "e-depot"], _SUFFIX
        )
    } == {"e-vendor", "e-buyer"}
    assert [
        r.id
        for r in graph_indexer.read_relationships(["r-supplies", "r-ships"], _SUFFIX)
    ] == ["r-supplies"]
    assert int(g.V().hasLabel(community).count().next()) == 1


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


def test_a_missing_index_is_recognized_from_the_real_404_body(
    local_config: Config,
) -> None:
    # The real server's 404 body, through the real connection class, must be
    # recognized as a missing index (the retriever then reports a suffix that
    # was never ingested; see test_retrieval_conformance.py).
    prefix = local_config.indexing.opensearch.entities_index_prefix
    client = OpenSearchClient(local_config)
    try:
        with pytest.raises(NotFoundError) as sync_exc:
            client.client.search(index=f"{prefix}-nosuch-{uuid.uuid4().hex[:8]}")
    finally:
        client.close()
    assert is_index_not_found(sync_exc.value)


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
    from unified_kg_rag.adapters.doc_status.sqlite import SQLiteDocStatusStore
    from unified_kg_rag.adapters.local.embeddings import OllamaEmbeddingFactory
    from unified_kg_rag.adapters.local.models import OllamaLanguageModelFactory
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
    config = local_store_config()
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
    # The Ollama embedding model's dimension is the configured one.
    config.models.embedding.dimension = 64
    server = FakeOllama(_respond, answer, config.models.embedding.dimension)
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
        assert set(server.embed_models) == {config.embedding_model_id}
    finally:
        chain.close()
        pipeline.close()
        graph = providers.build_graph_indexer()
        vectors = providers.build_vector_indexer()
        graph.clear([_SUFFIX])
        vectors.clear([_SUFFIX])
        graph.close()
        vectors.close()


def test_default_lucene_mapping_takes_vectors_wider_than_1024(
    local_config: Config,
) -> None:
    # The default vector mapping (lucene HNSW, cosinesimil) on OpenSearch
    # 2.13 accepts up to 16,000 dimensions (2.12+; 1,024 before), so a
    # 3,072-dimension model needs no engine switch.
    dimension = 3072
    mapping = OpenSearchIndexer._get_knn_vector_mapping(
        SimpleNamespace(  # type: ignore[arg-type]
            opensearch_config=local_config.indexing.opensearch,
            _embedding_dimension=dimension,
        )
    )
    assert mapping["method"]["engine"] == "lucene"
    assert mapping["method"]["space_type"] == "cosinesimil"
    client = OpenSearchClient(local_config)
    name = f"wide-{uuid.uuid4().hex[:8]}"
    vector = [1.0 / dimension] * dimension
    try:
        client.client.indices.create(
            index=name,
            body={
                "settings": {"index": {"knn": True}},
                "mappings": {"properties": {"embedding": mapping}},
            },
        )
        client.client.index(
            index=name, id="v1", body={"embedding": vector}, refresh=True
        )
        hits = client.client.search(
            index=name,
            body={
                "size": 1,
                "query": {"knn": {"embedding": {"vector": vector, "k": 1}}},
            },
        )["hits"]["hits"]
        assert [hit["_id"] for hit in hits] == ["v1"]
    finally:
        client.client.indices.delete(index=name, ignore_unavailable=True)
        client.close()
