# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``backend: local`` end to end on the embedded stores: no container, no AWS.

The preset's SQLite graph, vector and registry stores (files in ``tmp_path``)
serve a full ingest, an incremental run that adds, changes and deletes a
document, a refused run whose graph lost the registry's documents, every
search strategy, the evaluators and ``run-visualization``. The models are the
real Ollama adapters against a fake server (``FakeOllama``) and the reranker
a fake cross-encoder; every boto3 client, Bedrock factory and DynamoDB store
constructor is patched to fail, so the run proves local mode needs neither
AWS nor a running service.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
import httpx
import pytest
import yaml

from tests.fixtures.doc_status_records import all_records
from tests.fixtures.extras import VIZ_INSTALLED
from tests.fixtures.fakes.model_servers import FakeOllama
from tests.fixtures.fakes.rerank import FakeCrossEncoder
from tests.fixtures.store_backends import (
    RETRIEVERS,
    StoreHarness,
    backends,
    open_harness,
)
from tests.integration import test_ingestion_stages as stages
from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.aws import token_counter as token_counter_module
from unified_kg_rag.adapters.doc_status import dynamodb as dynamodb_module
from unified_kg_rag.adapters.doc_status.sqlite import SQLiteDocStatusStore
from unified_kg_rag.adapters.local.embeddings import OllamaEmbeddingFactory
from unified_kg_rag.adapters.local.models import OllamaLanguageModelFactory
from unified_kg_rag.adapters.local.rerank import CrossEncoderRerankFactory
from unified_kg_rag.adapters.local.search_engine import SQLiteSearchEngine
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.storage.sqlite_graph_indexer import SQLiteGraphIndexer
from unified_kg_rag.application.bootstrap import build_providers
from unified_kg_rag.application.cli import run_visualization
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.ingestion.report_references import (
    report_cited_ids,
    report_cited_records,
)
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    EvaluationGroundTruth,
    EvaluationQuery,
    EvaluatorType,
    Methodology,
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
    SearchStrategy,
)
from unified_kg_rag.domain.models.config import ChunkingStrategy, LanguageCode
from unified_kg_rag.evaluation import EvaluationManager
from unified_kg_rag.shared.exceptions import (
    ConfigurationError,
    ModelUnreachableError,
    StoreRequestRejectedError,
)

pytestmark = pytest.mark.integration

DIM = 32
NAMES = ("Vendor", "Buyer", "Depot", "Carrier")
ANSWER = "Depot stores gadgets for Vendor."


def _respond(system: str, human: str) -> str:
    """The fake model's answers: ingestion as in the stage tests, plus the
    query-time and judge prompts. Anything else (the answer generation and
    the reduce steps) raises, so ``FakeOllama`` answers ``ANSWER``."""
    if "knowledge graph extraction expert" in system:
        return stages._extraction(stages._source_text(human), NAMES)
    if "search strategy selector" in system:
        return "local"
    if "specialized entity extraction expert" in system:
        return "Depot, Vendor"
    if "expert keyword extractor" in system:
        return json.dumps(
            {
                "high_level_keywords": ["widget storage"],
                "low_level_keywords": ["Depot", "Vendor"],
            }
        )
    if "KEY POINTS from the reports" in system:
        return json.dumps({"points": [{"description": ANSWER, "score": 90}]})
    if "DRIFT search primer" in system:
        return json.dumps(
            {
                "intermediate_answer": ANSWER,
                "score": 0.5,
                "follow_up_queries": ["What does Depot store for Vendor?"],
            }
        )
    if "You control the retrieval" in system:
        return json.dumps({"sufficient": True, "known_facts": [ANSWER]})
    if "You translate a question into a query plan" in system:
        # The structured strategy's planner: the entity the question names.
        return json.dumps(
            {
                "nodes": [{"var": "x", "mention": "Depot"}],
                "returns": {"kind": "rows"},
                "entities": ["Depot"],
            }
        )
    if "precision relevance evaluator" in system:
        return "9"
    if "search convergence analyst" in system:
        return "0.9"
    return stages._respond(system, human)


@pytest.fixture
def no_aws(mocker) -> None:
    def forbid(target: Any, name: str) -> None:
        mocker.patch.object(
            target, name, side_effect=AssertionError(f"{target}.{name} called")
        )

    forbid(bedrock_module.BaseBedrockModelFactory, "__init__")
    forbid(token_counter_module.BedrockTokenCounter, "__init__")
    forbid(dynamodb_module.DynamoDBDocStatusStore, "__init__")
    forbid(boto3.Session, "client")


def _config(stores: Path, viz: Path) -> Config:
    # Only the models and the files are set; the stores are the preset's.
    config = Config(
        backend="local",
        models={
            "default": "synthetic-llm",
            "fast": "synthetic-fast",
            "embedding": {"model_id": "synthetic-embed", "dimension": DIM},
            "rerank": "synthetic-reranker",
        },
        providers={
            "graph": {"sqlite_path": str(stores / "graph.sqlite3")},
            "vector": {"sqlite_path": str(stores / "vector.sqlite3")},
            "doc_status": {
                "enabled": True,
                "sqlite_path": str(stores / "doc-status.sqlite3"),
            },
        },
    )
    chunking = config.processing.chunking
    chunking.chunker_type = ChunkingStrategy.SIMPLE
    chunking.fallback_chunk_size = 60
    chunking.chunk_overlap = 0
    chunking.min_chunk_size = 1
    config.processing.translation.enabled = False
    config.processing.gleaning.enabled = False
    config.processing.claim_extraction.enabled = False
    config.retry.max_attempts = 1
    config.graph.visualization.outputs_directory = str(viz)
    return config


def _providers(config: Config, server: FakeOllama, encoder: FakeCrossEncoder):
    transport = httpx.MockTransport(server)
    return build_providers(
        config,
        llm_factory=OllamaLanguageModelFactory(config, transport=transport),
        embedding_factory=OllamaEmbeddingFactory(config, transport=transport),
        rerank_factory=CrossEncoderRerankFactory(
            config, loader=lambda *_, **__: encoder
        ),
    )


def _ingest(
    config: Config, providers: Providers, source: Path, cache: Path
) -> PipelineContext:
    pipeline = DataIngestionPipeline(
        config,
        PipelineConfig(
            stages_enabled=dict.fromkeys(PipelineStageType, True),
            local_directory=cache,
        ),
        source_directory=source,
        providers=providers,
    )
    for stage in pipeline.stages:
        if isinstance(stage, ps.GraphResolutionStage):
            stage.resolver.entity_resolver.use_process_pool = False
            stage.resolver.relationship_resolver.use_process_pool = False
    try:
        return pipeline.run(source, pipeline_id=cache.name)
    finally:
        pipeline.close()


def _errors(context: PipelineContext) -> str:
    return " ".join(r.error_message or "" for r in context.stage_results)


def _entity_names(providers: Providers) -> set[str]:
    graph = providers.build_graph_indexer()
    try:
        assert isinstance(graph, SQLiteGraphIndexer)
        return {name for _, name in graph.read_entity_names()}
    finally:
        graph.close()


def _stored(config: Config, providers: Providers, prefix: str) -> list[dict]:
    """Every document of the ``prefix`` index in the SQLite vector store."""
    vectors = providers.build_vector_indexer()
    try:
        client = vectors.opensearch_client
        assert isinstance(client, SQLiteSearchEngine)
        alias = vectors._get_name(prefix, None)
        return client.search(
            index=alias, body={"size": 100, "query": {"match_all": {}}}
        )["hits"]["hits"]
    finally:
        vectors.close()


def _stored_texts(config: Config, providers: Providers) -> set[str]:
    prefix = config.indexing.opensearch.text_units_index_prefix
    return {
        hit["_source"]["text"].strip() for hit in _stored(config, providers, prefix)
    }


def _stored_ids(config: Config, providers: Providers) -> set[str]:
    os_config = config.indexing.opensearch
    prefixes = (
        os_config.text_units_index_prefix,
        os_config.entities_index_prefix,
        os_config.relationships_index_prefix,
        os_config.community_reports_index_prefix,
    )
    return {
        hit["_id"] for prefix in prefixes for hit in _stored(config, providers, prefix)
    }


def _registry(providers: Providers) -> dict[str, DocStatus]:
    registry = providers.build_doc_status()
    assert isinstance(registry, SQLiteDocStatusStore)
    return {r.file_path: r.status for r in all_records(registry)}


def _ask(chain: GraphRAGChain, strategy: SearchStrategy) -> RAGOutput:
    out = chain.invoke(
        RAGInput(
            query="Who stores the gadgets for Vendor?",
            search_strategy=strategy,
        )
    )
    assert isinstance(out, RAGOutput), strategy
    return out


def _ask_every_strategy(
    chain: GraphRAGChain, server: FakeOllama, stored_ids: set[str]
) -> None:
    """Ask every search strategy: each answers from the stores, cites only
    ``stored_ids`` and reports the model usage the server served."""
    for strategy in SearchStrategy:
        answers_before = server.answers
        calls_before = len(server.chat_models)
        tokens_before = (server.input_tokens, server.output_tokens)
        out = _ask(chain, strategy)
        assert out.answer == ANSWER, strategy
        # The query's usage is exactly what the model server served.
        usage = out.metadata["usage"]
        assert usage["llm_calls"] == len(server.chat_models) - calls_before
        assert (usage["input_tokens"], usage["output_tokens"]) == (
            server.input_tokens - tokens_before[0],
            server.output_tokens - tokens_before[1],
        ), strategy
        # Rerank requests count under their own role, and are no calls.
        assert set(usage["by_role"]) - {"rerank"} == {"query"}, strategy
        assert usage["by_role"].get("rerank", {}).get("llm_calls", 0) == 0
        assert server.answers > answers_before, strategy
        assert out.search_results.results, strategy
        # Cited: the sources are stored artifacts (and the structured
        # strategy's query result, computed from them).
        cited = {
            source["source"]
            for source in out.sources
            if source["metadata"].get("section_type") != "query_result"
        }
        assert cited and cited <= stored_ids, strategy
        if strategy is SearchStrategy.STRUCTURED:
            assert out.metadata["structured"]["status"] == "answered"
            assert out.metadata["structured"]["linked"]["x"]
            # The query result is numbered like any evidence section, and
            # its citation names the documents its rows rest on.
            (table,) = [c for c in out.citations if c.section_type == "query_result"]
            assert table.documents, table
            assert table.text_unit_ids, table
            assert set(table.text_unit_ids) <= stored_ids
        # AUTO is routed by the model's answer ("local").
        expected = "local" if strategy is SearchStrategy.AUTO else strategy.value
        assert out.metadata["search_strategy"] == expected


def _render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: Path) -> int:
    """Run ``run-visualization`` on ``data``; its exit code."""
    config_path = tmp_path / "viz.yaml"
    config_path.write_text(
        yaml.safe_dump({"logging": {"log_to_file": False}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-visualization",
            "--data-path",
            str(data),
            "--output-directory",
            str(tmp_path / "rendered"),
            "--config-path",
            str(config_path),
        ],
    )
    return int(run_visualization.main())


def test_local_mode_runs_end_to_end_on_the_embedded_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_aws
) -> None:
    stores, viz = tmp_path / "stores", tmp_path / "viz-data"
    config = _config(stores, viz)
    # Every strategy is asked: raptor's tree too.
    config.indexing.methodologies = [*config.indexing.methodologies, Methodology.RAPTOR]
    assert (config.providers.graph.kind, config.providers.vector.kind) == (
        "sqlite",
        "sqlite",
    )
    server = FakeOllama(_respond, ANSWER, DIM)
    encoder = FakeCrossEncoder("synthetic-reranker")
    providers = _providers(config, server, encoder)
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")

    # Full ingest: every artifact lands in the SQLite files.
    context = _ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
    assert _entity_names(providers) == {"Vendor", "Buyer", "Depot"}
    assert _stored_texts(config, providers) == {
        "Vendor supplies widgets to Buyer.",
        "Depot stores widgets for Vendor.",
    }
    assert _registry(providers) == {
        "contract.txt": DocStatus.PROCESSED,
        "logistics.txt": DocStatus.PROCESSED,
    }
    assert {p.name for p in stores.iterdir()} >= {
        "graph.sqlite3",
        "vector.sqlite3",
        "doc-status.sqlite3",
    }

    # Incremental: one document added, one changed, one deleted.
    (source / "freight.txt").write_text("Carrier moves gadgets for Depot.")
    (source / "logistics.txt").write_text(ANSWER)
    (source / "contract.txt").unlink()
    context = _ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
    incremental_run = context.pipeline_id
    delta = context.incremental_delta
    assert delta is not None
    assert (len(delta.new), len(delta.changed), len(delta.deleted)) == (1, 1, 1)
    # Buyer appeared only in the deleted document; Vendor is still referenced.
    assert _entity_names(providers) == {"Vendor", "Depot", "Carrier"}
    texts = {ANSWER, "Carrier moves gadgets for Depot."}
    assert _stored_texts(config, providers) == texts
    registry = {
        "logistics.txt": DocStatus.PROCESSED,
        "freight.txt": DocStatus.PROCESSED,
    }
    assert _registry(providers) == registry

    # A graph that lost the registry's documents (another graph file here)
    # is refused before any store write: the vector store and the indexed
    # documents' records stay as they were, and the new document has only
    # the record of the attempt the run opened (removed once it is gone).
    (source / "extra.txt").write_text("Vendor ships gadgets.")
    lost = config.model_copy(deep=True)
    lost.providers.graph.sqlite_path = str(tmp_path / "empty-graph.sqlite3")
    lost_providers = _providers(lost, server, encoder)
    context = _ingest(lost, lost_providers, source, tmp_path / "run-3")
    assert context.status is PipelineStageStatus.FAILED
    assert "graph store does not hold" in _errors(context)
    assert _entity_names(lost_providers) == set()
    assert _stored_texts(config, providers) == texts
    assert _registry(providers) == {**registry, "extra.txt": DocStatus.PENDING}
    (source / "extra.txt").unlink()

    # Every strategy answers from the stores and cites what it used.
    stored_ids = _stored_ids(config, providers)
    chain = GraphRAGChain(config, providers=providers)
    try:
        _ask_every_strategy(chain, server, stored_ids)
        assert encoder.calls >= 1
        assert set(server.embed_models) == {"synthetic-embed"}

        # The default evaluators, with sources resolved to file names
        # through the SQLite vector store.
        config.evaluation.enabled_evaluators = [
            EvaluatorType.RETRIEVAL,
            EvaluatorType.ANSWER_MATCH,
            EvaluatorType.GRAPH_AWARE,
        ]
        config.evaluation.outputs_directory = str(tmp_path / "eval")
        manager = EvaluationManager(config, rag_chain=chain)
        queries = [
            EvaluationQuery(
                query_id="q1",
                question="Who stores the gadgets for Vendor?",
                metadata={"search_strategy": "local"},
            )
        ]
        truths = [
            EvaluationGroundTruth(
                query_id="q1",
                ground_truth=ANSWER,
                reference_sources=["logistics.txt"],
                expected_entities=["Depot", "Vendor"],
            )
        ]
        calls_before = len(server.chat_models)
        results, _, summary = asyncio.run(
            manager.evaluate_dataset(queries, truths, show_progress=False)
        )
    finally:
        chain.close()

    # Per-query usage on the result; the run's answers and judge calls apart.
    assert results[0].usage is not None and results[0].usage.llm_calls >= 1
    answering, scoring = summary.usage["answer_generation"], summary.usage["scoring"]
    assert answering == results[0].usage
    assert answering.llm_calls + scoring.llm_calls == (
        len(server.chat_models) - calls_before
    )

    # The manifest names the index the run was scored against: the last
    # successful ingestion (the refused run 3 wrote nothing).
    build = summary.run_manifest["index"]["default"]
    assert (build["pipeline_id"], build["mode"], build["documents"]) == (
        incremental_run,
        "incremental",
        2,
    )
    stats = summary.metric_statistics
    assert stats["hit_at_k"]["mean"] == 1.0
    assert stats["exact_match"]["mean"] == 1.0
    assert stats["entity_coverage"]["mean"] == 1.0
    cited = {name for rank in results[0].retrieved_source_ids for name in rank}
    assert cited == {"logistics.txt", "freight.txt"}

    # run-visualization renders the exported graph; on
    # an install without the viz extra it stops with exit 2 instead.
    data = viz / "visualization_data.json"
    assert data.is_file()
    if VIZ_INSTALLED:
        assert _render(tmp_path, monkeypatch, data) == 0
        assert sorted(p.name for p in (tmp_path / "rendered").iterdir())
    else:
        assert _render(tmp_path, monkeypatch, data) == 2


@pytest.fixture(params=backends(RETRIEVERS))
def store_harness(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[StoreHarness]:
    yield from open_harness(request.param, tmp_path / "harness")


def _on_harness_stores(config: Config, harness: StoreHarness) -> None:
    """Point ``config`` at the stores (and the namespace) of ``harness``,
    whose teardown then clears what the run wrote."""
    stores = harness.config
    for name in ("graph", "vector"):
        mine, theirs = getattr(config.providers, name), getattr(stores.providers, name)
        mine.kind = theirs.kind
        mine.sqlite_path = theirs.sqlite_path
    for name in ("neptune", "opensearch"):
        mine, theirs = getattr(config.aws, name), getattr(stores.aws, name)
        mine.endpoint, mine.port = theirs.endpoint, theirs.port
    config.indexing.additional_suffix = stores.indexing.additional_suffix


def test_ingest_update_and_every_strategy_on_each_store_backend(
    tmp_path: Path, no_aws, store_harness: StoreHarness
) -> None:
    # The embedded run above, on each store backend with retrievers: the
    # SQLite files and (LOCAL_STORES=1) Gremlin Server and OpenSearch. A
    # full ingest, an incremental run that adds, changes and deletes a
    # document, then every strategy over what it left.
    harness = store_harness
    config = _config(tmp_path / "stores", tmp_path / "viz-data")
    config.indexing.methodologies = [*config.indexing.methodologies, Methodology.RAPTOR]
    _on_harness_stores(config, harness)
    server = FakeOllama(_respond, ANSWER, DIM)
    encoder = FakeCrossEncoder("synthetic-reranker")
    providers = _providers(config, server, encoder)
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")

    def entity_names() -> set[str]:
        graph = providers.build_graph_indexer()
        try:
            return {name for _, name in graph.read_entity_names()}
        finally:
            graph.close()

    def texts() -> set[str]:
        vectors = providers.build_vector_indexer()
        try:
            rows = vectors.scan_text_units(None, fields=["text"])
        finally:
            vectors.close()
        return {row["text"].strip() for row in rows}

    context = _ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
    assert entity_names() == {"Vendor", "Buyer", "Depot"}
    assert texts() == {
        "Vendor supplies widgets to Buyer.",
        "Depot stores widgets for Vendor.",
    }

    (source / "freight.txt").write_text("Carrier moves gadgets for Depot.")
    (source / "logistics.txt").write_text(ANSWER)
    (source / "contract.txt").unlink()
    context = _ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
    delta = context.incremental_delta
    assert delta is not None
    assert (len(delta.new), len(delta.changed), len(delta.deleted)) == (1, 1, 1)
    assert entity_names() == {"Vendor", "Depot", "Carrier"}
    assert texts() == {ANSWER, "Carrier moves gadgets for Depot."}
    assert _registry(providers) == {
        "logistics.txt": DocStatus.PROCESSED,
        "freight.txt": DocStatus.PROCESSED,
    }
    # The run wrote to the harness's stores: they hold its entities, and
    # the stored ids the answers may cite.
    assert len(harness.stored_ids("graph.entities")) == 3
    stored_ids = set().union(
        *(
            harness.stored_ids(kind)
            for kind in ("text_units", "entities", "relationships", "community_reports")
        )
    )

    chain = GraphRAGChain(config, providers=providers)
    try:
        _ask_every_strategy(chain, server, stored_ids)
    finally:
        chain.close()
    assert encoder.calls >= 1


def _refusing(monkeypatch: pytest.MonkeyPatch, refused: Any) -> None:
    """Have the embedded engine refuse an index whose body ``refused``
    matches, as OpenSearch refuses a setting, vector method or analyzer it
    does not know (the embedded engine ignores all but max_result_window)."""
    create = SQLiteSearchEngine.create_index

    def strict(self, index, body, *, exist_ok=True):
        if reason := refused(body):
            raise StoreRequestRejectedError(reason, reason="illegal_argument_exception")
        return create(self, index, body, exist_ok=exist_ok)

    monkeypatch.setattr(SQLiteSearchEngine, "create_index", strict)


def _fields(body: dict[str, Any]) -> list[dict[str, Any]]:
    return list(body.get("mappings", {}).get("properties", {}).values())


def _settings_typo(config: Config) -> None:
    config.indexing.opensearch.index_settings = {"refresh_intervl": "1s"}


def _refused_settings(body: dict[str, Any]) -> str | None:
    if "refresh_intervl" in body["settings"]:
        return "unknown setting [index.refresh_intervl]"
    return None


def _vector_method(config: Config) -> None:
    config.indexing.opensearch.vector_search = {
        **config.indexing.opensearch.vector_search,
        "m": 0,
    }


def _refused_method(body: dict[str, Any]) -> str | None:
    for field in _fields(body):
        if field.get("type") == "knn_vector" and not field["method"]["parameters"]["m"]:
            return 'Parameter validation failed for Integer parameter "m"'
    return None


def _missing_analyzer(config: Config) -> None:
    config.indexing.opensearch.language_analyzers = {"en": "no_such_analyzer"}


def _refused_analyzer(body: dict[str, Any]) -> str | None:
    if any(field.get("analyzer") == "no_such_analyzer" for field in _fields(body)):
        return "analyzer [no_such_analyzer] has not been configured in mappings"
    return None


@pytest.mark.parametrize(
    ("change", "refused", "named"),
    [
        (_settings_typo, _refused_settings, "refresh_intervl"),
        # The probe index had no fields, so these passed it.
        (_vector_method, _refused_method, '"m"'),
        (_missing_analyzer, _refused_analyzer, "no_such_analyzer"),
    ],
    ids=["index-setting", "vector-method", "analyzer"],
)
def test_a_reset_with_a_refused_index_fails_before_any_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_aws, change, refused, named
) -> None:
    # Only a missing extraction-records index was created before extraction,
    # so a reset (or an incremental run) over a built namespace extracted and
    # summarized the corpus again before the indexing stage's first index
    # creation was refused.
    config = _config(tmp_path / "stores", tmp_path / "viz-data")
    server = FakeOllama(_respond, ANSWER, DIM)
    encoder = FakeCrossEncoder("synthetic-reranker")
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    context = _ingest(
        config, _providers(config, server, encoder), source, tmp_path / "run-1"
    )
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)

    _refusing(monkeypatch, refused)
    # A new document: extracting it would be a model call.
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    typo = config.model_copy(deep=True)
    typo.indexing.reset = True
    change(typo)
    calls = len(server.chat_models), len(server.embed_models)
    with pytest.raises(StoreRequestRejectedError, match=named):
        _ingest(typo, _providers(typo, server, encoder), source, tmp_path / "run-2")

    assert (len(server.chat_models), len(server.embed_models)) == calls


def test_an_incremental_run_with_nothing_to_extract_is_not_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_aws
) -> None:
    # A deletion-only or unchanged delta creates no index: its deletions go
    # to the live ones, so a setting the engine would refuse does not stop it.
    config = _config(tmp_path / "stores", tmp_path / "viz-data")
    server = FakeOllama(_respond, ANSWER, DIM)
    encoder = FakeCrossEncoder("synthetic-reranker")
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    context = _ingest(
        config, _providers(config, server, encoder), source, tmp_path / "run-1"
    )
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)

    _refusing(monkeypatch, _refused_settings)
    (source / "logistics.txt").unlink()
    typo = config.model_copy(deep=True)
    _settings_typo(typo)
    context = _ingest(
        typo, _providers(typo, server, encoder), source, tmp_path / "run-2"
    )
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)


def test_a_refused_index_setting_fails_before_translation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_aws
) -> None:
    # The settings were checked when graph_extraction prepared its records
    # store, after translation had translated every text unit (a run the
    # user then reran without --pipeline-id paid for them again). The run
    # now checks them before its first stage that may call a model.
    config = _config(tmp_path / "stores", tmp_path / "viz-data")
    config.processing.translation.enabled = True
    config.processing.translation.source_language = LanguageCode.KO
    config.processing.translation.target_language = LanguageCode.EN
    _settings_typo(config)
    server = FakeOllama(_respond, ANSWER, DIM)
    encoder = FakeCrossEncoder("synthetic-reranker")
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    _refusing(monkeypatch, _refused_settings)

    with pytest.raises(StoreRequestRejectedError, match="refresh_intervl"):
        _ingest(config, _providers(config, server, encoder), source, tmp_path / "run")

    assert server.chat_models == []
    assert server.embed_models == []


def _translate_to_korean(config: Config) -> None:
    config.processing.translation.enabled = True
    config.processing.translation.source_language = LanguageCode.EN
    config.processing.translation.target_language = LanguageCode.KO


def _other_embedding_dimension(config: Config) -> None:
    config.models.embedding.dimension = DIM * 2


@pytest.mark.parametrize(
    ("change", "named"),
    [
        (_translate_to_korean, "other language settings"),
        (_other_embedding_dimension, "holds embeddings of"),
    ],
    ids=["translation", "embedding"],
)
def test_an_incremental_run_over_other_live_indices_fails_before_any_model_call(
    tmp_path: Path, no_aws, change, named: str
) -> None:
    # The live indices' settings were compared with the run's in the
    # indexing stage only: an incremental run with another translation,
    # analyzer or embedding setting re-translated and re-extracted the whole
    # corpus first. The reset the error recommends then pays for it once.
    config = _config(tmp_path / "stores", tmp_path / "viz-data")
    server = FakeOllama(_respond, ANSWER, DIM)
    encoder = FakeCrossEncoder("synthetic-reranker")
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    context = _ingest(
        config, _providers(config, server, encoder), source, tmp_path / "run-1"
    )
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)

    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    changed = config.model_copy(deep=True)
    change(changed)
    if changed.models.embedding.dimension != DIM:
        server = FakeOllama(_respond, ANSWER, changed.models.embedding.dimension)
    calls = len(server.chat_models), len(server.embed_models)
    with pytest.raises(ConfigurationError, match=named):
        _ingest(
            changed, _providers(changed, server, encoder), source, tmp_path / "run-2"
        )
    assert (len(server.chat_models), len(server.embed_models)) == calls

    changed.indexing.reset = True
    context = _ingest(
        changed, _providers(changed, server, encoder), source, tmp_path / "run-3"
    )
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)


def test_an_incremental_run_at_the_shard_cap_writes_into_its_live_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_aws
) -> None:
    # The run always created a probe index, so a cluster with no room for
    # another index refused an incremental run that creates none.
    config = _config(tmp_path / "stores", tmp_path / "viz-data")
    server = FakeOllama(_respond, ANSWER, DIM)
    encoder = FakeCrossEncoder("synthetic-reranker")
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    built = config.model_copy(deep=True)
    built.indexing.reset = True  # a full build: every index exists
    context = _ingest(
        built, _providers(built, server, encoder), source, tmp_path / "run-1"
    )
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)

    _refusing(
        monkeypatch,
        lambda _: "validation_exception: this action would add [1] total shards, "
        "but this cluster currently has [9]/[9] maximum shards open;",
    )
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    context = _ingest(
        config, _providers(config, server, encoder), source, tmp_path / "run-2"
    )
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)

    calls = len(server.chat_models)
    with pytest.raises(StoreRequestRejectedError, match="max_shards_per_node"):
        _ingest(built, _providers(built, server, encoder), source, tmp_path / "run-3")
    assert len(server.chat_models) == calls


def test_a_global_answers_report_sources_name_the_records_their_findings_cite(
    tmp_path: Path, no_aws
) -> None:
    # The findings' [Data: ...] references were resolved to record ids at
    # indexing and then dropped: a report source could not say which
    # entities and relationships its findings rest on.
    config = _config(tmp_path / "stores", tmp_path / "viz-data")

    def respond(system: str, human: str) -> str:
        if "community analysis" in system:
            return stages._REPORT.replace(
                "Vendor works with Buyer and Depot.",
                "Vendor works with Buyer and Depot "
                "[Data: Entities (1, 2); Relationships (1, +more)].",
            )
        return _respond(system, human)

    server = FakeOllama(respond, ANSWER, DIM)
    providers = _providers(config, server, FakeCrossEncoder("synthetic-reranker"))
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    context = _ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)

    os_config = config.indexing.opensearch
    stored = {
        prefix: {hit["_id"] for hit in _stored(config, providers, prefix)}
        for prefix in (
            os_config.entities_index_prefix,
            os_config.relationships_index_prefix,
        )
    }
    # Map even the one report, so the answer reads its key point.
    config.search.global_search.map_reduce_min_results = 1
    chain = GraphRAGChain(config, providers=providers)
    try:
        out = _ask(chain, SearchStrategy.GLOBAL)
    finally:
        chain.close()
    reports = [
        source["metadata"]
        for source in out.sources
        if source["metadata"].get("section_type") == "community"
    ]
    assert reports, out.sources
    for metadata in reports:
        entity_ids, relationship_ids = report_cited_ids(metadata)
        assert len(entity_ids) == 2 and len(relationship_ids) == 1, metadata
        assert set(entity_ids) <= stored[os_config.entities_index_prefix]
        assert set(relationship_ids) <= stored[os_config.relationships_index_prefix]
        # The references left the searched text; the source keeps each
        # finding's ids, not its text again.
        assert "[Data:" not in metadata["full_content"]
        assert all(
            set(finding) <= {"entity_ids", "relationship_ids"}
            for finding in metadata["findings"]
        )
    # Each key point cites the text units of the records its reports'
    # findings cite (read from the stores), not the reports' communities.
    units = {
        hit["_id"]: hit["_source"]["text_unit_ids"]
        for prefix in stored
        for hit in _stored(config, providers, prefix)
    }
    points = [
        r.metadata for r in out.search_results.results if r.metadata.get("synthesized")
    ]
    assert points, out.search_results.results
    for point in points:
        cited = {
            unit
            for metadata in reports
            if str(metadata.get("id")) in point["report_ids"]
            for record in report_cited_records(metadata)
            for unit in units.get(record[1], [])
        }
        assert cited and set(point["text_unit_ids"]) == cited, point


def test_a_key_point_names_only_the_documents_its_findings_cite(
    tmp_path: Path, no_aws
) -> None:
    # rv5 M1: a key point named every document of its report's community
    # (membership). Its finding cites one relationship, stated by one of
    # three documents (one in Korean, one in Japanese): the point names it.
    config = _config(tmp_path / "stores", tmp_path / "viz-data")
    config.search.global_search.map_reduce_min_results = 1

    def respond(system: str, human: str) -> str:
        if "community analysis" in system:
            return stages._REPORT.replace(
                "Vendor works with Buyer and Depot.",
                "Vendor works with Buyer and Depot [Data: Relationships (1)].",
            )
        return _respond(system, human)

    server = FakeOllama(respond, ANSWER, DIM)
    providers = _providers(config, server, FakeCrossEncoder("synthetic-reranker"))
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.ko.txt").write_text("Vendor supplies widgets to Buyer. 공급사.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    (source / "freight.ja.txt").write_text("Carrier moves gadgets for Depot. 運送業者.")
    context = _ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)

    os_config = config.indexing.opensearch
    relationships = {
        hit["_id"]: hit["_source"]["text_unit_ids"]
        for hit in _stored(config, providers, os_config.relationships_index_prefix)
    }
    chain = GraphRAGChain(config, providers=providers)
    try:
        out = _ask(chain, SearchStrategy.GLOBAL)
    finally:
        chain.close()
    reports = {
        str(r.source): r.metadata
        for r in out.search_results.results
        if r.retriever_type == "community"
    }
    points = [
        r.metadata for r in out.search_results.results if r.metadata.get("synthesized")
    ]
    assert points and reports
    for point in points:
        credited = [reports[i] for i in point["report_ids"] if i in reports]
        membership = {u for m in credited for u in m["text_unit_ids"]}
        (cited,) = {r for m in credited for _, r in report_cited_records(m)}
        assert point["text_unit_ids"] == relationships[cited]
        assert set(point["text_unit_ids"]) < membership


def test_a_query_without_a_suffix_reads_the_configs_index_value(
    tmp_path: Path, no_aws
) -> None:
    # Ingestion wrote the namespace of processing.document_parsing.index_value;
    # a query without a suffix (the console's Ask page sends none) read
    # 'default' and failed with IndexNotFoundError after a successful ingest.
    config = _config(tmp_path / "stores", tmp_path / "viz-data")
    config.processing.document_parsing.index_value = "tenant"
    server = FakeOllama(_respond, ANSWER, DIM)
    providers = _providers(config, server, FakeCrossEncoder("synthetic-reranker"))
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "logistics.txt").write_text("Depot stores gadgets for Vendor.")
    context = _ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)

    chain = GraphRAGChain(config, providers=providers)
    try:
        record = asyncio.run(chain.aread_build_record(None))
        out = _ask(chain, SearchStrategy.LOCAL)
    finally:
        chain.close()
    assert record is not None
    assert out.answer == ANSWER and out.sources, out.metadata
    assert not out.metadata.get("error"), out.metadata


def _down(request: httpx.Request) -> httpx.Response:
    """A model server that is not running."""
    raise httpx.ConnectError("All connection attempts failed", request=request)


def _down_providers(config: Config) -> Providers:
    """The providers of ``_providers`` with every model call refused."""
    transport = httpx.MockTransport(_down)
    return build_providers(
        config,
        llm_factory=OllamaLanguageModelFactory(config, transport=transport),
        embedding_factory=OllamaEmbeddingFactory(config, transport=transport),
        rerank_factory=CrossEncoderRerankFactory(
            config, loader=lambda *_, **__: FakeCrossEncoder("synthetic-reranker")
        ),
    )


def test_an_unreachable_model_server_fails_ingestion_before_any_extraction(
    tmp_path: Path, no_aws
) -> None:
    # Regression (rv7 features M1): the embedding probe swallowed the
    # transport error, the stages retried for minutes, and the run ended
    # with "processed 2 inputs but produced 0 outputs".
    config = _config(tmp_path / "stores", tmp_path / "viz-data")
    providers = _down_providers(config)
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "logistics.txt").write_text("Depot stores gadgets for Vendor.")
    with pytest.raises(ModelUnreachableError) as raised:
        _ingest(config, providers, source, tmp_path / "run-1")
    message = str(raised.value)
    assert (
        "providers.llm and providers.embedding (ollama at http://localhost:11434)"
        in message
    )
    assert "ollama serve" in message and "synthetic-embed" in message
    # Before any extraction: no stage ran.
    assert not (tmp_path / "stores" / "graph.sqlite3").exists()


def test_an_unreachable_model_server_names_itself_at_query_time(
    tmp_path: Path, no_aws
) -> None:
    config = _config(tmp_path / "stores", tmp_path / "viz-data")
    server = FakeOllama(_respond, ANSWER, DIM)
    providers = _providers(config, server, FakeCrossEncoder("synthetic-reranker"))
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "logistics.txt").write_text("Depot stores gadgets for Vendor.")
    context = _ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)

    down = _down_providers(config)
    chain = GraphRAGChain(config, providers=down)
    try:
        with pytest.raises(ModelUnreachableError, match="ollama serve"):
            _ask(chain, SearchStrategy.LOCAL)
    finally:
        chain.close()
    config.processing.ignore_errors = True
    chain = GraphRAGChain(config, providers=down)
    try:
        out = _ask(chain, SearchStrategy.LOCAL)
    finally:
        chain.close()
    assert out.metadata["error"] is True
    assert out.metadata["error_type"] == "ModelUnreachableError"
    assert out.metadata["error_class"] == "model_unreachable"
