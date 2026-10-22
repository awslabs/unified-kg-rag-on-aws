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
from pathlib import Path
from typing import Any

import boto3
import httpx
import pytest
import yaml

from tests.fixtures.fakes.model_servers import FakeOllama
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
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    EvaluationGroundTruth,
    EvaluationQuery,
    EvaluatorType,
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
    SearchStrategy,
)
from unified_kg_rag.domain.models.config import ChunkingStrategy, LanguageCode
from unified_kg_rag.evaluation import EvaluationManager
from unified_kg_rag.shared.exceptions import StoreRequestRejectedError

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
    if "precision relevance evaluator" in system:
        return "9"
    if "search convergence analyst" in system:
        return "0.9"
    # LangChain's judges send a single human message.
    if "You are assessing a submitted answer" in human:
        return "The submission matches the reference.\nY"
    if "assessing the correctness of a" in human:
        return json.dumps({"score": 1.0, "reasoning": "Matches the reference."})
    return stages._respond(system, human)


class FakeCrossEncoder:
    def __init__(self, model_id: str, **_: Any) -> None:
        self.model_id = model_id
        self.calls = 0

    def predict(self, pairs: list[tuple[str, str]], **_: Any) -> list[float]:
        self.calls += 1
        return [float("Depot" in passage) for _, passage in pairs]


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
    config.processing.max_attempts = 1
    config.fixing.enabled = False
    config.graph.visualization.enabled = False
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
    return {r.file_path: r.status for r in registry.list_all()}


def _ask(chain: GraphRAGChain, strategy: SearchStrategy) -> RAGOutput:
    out = chain.invoke(
        RAGInput(
            query="Who stores the gadgets for Vendor?",
            search_strategy=strategy,
        )
    )
    assert isinstance(out, RAGOutput), strategy
    return out


def _render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: Path, method: str
) -> list[Path]:
    config_path = tmp_path / f"viz-{method}.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "graph": {"visualization": {"embedding_method": method}},
                "logging": {"log_to_file": False},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / f"rendered-{method}"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-visualization",
            "--data-path",
            str(data),
            "--output-directory",
            str(output),
            "--config-path",
            str(config_path),
            "--backend",
            "local",
        ],
    )
    assert run_visualization.main() == 0
    return sorted(output.iterdir())


def test_local_mode_runs_end_to_end_on_the_embedded_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_aws
) -> None:
    stores, viz = tmp_path / "stores", tmp_path / "viz-data"
    config = _config(stores, viz)
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
        for strategy in SearchStrategy:
            answers_before = server.answers
            out = _ask(chain, strategy)
            assert out.answer == ANSWER, strategy
            assert server.answers > answers_before, strategy
            assert out.search_results.results, strategy
            # Cited: the sources are stored artifacts.
            cited = {source["source"] for source in out.sources}
            assert cited and cited <= stored_ids, strategy
            # AUTO is routed by the model's answer ("local").
            expected = "local" if strategy is SearchStrategy.AUTO else strategy.value
            assert out.metadata["search_strategy"] == expected
        assert encoder.calls >= 1
        assert set(server.embed_models) == {"synthetic-embed"}

        # The evaluators, with a LangChain judge on the same fake model and
        # sources resolved to file names through the SQLite vector store.
        config.evaluation.enabled_evaluators = [
            EvaluatorType.RETRIEVAL,
            EvaluatorType.ANSWER_MATCH,
            EvaluatorType.GRAPH_AWARE,
            EvaluatorType.LANGCHAIN,
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
        results, _, summary = asyncio.run(
            manager.evaluate_dataset(queries, truths, show_progress=False)
        )
    finally:
        chain.close()

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
    assert stats["correctness"]["mean"] == 1.0
    assert stats["partial_correctness"]["mean"] == 1.0
    cited = {name for rank in results[0].retrieved_source_ids for name in rank}
    assert cited == {"logistics.txt", "freight.txt"}

    # run-visualization renders the exported graph with either layout
    # method under --backend local.
    data = viz / "visualization_data.json"
    assert data.is_file()
    for method in ("none", "description"):
        assert _render(tmp_path, monkeypatch, data, method)


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
