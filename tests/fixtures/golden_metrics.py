# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The golden-metrics regression gate's harness (docs/guides/evaluate.md
"Golden metrics").

It ingests the synthetic corpora the repository already has, the
``structural`` benchmark (``prepare-benchmark structural --size small
--seed 0``, generated at run time) and the multilingual bridge corpus
(``tests/fixtures/bridge_corpus.py``), on the embedded SQLite stores with
deterministic models: ``FakeOllama`` answering with the console demo's
prompt-reading responder (extraction by name, extractive answers) and its
hashing vectors, and no reranker (a reranker re-sorts the candidates and
would hide a fusion change). Then ``run-eval`` answers every dataset with
every search strategy and search type and scores the answers with the
LLM-free evaluators only (retrieval, answer_match, graph_aware). The demo's
responder is a copy loaded with the case's entity names, so the gate moves
when that responder changes too. :func:`measure` returns the metric means;
``tests/fixtures/golden_metrics.json`` holds the committed baseline
(``scripts/gen_golden_metrics.py`` writes it) and
``tests/integration/test_golden_metrics.py`` compares the two.

The numbers measure this pipeline on fake models: they catch a change in
ranking, fusion, graph building, context packing or scoring, not answer
quality, which only a real model shows.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import json
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import yaml

from tests.fixtures import console_demo
from tests.fixtures.bridge_corpus import CORPORA as BRIDGE_CORPORA
from tests.fixtures.bridge_corpus import SUPPORT as BRIDGE_SUPPORT
from tests.fixtures.fakes.embeddings import HashingEmbeddings
from tests.fixtures.fakes.model_servers import FakeOllama
from unified_kg_rag.adapters.local.embeddings import OllamaEmbeddingFactory
from unified_kg_rag.adapters.local.models import OllamaLanguageModelFactory
from unified_kg_rag.application.bootstrap import build_providers
from unified_kg_rag.application.cli import run_evaluation
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import (
    Config,
    EvaluatorType,
    Methodology,
    PipelineConfig,
    PipelineStageStatus,
    PipelineStageType,
    SearchStrategy,
    SearchType,
)
from unified_kg_rag.domain.models.config import (
    ChunkingStrategy,
    ChunkLengthUnit,
    LanguageCode,
    SeedGraphConfig,
)
from unified_kg_rag.evaluation.benchmarks import SPLITS, structural

BASELINE = Path(__file__).with_name("golden_metrics.json")
SCHEMA_VERSION = 1
DIMENSIONS = 64
STRATEGIES = tuple(SearchStrategy)
SEARCH_TYPES = tuple(SearchType)
EVALUATORS = (
    EvaluatorType.RETRIEVAL,
    EvaluatorType.ANSWER_MATCH,
    EvaluatorType.GRAPH_AWARE,
)
# Every methodology on one index, so every strategy answers.
METHODOLOGIES = (
    Methodology.GRAPHRAG,
    Methodology.LIGHTRAG,
    Methodology.HIPPORAG,
    Methodology.RAPTOR,
)


def _responder(entities: dict[str, str]) -> types.ModuleType:
    """A private copy of the console demo's model responder that knows
    ``entities`` (name -> type) instead of the demo corpus'."""
    spec = importlib.util.spec_from_file_location(
        "golden_fake_models", console_demo.DEMO / "fake_models.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ENTITIES = dict(entities)
    module.RELATION_TYPES = {}
    return module


def _config(name: str, stores: Path, chunk_size: int, language: LanguageCode) -> Config:
    config = Config(
        backend="local",
        models={
            "default": "synthetic-llm",
            "fast": "synthetic-fast",
            "embedding": {"model_id": "synthetic-embed", "dimension": DIMENSIONS},
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
    chunking.length_unit = ChunkLengthUnit.CHARS
    chunking.fallback_chunk_size = chunk_size
    chunking.chunk_overlap = 0
    chunking.min_chunk_size = 1
    config.processing.translation.enabled = False
    config.processing.translation.source_language = language
    config.processing.translation.target_language = language
    config.processing.gleaning.enabled = False
    config.processing.claim_extraction.enabled = False
    config.retry.ingestion_item.max_attempts = 1
    config.indexing.methodologies = list(METHODOLOGIES)
    config.graph.visualization.export_data = False
    config.evaluation.enabled_evaluators = list(EVALUATORS)
    config.providers.rerank.kind = "none"
    config.logging.log_to_file = False
    # Document and text-unit ids derive from the source scope, the source
    # directory by default: a fixed one keeps them, and the order of tied
    # scores (by id), the same in every temporary directory.
    config.processing.document_parsing.source_scope = f"golden-metrics/{name}"
    return config


class _Case:
    """One index and its ``run-eval`` dataset."""

    def __init__(
        self,
        name: str,
        root: Path,
        corpus: Path,
        dataset: Path,
        config: Config,
        entities: dict[str, str],
        split: bool,
    ) -> None:
        self.name = name
        self.root = root
        self.corpus = corpus
        self.dataset = dataset
        self.config = config
        self.split = split
        responder = _responder(entities)
        self.server = FakeOllama(
            lambda system, human: str(responder.respond(system, human)[0]),
            "",
            DIMENSIONS,
        )
        # The demo's hashing vectors: words, and character bigrams of
        # Hangul, Kana and Han runs, so Korean, Japanese and Chinese texts
        # that share words are close too (HashingEmbeddings hashes such a
        # run as one word, and every passage would tie).
        self.server.embedder = _DemoEmbeddings(responder)
        transport = httpx.MockTransport(self.server)
        self.providers = build_providers(
            config,
            llm_factory=OllamaLanguageModelFactory(config, transport=transport),
            embedding_factory=OllamaEmbeddingFactory(config, transport=transport),
        )


class _DemoEmbeddings(HashingEmbeddings):
    def __init__(self, responder: types.ModuleType) -> None:
        super().__init__(DIMENSIONS)
        self.responder = responder

    def embed_query(self, text: str) -> list[float]:
        return list(self.responder.embed(text, self.dimensions))


def _structural_case(root: Path) -> _Case:
    prepared = root / "prepared"
    structural.prepare(prepared, seed=0, size="small")
    world = structural.generate_world(0, "small")
    entities = {node.name: node.type for node in world.nodes.values()}
    # 200 characters: the longest vendor documents (up to 230) are two chunks.
    config = _config("structural", root / "stores", 200, LanguageCode.EN)
    snippet = (prepared / "structural-config.yaml").read_text(encoding="utf-8")
    extraction = yaml.safe_load(snippet)["processing"]
    graph = config.processing.graph_extraction
    for key, value in extraction["graph_extraction"].items():
        setattr(graph, key, value)
    (seed,) = extraction["seed_graphs"]
    config.processing.seed_graphs = [
        SeedGraphConfig(**{**seed, "path": str(prepared / seed["path"])})
    ]
    return _Case(
        "structural",
        root,
        prepared / "corpus",
        prepared / "eval_data.json",
        config,
        entities,
        split=True,
    )


def _bridge_case(root: Path, language: LanguageCode) -> _Case:
    corpus = BRIDGE_CORPORA[language]
    source = root / "corpus"
    source.mkdir(parents=True)
    for name, text in corpus["files"].items():
        (source / name).write_text(text, encoding="utf-8")
    dataset = root / "eval_data.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "query_id": f"bridge-{language.value}",
                    "question": corpus["question"],
                    "answer": corpus["answer"],
                    "reference_sources": sorted(BRIDGE_SUPPORT),
                    "expected_entities": list(corpus["names"]),
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    entities = dict.fromkeys(corpus["names"], "ORGANIZATION")
    return _Case(
        f"bridge-{language.value}",
        root,
        source,
        dataset,
        # 60 characters: each passage one chunk (bridge_corpus.py).
        _config(f"bridge-{language.value}", root / "stores", 60, language),
        entities,
        split=False,
    )


def _ingest(case: _Case) -> None:
    pipeline = DataIngestionPipeline(
        case.config,
        PipelineConfig(
            stages_enabled=dict.fromkeys(PipelineStageType, True),
            local_directory=case.root / "run",
        ),
        source_directory=case.corpus,
        providers=case.providers,
    )
    for stage in pipeline.stages:
        if isinstance(stage, ps.GraphResolutionStage):
            stage.resolver.entity_resolver.use_process_pool = False
            stage.resolver.relationship_resolver.use_process_pool = False
    try:
        context = pipeline.run(case.corpus, pipeline_id="golden")
    finally:
        pipeline.close()
    if context.status is not PipelineStageStatus.COMPLETED:
        errors = " ".join(r.error_message or "" for r in context.stage_results)
        raise RuntimeError(f"{case.name}: ingestion failed: {errors}")


def _run_eval(case: _Case, chain: GraphRAGChain, argv: list[str]) -> int:
    args = run_evaluation.CommandLineInterface().parser.parse_args(
        [*argv, "--output-format", "json"]
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return asyncio.run(
            run_evaluation.EvaluationRunner(args, chain, case.config).run()
        )


def _summaries(directory: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Each strategy's run summary (``evaluation_summary_<strategy>_<stamp>_<id>.json``)."""
    strategies = {s.value for s in STRATEGIES}
    for path in sorted(directory.rglob("evaluation_summary_*.json")):
        strategy = path.stem.removeprefix("evaluation_summary_").split("_", 1)[0]
        assert strategy in strategies, path
        yield strategy, json.loads(path.read_text(encoding="utf-8"))


def _scores(summary: dict[str, Any]) -> dict[str, Any]:
    stats = summary["metric_statistics"]
    return {
        "queries": summary["total_queries"],
        "failed": summary["failed_evaluations"],
        "metrics": {name: stats[name]["mean"] for name in sorted(stats)},
    }


CASES = ("structural", *(f"bridge-{language.value}" for language in BRIDGE_CORPORA))


def build_case(name: str, root: Path) -> _Case:
    """Case ``name`` (one of ``CASES``) in the empty directory ``root``."""
    if name == "structural":
        return _structural_case(root)
    return _bridge_case(root, LanguageCode(name.removeprefix("bridge-")))


def measure_case(
    name: str, root: Path, search_types: tuple[SearchType, ...] = SEARCH_TYPES
) -> dict[str, Any]:
    """Case ``name``'s scores with each of ``search_types`` and every
    strategy: ``{split: {"<search type>/<strategy>": {"queries", "failed",
    "metrics": {name: mean}}}}``, the splits ``tune`` and ``test`` for a split
    dataset, ``all`` for one that is not."""
    case = build_case(name, root)
    _ingest(case)
    chain = GraphRAGChain(case.config, providers=case.providers)
    out: dict[str, Any] = {}
    try:
        for search_type in search_types:
            for split in SPLITS if case.split else (None,):
                directory = case.root / "eval" / search_type.value / (split or "all")
                argv = [
                    "--data-path",
                    str(case.dataset),
                    "--search-strategy",
                    ",".join(s.value for s in STRATEGIES),
                    "--search-type",
                    search_type.value,
                    "--evaluators",
                    ",".join(e.value for e in EVALUATORS),
                    "--output-directory",
                    str(directory),
                    *(["--split", split] if split else []),
                ]
                code = _run_eval(case, chain, argv)
                if code != 0:
                    raise RuntimeError(f"{case.name}: run-eval exited {code}")
                for strategy, summary in _summaries(directory):
                    key = f"{search_type.value}/{strategy}"
                    out.setdefault(split or "all", {})[key] = _scores(summary)
    finally:
        chain.close()
    return out


def measure(workdir: Path) -> dict[str, Any]:
    """Every case's scores (:func:`measure_case`), as the baseline holds them."""
    return {
        "schema_version": SCHEMA_VERSION,
        "cases": {name: measure_case(name, workdir / name) for name in CASES},
    }


def select(
    result: dict[str, Any], case: str, search_type: SearchType
) -> dict[str, Any]:
    """The part of a :func:`measure` result of one case and search type."""
    prefix = f"{search_type.value}/"
    splits = result["cases"].get(case, {})
    return {
        "schema_version": result.get("schema_version"),
        "cases": {
            case: {
                split: {k: v for k, v in runs.items() if k.startswith(prefix)}
                for split, runs in splits.items()
            }
        },
    }


# The partitions the gate compares: a split dataset's held-out half (the
# tune half is for development, docs/concepts/defaults-and-evidence.md),
# or the whole of one that is not split.
GATED = ("test", "all")


def compare(
    baseline: dict[str, Any], measured: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """The differences between two :func:`measure` results: those of a
    gated partition (``GATED``: the gate fails on any) and those of the
    tune half (reported only). Scores are compared exactly: the run is
    deterministic (fake models, hashing embeddings, SQLite stores, a fixed
    seed), so any difference is a change of the pipeline, not noise."""
    gated: list[str] = []
    reported: list[str] = []
    if baseline.get("schema_version") != measured.get("schema_version"):
        gated.append(
            f"schema_version {baseline.get('schema_version')} != "
            f"{measured.get('schema_version')}: regenerate the baseline"
        )
        return gated, reported
    old, new = baseline["cases"], measured["cases"]
    for case in sorted(old.keys() | new.keys()):
        if case not in old or case not in new:
            gated.append(f"{case}: only in the {'baseline' if case in old else 'run'}")
            continue
        for split in sorted(old[case].keys() | new[case].keys()):
            out = gated if split in GATED else reported
            before, after = old[case].get(split, {}), new[case].get(split, {})
            for run in sorted(before.keys() | after.keys()):
                where = f"{case} [{split}] {run}"
                if run not in before or run not in after:
                    side = "baseline" if run in before else "run"
                    out.append(f"{where}: only in the {side}")
                    continue
                out.extend(_differences(where, before[run], after[run]))
    return gated, reported


def _differences(
    where: str, before: dict[str, Any], after: dict[str, Any]
) -> list[str]:
    found = [
        f"{where} {key}: {before[key]} -> {after[key]}"
        for key in ("queries", "failed")
        if before[key] != after[key]
    ]
    old, new = before["metrics"], after["metrics"]
    for name in sorted(old.keys() | new.keys()):
        if name not in old or name not in new:
            side = "baseline" if name in old else "run"
            found.append(f"{where} {name}: only in the {side}")
        elif old[name] != new[name]:
            found.append(f"{where} {name}: {old[name]!r} -> {new[name]!r}")
    return found


def read_baseline(path: Path = BASELINE) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def write_baseline(measured: dict[str, Any], path: Path = BASELINE) -> None:
    path.write_text(
        json.dumps(measured, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
