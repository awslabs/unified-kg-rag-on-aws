# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The golden-metrics regression gate's harness (docs/guides/evaluate.md
"Golden metrics").

It ingests the synthetic corpora the repository already has, the
``structural`` benchmark (``prepare-benchmark structural --size small
--seed 0``, generated at run time), a deep variant of it, that variant
built with gleaning and then indexed incrementally (``_increment``) and the
multilingual bridge corpus (``tests/fixtures/bridge_corpus.py``), on the
embedded SQLite stores with deterministic models: ``FakeOllama`` answering
with the console demo's prompt-reading responder (extraction by name,
extractive answers, and keywords, report ratings, iterative verdicts and
DRIFT follow-ups read from the question and the entities its context
names) and its hashing vectors, and no reranker (a reranker re-sorts the
candidates and would hide a fusion change). Then ``run-eval`` answers every
dataset with every search strategy (and the deep case's global search
again with the settings in ``ARMS``) and scores the answers with the
LLM-free evaluators only (retrieval, answer_match, graph_aware). The demo's
responder is a copy loaded with the case's entity names, so the gate moves
when that responder changes too. :func:`measure` returns, per run, the
metric means and a fingerprint of each answer (:func:`fingerprint`: the
ranked sources, the context read and the answer), so a change that moves
no mean still shows; ``tests/fixtures/golden_metrics.json`` holds the
committed baseline (``scripts/gen_golden_metrics.py`` writes it) and
``tests/integration/test_golden_metrics.py`` compares the two.

The numbers measure this pipeline on fake models: they catch a change in
ranking, fusion, graph building, context packing, a strategy's own steps or
scoring, not answer quality, which only a real model shows.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import types
from collections.abc import Callable, Iterator
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
from unified_kg_rag.domain.retrieval.store_names import resolve_namespace
from unified_kg_rag.evaluation.benchmarks import SPLITS, structural

BASELINE = Path(__file__).with_name("golden_metrics.json")
SCHEMA_VERSION = 2
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


# The deep case: the structural world at ``medium`` size (40 vendors, 30
# buyers, 20 products; 128 text units), so the summary tree has two levels
# and more leaves than a tree search reads (``search.tree_search.text_top_k``
# 20), and the first Leiden pass splits a community over this many members,
# so the hierarchy has depths -1 to 2 (measured: 33 reports, 2/8/19/4 at
# depths -1/0/1/2) and every cut is a different set of reports. The default
# (20) splits none of the 88 entities' 8 communities.
DEEP_SIZE: structural.Size = "medium"
DEEP_MAX_CLUSTER_SIZE = 5
# The deep dataset: the first questions of each template in each split, in
# the order the benchmark writes them (206 in all at medium size; 24 kept). A fixed
# rule, not a pick by score: the gate's partitions are not tuned on.
DEEP_QUESTIONS_PER_TEMPLATE = 1


def _global_dynamic(config: Config) -> None:
    config.search.global_search.use_dynamic_selection = True


def _global_capped(config: Config) -> None:
    # Under the 8 first-pass reports: the cut steps down past depth 0 to the
    # coarser level (2 reports).
    config.search.global_search.max_reports_per_level = 4


# Extra runs of one strategy on the deep case, with settings off by default
# that have a path of their own: name -> (strategy, change to the config).
ARMS: dict[str, tuple[SearchStrategy, Callable[[Config], None]]] = {
    "dynamic": (SearchStrategy.GLOBAL, _global_dynamic),
    "capped": (SearchStrategy.GLOBAL, _global_capped),
}

# The incremental case: the deep world built with gleaning on, then one
# incremental run that adds vendor documents, edits the first vendor's and
# the first buyer's and deletes the last vendor's (in name order): the
# paths a full build never takes (the delta merge, the removal's strip and
# the report refresh, an incremental build record, a gleaning failure kept
# ungleaned). Before the increment the stored reports lose
# ``min_cut_depth``, as a namespace a release before 0.4.0 built holds
# them, so its global search reads reports of two releases
# (``GlobalSearchStrategy._has_coarser_cut``).
#
# The vendors added: two pairs of clusters, each three new vendors
# supplying a new buyer, the first vendor of a pair's first cluster also
# supplying its second cluster's buyer, so the delta's first pass finds the
# four clusters and a coarser level joins each pair (measured: 2 of the
# delta's reports at depth -1), as the full build's hierarchy has one, and
# the capped global cut could step onto it. One cluster's names hold the
# word a model answers for a query that names no entity ("none", which the
# fake model answers for the absent-entity questions): the focus guard must
# not read it as a name (``linking.focus_names_anything``).
INCREMENT_GROUPS = (
    (("Quenmar", "Quenmar Foods"), ("Quenvik", "Quenvik Clinics")),
    (("Zephyr", "Zephyr Foods"), ("None Such", "None Such Clinics")),
)
INCREMENT_KINDS = ("Works", "Trading", "Supplies")
# Gleaning (max_rounds 2): each text unit that names the first vendor has
# its round 1 add this product and round 2 fail (an unparsable answer), so
# the unit keeps its ungleaned extraction and record.
GLEANED_PRODUCT = "Brass Valve"
# The ``ARMS`` each case runs beside its default runs.
CASE_ARMS: dict[str, tuple[str, ...]] = {
    "structural-deep": tuple(ARMS),
    "structural-incremental": ("capped",),
}


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
        arms: tuple[str, ...] = (),
        gleaned: str | None = None,
        increment: Callable[[_Case], None] | None = None,
    ) -> None:
        self.name = name
        self.root = root
        self.corpus = corpus
        self.dataset = dataset
        self.config = config
        self.split = split
        self.arms = arms
        # Run after the full build (``measure_case``): an incremental run.
        self.increment = increment
        responder = _responder(entities)

        def respond(system: str, human: str) -> str:
            if gleaned is not None and "graph refinement specialist" in system:
                if gleaned in human:
                    return _gleaning_answer(gleaned, human)
            return str(responder.respond(system, human)[0])

        self.server = FakeOllama(respond, "", DIMENSIONS)
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


def _gleaning_answer(vendor: str, human: str) -> str:
    """The gleaning answer for a text unit naming ``vendor``: round 1 adds
    :data:`GLEANED_PRODUCT` (made by ``vendor``), round 2 (whose prompt
    lists it) is unparsable."""
    if GLEANED_PRODUCT in human:
        return "no plan"
    evidence = f"<text_evidence>{vendor}</text_evidence>"
    return (
        "<refinement_plan><quality_scores><completeness_score>0.5"
        "</completeness_score><accuracy_score>0.9</accuracy_score>"
        "</quality_scores><identified_issues>"
        "<issue><issue_type>MISSING_ENTITY</issue_type><details>"
        f"<name>{GLEANED_PRODUCT}</name><type>PRODUCT</type>"
        f"<description>A part {vendor} makes.</description>"
        f"<confidence>8</confidence></details>{evidence}</issue>"
        "<issue><issue_type>MISSING_RELATIONSHIP</issue_type><details>"
        f"<source>{vendor}</source><target>{GLEANED_PRODUCT}</target>"
        f"<type>MAKES</type><description>{vendor} makes it.</description>"
        f"<strength>7</strength></details>{evidence}</issue>"
        "</identified_issues></refinement_plan>"
    )


def _documents(corpus: Path, prefix: str) -> list[Path]:
    """The corpus' documents of one kind (``v``: vendors, ``b``: buyers), in
    name order."""
    return sorted(path for path in corpus.iterdir() if path.name.startswith(prefix))


def _title(path: Path) -> str:
    return path.read_text(encoding="utf-8").splitlines()[0].strip()


def _increment_entities() -> dict[str, str]:
    """The names the incremental run's documents add, by type."""
    names = {}
    for pair in INCREMENT_GROUPS:
        for prefix, buyer in pair:
            names.update({f"{prefix} {kind}": "VENDOR" for kind in INCREMENT_KINDS})
            names[buyer] = "BUYER"
    return names


def _increment(case: _Case) -> None:
    """The incremental case's second run (see :data:`INCREMENT_GROUPS`)."""
    _drop_cut_bounds(case)
    corpus = case.corpus
    vendors = _documents(corpus, "v")
    for first_cluster, second_cluster in INCREMENT_GROUPS:
        for (prefix, buyer), bridged in (
            (first_cluster, second_cluster[1]),
            (second_cluster, None),
        ):
            for kind in INCREMENT_KINDS:
                name = f"{prefix} {kind}"
                supplied = buyer
                if bridged is not None and kind == INCREMENT_KINDS[0]:
                    supplied += f", {bridged}"
                text = (
                    f"{name} is a vendor based in Westria. {name} supplies {supplied}."
                )
                path = corpus / f"x-{name.lower().replace(' ', '-')}.txt"
                path.write_text(f"{name}\n\n{text}\n", encoding="utf-8")
    (first, first_buyer), _ = INCREMENT_GROUPS[0]
    for path, sentence in (
        (vendors[0], f" {_title(vendors[0])} supplies {first_buyer}."),
        (_documents(corpus, "b")[0], f" It also buys from {first} Works."),
    ):
        path.write_text(
            path.read_text(encoding="utf-8").rstrip() + sentence + "\n",
            encoding="utf-8",
        )
    vendors[-1].unlink()
    _ingest(case, pipeline_id="golden-increment")


def _drop_cut_bounds(case: _Case) -> None:
    """Set ``min_cut_depth`` to null on every stored community report: the
    shape the report refresh writes for a report a release before 0.4.0
    wrote (it keeps the stored null). A 0.3.0 store itself holds no field
    at all; both shapes read as an earlier release's report
    (``holds_root_depth_reports`` searches for reports without the field),
    and the vector conformance suite writes the absent one."""
    # The embedded store's indexer: its live index and engine.
    vector: Any = case.providers.build_vector_indexer()
    try:
        suffix = resolve_namespace(case.config, None)
        pages = vector.scan_items("community_reports", suffix)
        assert pages is not None
        ids = [str(item["id"]) for page in pages for item in page]
        alias = vector._get_name(
            case.config.indexing.opensearch.community_reports_index_prefix, suffix
        )
        index = vector._live_index(alias)
        assert ids and index
        vector.opensearch_client.bulk_update(
            index, [{"id": i, "min_cut_depth": None} for i in ids]
        )
        assert vector.holds_root_depth_reports(suffix)
    finally:
        vector.close()


def _structural_case(
    root: Path, deep: bool = False, incremental: bool = False
) -> _Case:
    name = (
        "structural-incremental"
        if incremental
        else "structural-deep" if deep else "structural"
    )
    deep = deep or incremental
    size: structural.Size = DEEP_SIZE if deep else "small"
    prepared = root / "prepared"
    structural.prepare(prepared, seed=0, size=size)
    world = structural.generate_world(0, size)
    entities = {node.name: node.type for node in world.nodes.values()}
    if incremental:
        entities.update(_increment_entities())
    # 200 characters: the longest vendor documents (up to 230) are two chunks.
    config = _config(name, root / "stores", 200, LanguageCode.EN)
    dataset = prepared / "eval_data.json"
    if deep:
        config.graph.community_detection.max_cluster_size = DEEP_MAX_CLUSTER_SIZE
        dataset = root / "eval_data.json"
        dataset.write_text(
            json.dumps(_first_of_each_template(prepared / "eval_data.json")),
            encoding="utf-8",
        )
    gleaned = None
    if incremental:
        config.processing.gleaning.enabled = True
        config.processing.gleaning.max_rounds = 2
        gleaned = _title(_documents(prepared / "corpus", "v")[0])
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
        name,
        root,
        prepared / "corpus",
        dataset,
        config,
        entities,
        split=True,
        arms=CASE_ARMS.get(name, ()),
        gleaned=gleaned,
        increment=_increment if incremental else None,
    )


def _first_of_each_template(path: Path) -> list[dict[str, Any]]:
    """The first ``DEEP_QUESTIONS_PER_TEMPLATE`` questions of each template
    and split of a structural dataset, in its order."""
    taken: dict[tuple[str, str], int] = {}
    kept = []
    for question in json.loads(path.read_text(encoding="utf-8")):
        metadata = question["metadata"]
        key = (metadata["template"], metadata["split"])
        if taken.get(key, 0) < DEEP_QUESTIONS_PER_TEMPLATE:
            taken[key] = taken.get(key, 0) + 1
            kept.append(question)
    return kept


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


def _ingest(case: _Case, pipeline_id: str = "golden") -> None:
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
        context = pipeline.run(case.corpus, pipeline_id=pipeline_id)
    finally:
        pipeline.close()
    if context.status is not PipelineStageStatus.COMPLETED:
        errors = " ".join(r.error_message or "" for r in context.stage_results)
        raise RuntimeError(f"{case.name}: ingestion failed: {errors}")


def _summaries(
    directory: Path,
) -> Iterator[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
    """Each strategy's run summary (``evaluation_summary_<strategy>_<stamp>_<id>.json``)
    and its per-question results (``evaluation_results_...``, same suffix)."""
    strategies = {s.value for s in STRATEGIES}
    for path in sorted(directory.rglob("evaluation_summary_*.json")):
        suffix = path.stem.removeprefix("evaluation_summary_")
        strategy = suffix.split("_", 1)[0]
        assert strategy in strategies, path
        results = path.with_name(f"evaluation_results_{suffix}.json")
        yield (
            strategy,
            json.loads(path.read_text(encoding="utf-8")),
            json.loads(results.read_text(encoding="utf-8")),
        )


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.blake2b(text.encode("utf-8"), digest_size=5).hexdigest()


def fingerprint(result: dict[str, Any]) -> str:
    """One question's answer as ``<ranked>-<read>``: a digest of the sources
    the strategy returned, in rank order, with their types and the ones the
    answer cites, and one of what the answer step read and wrote (the
    contexts, entities and relationships, the answer, whether it abstained
    or failed). Timings and usage are left out: they vary between runs."""
    ranked = [
        result.get("retrieved_source_ids"),
        result.get("retrieved_source_types"),
        result.get("cited_source_ids"),
        result.get("cited_section_types"),
    ]
    read = [
        result.get("retrieved_contexts"),
        result.get("retrieved_entities"),
        result.get("retrieved_relationships"),
        result.get("generated_answer"),
        result.get("abstained"),
        result.get("error"),
    ]
    return f"{_digest(ranked)}-{_digest(read)}"


def _scores(summary: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    stats = summary["metric_statistics"]
    return {
        "queries": summary["total_queries"],
        "failed": summary["failed_evaluations"],
        "metrics": {name: stats[name]["mean"] for name in sorted(stats)},
        "answers": {str(r["query_id"]): fingerprint(r) for r in results},
    }


CASES = (
    "structural",
    "structural-deep",
    "structural-incremental",
    *(f"bridge-{language.value}" for language in BRIDGE_CORPORA),
)
# The search types each case runs. The deep case is there for the
# strategies' own paths, which run under any search type: the default one.
CASE_SEARCH_TYPES: dict[str, tuple[SearchType, ...]] = {
    name: (SearchType.HYBRID,) if name.startswith("structural-") else SEARCH_TYPES
    for name in CASES
}
# The gate's tests: one per case, search type and split (``None``: the
# whole of an unsplit dataset), so ``-n`` spreads the deep case too.
RUNS: tuple[tuple[str, SearchType, str | None], ...] = tuple(
    (name, search_type, split)
    for name in CASES
    for search_type in CASE_SEARCH_TYPES[name]
    for split in (SPLITS if name.startswith("structural") else (None,))
)


def build_case(name: str, root: Path) -> _Case:
    """Case ``name`` (one of ``CASES``) in the empty directory ``root``."""
    if name.startswith("structural"):
        return _structural_case(
            root,
            deep=name == "structural-deep",
            incremental=name == "structural-incremental",
        )
    return _bridge_case(root, LanguageCode(name.removeprefix("bridge-")))


def _evaluate(
    case: _Case,
    chain: GraphRAGChain,
    config: Config,
    strategies: tuple[SearchStrategy, ...],
    search_type: SearchType,
    split: str | None,
    directory: Path,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """``run-eval`` of ``strategies`` on ``split``: each strategy's scores."""
    argv = [
        "--data-path",
        str(case.dataset),
        "--search-strategy",
        ",".join(s.value for s in strategies),
        "--search-type",
        search_type.value,
        "--evaluators",
        ",".join(e.value for e in EVALUATORS),
        "--output-directory",
        str(directory),
        *(["--split", split] if split else []),
    ]
    args = run_evaluation.CommandLineInterface().parser.parse_args(
        [*argv, "--output-format", "json"]
    )
    with contextlib.redirect_stdout(io.StringIO()):
        code = asyncio.run(run_evaluation.EvaluationRunner(args, chain, config).run())
    if code != 0:
        raise RuntimeError(f"{case.name}: run-eval exited {code}")
    for strategy, summary, results in _summaries(directory):
        yield strategy, _scores(summary, results)


def measure_case(
    name: str,
    root: Path,
    search_types: tuple[SearchType, ...] | None = None,
    splits: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Case ``name``'s scores with each of ``search_types`` (by default
    ``CASE_SEARCH_TYPES``) and every strategy, and with its ``ARMS``, on
    each of ``splits`` (by default both of a split dataset):
    ``{split: {"<search type>/<strategy>[@<arm>]": {"queries", "failed",
    "metrics": {name: mean}, "answers": {query id: fingerprint}}}}``, the
    splits ``tune`` and ``test`` for a split dataset, ``all`` for one that
    is not."""
    case = build_case(name, root)
    _ingest(case)
    if case.increment is not None:
        case.increment(case)
    chains = {"": (case.config, GraphRAGChain(case.config, providers=case.providers))}
    for arm in case.arms:
        config = copy.deepcopy(case.config)
        ARMS[arm][1](config)
        chains[arm] = (config, GraphRAGChain(config, providers=case.providers))
    out: dict[str, Any] = {}
    try:
        for search_type in search_types or CASE_SEARCH_TYPES[name]:
            for split in (splits or SPLITS) if case.split else (None,):
                for arm, (config, chain) in chains.items():
                    strategies = (ARMS[arm][0],) if arm else STRATEGIES
                    directory = (
                        case.root
                        / "eval"
                        / search_type.value
                        / (split or "all")
                        / (arm or "default")
                    )
                    for strategy, scores in _evaluate(
                        case, chain, config, strategies, search_type, split, directory
                    ):
                        key = f"{search_type.value}/{strategy}"
                        key += f"@{arm}" if arm else ""
                        out.setdefault(split or "all", {})[key] = scores
    finally:
        for _, chain in chains.values():
            chain.close()
    return out


def measure(workdir: Path) -> dict[str, Any]:
    """Every case's scores (:func:`measure_case`), as the baseline holds them."""
    return {
        "schema_version": SCHEMA_VERSION,
        "cases": {name: measure_case(name, workdir / name) for name in CASES},
    }


def select(
    result: dict[str, Any],
    case: str,
    search_type: SearchType,
    split: str | None = None,
) -> dict[str, Any]:
    """The part of a :func:`measure` result of one case and search type
    (and ``split``, by default every one)."""
    prefix = f"{search_type.value}/"
    splits = result["cases"].get(case, {})
    return {
        "schema_version": result.get("schema_version"),
        "cases": {
            case: {
                name: {k: v for k, v in runs.items() if k.startswith(prefix)}
                for name, runs in splits.items()
                if split is None or name == split
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
    old, new = before["answers"], after["answers"]
    for query in sorted(old.keys() | new.keys()):
        if query not in old or query not in new:
            side = "baseline" if query in old else "run"
            found.append(f"{where} answer {query}: only in the {side}")
            continue
        (old_ranked, old_read), (new_ranked, new_read) = (
            old[query].split("-"),
            new[query].split("-"),
        )
        changed = [
            part
            for part, moved in (
                ("the ranked sources", old_ranked != new_ranked),
                ("the context or the answer", old_read != new_read),
            )
            if moved
        ]
        if changed:
            found.append(f"{where} answer {query}: {' and '.join(changed)} changed")
    return found


def read_baseline(path: Path = BASELINE) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def write_baseline(measured: dict[str, Any], path: Path = BASELINE) -> None:
    path.write_text(
        json.dumps(measured, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
