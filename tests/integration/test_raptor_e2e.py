# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The raptor methodology end to end on the embedded stores (no AWS, no
container; docs/design.md#65-raptor-summary-tree-raptor-tree-opt-in).

A raptor-only ingestion runs no graph stage and builds a summary tree over
the stored text units; adding raptor to an existing namespace builds only
the tree; an incremental run converges with a full build of the same
corpus, reusing the summaries of unchanged clusters; a removal leaves no
summary of removed text, even when the run stops before the tree is built
again.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401 - fixture
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.models import (
    Config,
    Methodology,
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
)

pytestmark = pytest.mark.integration

TREE_PROMPT = "You write the summaries of a summary tree"

# Twelve topics, two passages each: enough text units for a level of
# clusters at min_nodes_to_cluster 4.
TOPICS = [
    ("Depot", "stores widgets for Vendor"),
    ("Carrier", "moves gadgets for Depot"),
    ("Buyer", "orders widgets from Vendor"),
    ("Bank", "lends money to Buyer"),
    ("Insurer", "covers the Depot against fire"),
    ("Auditor", "checks the books of Vendor"),
    ("Port", "ships containers for Carrier"),
    ("Mill", "makes steel for Vendor"),
]


def tree_summary(human: str) -> str:
    """The fake summary model: a deterministic function of the passages
    (their first words), so two builds over the same clusters agree."""
    passages = re.findall(r"^\[\d+\] (.*)$", human, flags=re.MULTILINE)
    return "Summary: " + "; ".join(" ".join(p.split()[:3]) for p in passages)


class _Calls:
    def __init__(self) -> None:
        self.kinds: Counter[str] = Counter()
        self.tree_prompts: list[str] = []

    def __call__(self, system: str, human: str) -> str:
        if TREE_PROMPT in system:
            self.kinds["summary"] += 1
            self.tree_prompts.append(system + "\n" + human)
            return tree_summary(human)
        if "knowledge graph extraction expert" in system:
            self.kinds["extraction"] += 1
        elif "community analysis" in system:
            self.kinds["report"] += 1
        else:
            self.kinds["other"] += 1
        return e2e._respond(system, human)

    def since(self, before: Counter[str]) -> Counter[str]:
        return self.kinds - before


def corpus(root: Path, topics: list[tuple[str, str]] = TOPICS) -> Path:
    source = root / "corpus"
    source.mkdir(parents=True, exist_ok=True)
    for name, fact in topics:
        (source / f"{name.lower()}.txt").write_text(
            f"{name} {fact}.\n\n{name} has an office in the north."
        )
    return source


def setup(
    tmp_path: Path,
    methodologies: list[Methodology] | None = None,
    *,
    registry: bool = True,
) -> tuple[Config, Providers, FakeOllama, _Calls]:
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.providers.doc_status.enabled = registry
    config.indexing.methodologies = methodologies or [Methodology.RAPTOR]
    tree = config.indexing.summary_tree
    tree.min_nodes_to_cluster = 4
    tree.target_children = 3
    calls = _Calls()
    server = FakeOllama(calls, e2e.ANSWER, e2e.DIM)
    encoder = e2e.FakeCrossEncoder("synthetic-reranker")
    return config, e2e._providers(config, server, encoder), server, calls


def ran(context: PipelineContext) -> set[str]:
    return {
        r.stage_name
        for r in context.stage_results
        if r.status is PipelineStageStatus.COMPLETED
    }


def ingest(
    config: Config,
    providers: Providers,
    source: Path,
    cache: Path,
    stages: set[PipelineStageType] | None = None,
) -> PipelineContext:
    if stages is None:
        return e2e._ingest(config, providers, source, cache)
    pipeline = DataIngestionPipeline(
        config,
        PipelineConfig(
            stages_enabled={stage: stage in stages for stage in PipelineStageType},
            local_directory=cache,
        ),
        source_directory=source,
        providers=providers,
    )
    try:
        return pipeline.run(source, pipeline_id=cache.name)
    finally:
        pipeline.close()


def record(providers: Providers) -> dict[str, Any]:
    indexer = providers.build_vector_indexer()
    try:
        stored = indexer.read_build_record(None)
    finally:
        indexer.close()
    assert stored is not None
    return stored


def tree_nodes(providers: Providers) -> list[dict[str, Any]]:
    indexer = providers.build_vector_indexer()
    try:
        return (
            indexer.read_summary_nodes(
                None,
                fields=[
                    "level",
                    "text",
                    "child_ids",
                    "text_unit_ids",
                    "attributes",
                    "language",
                ],
            )
            or []
        )
    finally:
        indexer.close()


def tree_record(providers: Providers) -> dict[str, Any] | None:
    indexer = providers.build_vector_indexer()
    try:
        return indexer.read_summary_tree_record(None)
    finally:
        indexer.close()


def shape(nodes: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return sorted(
        (n["id"], n["level"], n["text"], tuple(n["text_unit_ids"])) for n in nodes
    )


def test_a_raptor_only_ingestion_builds_a_tree_and_no_graph(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    config, providers, _, calls = setup(tmp_path)
    source = corpus(tmp_path)

    context = ingest(config, providers, source, tmp_path / "run-1")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert {"text_chunking", "indexing", "summary_tree"} <= ran(context)
    assert not ran(context) & {
        "graph_extraction",
        "gleaning",
        "graph_resolution",
        "claim_extraction",
        "claim_resolution",
        "graph_analysis",
        "community_detection",
    }
    assert calls.kinds["extraction"] == calls.kinds["report"] == 0
    assert calls.kinds["summary"] > 0
    stored = record(providers)
    assert stored["artifacts"] == ["text_units", "summary_tree"]
    assert stored["methodologies"] == ["raptor"]
    nodes = tree_nodes(providers)
    assert nodes and {n["level"] for n in nodes} >= {1}
    assert all(n["text"].startswith("Summary: ") for n in nodes)
    tree = tree_record(providers)
    assert tree is not None and tree["nodes"] == len(nodes)
    # No graph was built.
    assert e2e._entity_names(providers) == set()

    # A run with nothing new builds nothing: the tree is current.
    before = Counter(calls.kinds)
    context = ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert calls.since(before) == Counter()
    assert shape(tree_nodes(providers)) == shape(nodes)
    assert record(providers)["methodologies"] == ["raptor"]


def test_adding_raptor_builds_only_the_tree(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    config, providers, _, calls = setup(
        tmp_path, [Methodology.GRAPHRAG, Methodology.LIGHTRAG]
    )
    source = corpus(tmp_path)
    context = ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert "summary_tree" not in ran(context)
    assert calls.kinds["summary"] == 0
    assert "summary_tree" not in record(providers)["artifacts"]

    config.indexing.methodologies = [
        Methodology.GRAPHRAG,
        Methodology.LIGHTRAG,
        Methodology.RAPTOR,
    ]
    before = Counter(calls.kinds)
    context = ingest(config, providers, source, tmp_path / "run-2")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert config.indexing.reset is False
    assert set(calls.since(before)) == {"summary"}
    # The namespace was not rebuilt: run-1's record, the tree named in it.
    assert record(providers)["pipeline_id"] == "run-1"
    assert record(providers)["methodologies"] == [
        "graphrag",
        "lightrag",
        "hipporag",
        "raptor",
    ]
    assert tree_nodes(providers)


def test_an_incremental_tree_converges_with_a_full_build(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    config, providers, _, calls = setup(tmp_path / "inc")
    # One source scope for both corpora: document (and text unit) ids are
    # keyed by it.
    config.processing.document_parsing.source_scope = "corpus"
    source = corpus(tmp_path / "inc")
    context = ingest(config, providers, source, tmp_path / "inc-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    first = Counter(calls.kinds)

    # Add one document, change one, delete one.
    (source / "lab.txt").write_text("Lab tests widgets for Buyer.\n\nLab has a roof.")
    (source / "bank.txt").write_text("Bank lends money to Vendor.\n\nBank is old.")
    (source / "mill.txt").unlink()
    before = Counter(calls.kinds)
    context = ingest(config, providers, source, tmp_path / "inc-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    delta_calls = calls.since(before)
    built = tree_record(providers)
    assert built is not None
    # Unchanged clusters reused their summaries.
    assert built["reused_summaries"] > 0
    assert delta_calls["summary"] < first["summary"] + built["nodes"]

    # A full build of the same corpus in fresh stores builds the same tree.
    full_config, full_providers, _, _ = setup(tmp_path / "full")
    full_config.processing.document_parsing.source_scope = "corpus"
    full_source = tmp_path / "full" / "corpus"
    full_source.mkdir(parents=True)
    for path in source.iterdir():
        (full_source / path.name).write_text(path.read_text())
    context = ingest(full_config, full_providers, full_source, tmp_path / "full-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    incremental = tree_nodes(providers)
    assert {n["id"] for n in incremental} == {
        n["id"] for n in tree_nodes(full_providers)
    }
    assert shape(incremental) == shape(tree_nodes(full_providers))


def test_a_removal_leaves_no_summary_of_removed_text(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    config, providers, _, _ = setup(tmp_path)
    source = corpus(tmp_path)
    context = ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    gone = {
        hit["_id"]
        for hit in e2e._stored(
            config, providers, config.indexing.opensearch.text_units_index_prefix
        )
        if "Mill" in hit["_source"]["text"]
    }
    assert gone and any(set(n["text_unit_ids"]) & gone for n in tree_nodes(providers))

    # The run stops after indexing (no summary_tree stage): the summaries of
    # the deleted document are gone all the same, and raptor is no longer
    # held until a run builds the tree again.
    (source / "mill.txt").unlink()
    stages = {
        PipelineStageType.DOCUMENT_PARSING,
        PipelineStageType.DOCUMENT_LOADING,
        PipelineStageType.TEXT_CHUNKING,
        PipelineStageType.INDEXING,
    }
    context = ingest(config, providers, source, tmp_path / "run-2", stages)
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    left = tree_nodes(providers)
    assert left and not any(set(n["text_unit_ids"]) & gone for n in left)
    assert "raptor" not in record(providers)["methodologies"]

    # The next run builds the tree again.
    context = ingest(config, providers, source, tmp_path / "run-3")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert record(providers)["methodologies"] == ["raptor"]
    assert not any(set(n["text_unit_ids"]) & gone for n in tree_nodes(providers))
