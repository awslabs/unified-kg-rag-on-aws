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

from tests.fixtures.fakes.chat_models import SUMMARY_TREE_PROMPT, CountingResponder
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
) -> tuple[Config, Providers, FakeOllama, CountingResponder]:
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.providers.doc_status.enabled = registry
    config.indexing.methodologies = methodologies or [Methodology.RAPTOR]
    # Text unit ids hash the source scope, which defaults to the corpus
    # directory: a fixed scope keeps the ids, and so the clusters the
    # assertions read, the same whatever tmp_path pytest hands out.
    config.processing.document_parsing.source_scope = "corpus"
    tree = config.indexing.summary_tree
    tree.min_nodes_to_cluster = 4
    tree.target_children = 3
    calls = CountingResponder(e2e._respond, summarize=tree_summary)
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


def live_indices(config: Config, providers: Providers) -> set[str]:
    """The vector-store index prefixes with a live index in the namespace."""
    from unified_kg_rag.domain.retrieval.index_prefixes import index_artifacts

    indexer = providers.build_vector_indexer()
    try:
        return {
            prefix
            for prefix in index_artifacts(config)
            if indexer.opensearch_client.get_index_name_by_alias(
                indexer._get_name(prefix, None)
            )
        }
    finally:
        indexer.close()


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
    # No graph was built: the graph store was never opened, and the vector
    # store holds no index of what the run does not build.
    graph = Path(config.providers.graph.sqlite_path)
    assert not graph.exists() or not graph.stat().st_size
    indices = config.indexing.opensearch
    assert live_indices(config, providers) == {
        indices.text_units_index_prefix,
        indices.summary_nodes_index_prefix,
    }

    # A run with nothing new builds nothing: the tree is current.
    before = Counter(calls.kinds)
    context = ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert calls.since(before) == Counter()
    assert shape(tree_nodes(providers)) == shape(nodes)
    assert record(providers)["methodologies"] == ["raptor"]


def test_a_raptor_only_full_build_creates_no_graph_indices(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # Without the registry every run is a full build, which empties each
    # index type it does not build: only where one is live.
    config, providers, _, _ = setup(tmp_path, registry=False)
    context = ingest(config, providers, corpus(tmp_path), tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    indices = config.indexing.opensearch
    assert live_indices(config, providers) == {
        indices.text_units_index_prefix,
        indices.summary_nodes_index_prefix,
    }


def test_a_raptor_only_namespace_needs_no_graph_store(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # A Gremlin graph store without an endpoint: a run that builds no graph
    # artifact never builds its writer, through additions, a removal and a
    # reset alike.
    from unified_kg_rag.adapters.storage.null_graph_indexer import NullGraphIndexer

    config, providers, _, _ = setup(tmp_path)
    config.providers.graph.kind = "gremlin"
    config.aws.neptune.use_iam = False
    config.aws.neptune.endpoint = None
    assert isinstance(providers.build_graph_indexer(), NullGraphIndexer)
    source = corpus(tmp_path)
    context = ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    (source / "mill.txt").unlink()
    context = ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    config.indexing.reset = True
    context = ingest(config, providers, source, tmp_path / "run-3")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
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


def test_a_raptor_only_reset_clears_the_graph_it_no_longer_maintains(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # A reset to raptor alone left the graph store's vertices and edges,
    # those of documents deleted since included; a reset empties its vector
    # indices of what it no longer builds, and with a graph store configured
    # (the embedded one here) its graph too.
    from unified_kg_rag.adapters.storage.sqlite_graph_indexer import (
        SQLiteGraphIndexer,
    )

    config, providers, _, _ = setup(
        tmp_path, [Methodology.GRAPHRAG, Methodology.LIGHTRAG, Methodology.RAPTOR]
    )
    source = corpus(tmp_path)
    context = ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)

    def graph_names() -> list[str]:
        graph = SQLiteGraphIndexer(config)
        try:
            return sorted(name for _, name in graph.read_entity_names())
        finally:
            graph.close()

    assert graph_names()
    entities = config.indexing.opensearch.entities_index_prefix
    assert e2e._stored(config, providers, entities)

    (source / "mill.txt").unlink()
    config.indexing.methodologies = [Methodology.RAPTOR]
    config.indexing.reset = True
    context = ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert graph_names() == []
    assert e2e._stored(config, providers, entities) == []
    assert record(providers)["methodologies"] == ["raptor"]


def test_a_reset_without_raptor_deletes_the_tree(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    config, providers, _, _ = setup(
        tmp_path, [Methodology.LIGHTRAG, Methodology.RAPTOR]
    )
    source = corpus(tmp_path)
    context = ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert tree_nodes(providers)
    indexer = providers.build_vector_indexer()
    alias = indexer._get_name(
        config.indexing.opensearch.summary_nodes_index_prefix, None
    )
    # The index of a build killed before its swap.
    indexer.opensearch_client.create_index(
        f"{alias}-{99999999999999999}",
        indexer._spec("summary nodes").mapping(),
        exist_ok=False,
    )
    indexer.close()

    config.indexing.methodologies = [Methodology.LIGHTRAG]
    config.indexing.reset = True
    context = ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert tree_nodes(providers) == []
    indexer = providers.build_vector_indexer()
    try:
        assert not indexer.opensearch_client.get_aliases_by_index(f"{alias}-*")
    finally:
        indexer.close()
    assert record(providers)["methodologies"] == ["lightrag", "hipporag"]


@pytest.mark.parametrize("registry", [True, False])
def test_adding_raptor_with_only_the_tree_stage_survives_changed_cache_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry: bool,
    no_aws,  # noqa: F811 - the fixture
) -> None:
    # The guide's path: `--enabled-stages summary_tree` under the pipeline id
    # that built the namespace. After an upgrade changes the stage cache keys
    # the earlier stages no longer look completed; the tree reads only the
    # stores, so it is built anyway, and nothing upstream runs again; it
    # probes only the indices the namespace lacks, as an incremental run.
    from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
    from unified_kg_rag.application.ingestion import pipeline_manager

    config, providers, _, calls = setup(
        tmp_path, [Methodology.GRAPHRAG, Methodology.LIGHTRAG], registry=registry
    )
    source = corpus(tmp_path)
    cache = tmp_path / "run-1"
    context = ingest(config, providers, source, cache)
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)

    keyed = pipeline_manager.stage_cache_key
    monkeypatch.setattr(
        pipeline_manager,
        "stage_cache_key",
        lambda *args, **kwargs: keyed(*args, **kwargs) + "-upgraded",
    )
    probes: list[bool] = []
    monkeypatch.setattr(
        OpenSearchIndexer,
        "check_index_creation",
        lambda self, suffix=None, *, rebuild=True: probes.append(rebuild),
    )
    config.indexing.methodologies = [
        Methodology.GRAPHRAG,
        Methodology.LIGHTRAG,
        Methodology.RAPTOR,
    ]
    before = Counter(calls.kinds)
    context = ingest(config, providers, source, cache, {PipelineStageType.SUMMARY_TREE})

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert set(calls.since(before)) == {"summary"}
    assert tree_nodes(providers)
    # Without the registry the run is a full build, which probed every index.
    assert probes == ([] if registry else [False])


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
    # the deleted document are gone all the same, and raptor is still held:
    # the tree left cites only stored text.
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
    assert record(providers)["methodologies"] == ["raptor"]

    # The next run builds the tree again.
    context = ingest(config, providers, source, tmp_path / "run-3")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert record(providers)["methodologies"] == ["raptor"]
    assert not any(set(n["text_unit_ids"]) & gone for n in tree_nodes(providers))


@pytest.mark.parametrize("refresh", ["every_run", "on_reset"])
def test_summaries_a_removal_deleted_are_built_again_over_the_same_text(
    tmp_path: Path, no_aws, refresh: str  # noqa: F811 - the fixture
) -> None:
    # Regression (rv25 nightly faults): a removal deleted the summaries
    # citing text units the stores then held again unchanged (a document
    # removed and restored, a crash between the two). The leaf digest was
    # the stored tree's, so the tree read as current and the namespace lost
    # those summaries for good. A FAILED document's retry, which writes its
    # units back as the same leaves, keeps them instead
    # (test_raptor_failed_retry).
    config, providers, _, _ = setup(tmp_path)
    config.indexing.summary_tree.refresh = refresh  # type: ignore[assignment]
    source = corpus(tmp_path)
    context = ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    built = shape(tree_nodes(providers))
    mill = sorted(
        hit["_id"]
        for hit in e2e._stored(
            config, providers, config.indexing.opensearch.text_units_index_prefix
        )
        if "Mill" in hit["_source"]["text"]
    )
    indexer = providers.build_vector_indexer()
    try:
        # A removal's deletion of their summaries (TextUnitRemoval), the
        # units then held again unchanged.
        deleted = indexer.delete_summary_nodes_citing(mill, "")
    finally:
        indexer.close()
    assert deleted.successful_items > 0 and len(tree_nodes(providers)) < len(built)
    assert (tree_record(providers) or {}).get("pruned") is True

    context = ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    if refresh == "every_run":
        assert shape(tree_nodes(providers)) == built
        assert (tree_record(providers) or {}).get("pruned") is False
    else:
        # on_reset: a delta run only deletes the nodes of removed text and
        # reports what no summary covers, until a reset.
        assert len(tree_nodes(providers)) < len(built)


def test_a_tree_over_the_limit_fails_before_its_vectors_are_read(
    tmp_path: Path, no_aws, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    import numpy as np

    from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer

    config, providers, _, _ = setup(tmp_path)
    by_id = OpenSearchIndexer.read_text_unit_fields
    read: list[Any] = []

    def recorded(self: Any, ids: Any, suffix: Any, *, fields: list[str]) -> Any:
        units = by_id(self, ids, suffix, fields=fields)
        if "text_embedding" in fields:
            read.extend(row["text_embedding"] for row in units)
        return units

    monkeypatch.setattr(OpenSearchIndexer, "read_text_unit_fields", recorded)
    source = corpus(tmp_path)
    context = ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    # The vectors are read as float32 arrays.
    assert read and all(
        isinstance(v, np.ndarray) and v.dtype == np.float32 for v in read
    )
    read.clear()
    config.indexing.summary_tree.max_leaves_in_memory = 3
    (source / "zeta.txt").write_text("Zeta repairs ships for Port.")
    context = ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.FAILED
    assert "max_leaves_in_memory" in e2e._errors(context)
    assert read == []
    # Nothing was written: the stored tree is intact and stays named, so
    # tree queries still read it (an earlier version stopped naming it).
    assert "summary_tree" in record(providers)["artifacts"]
    assert tree_nodes(providers)


def test_the_leaf_digest_holds_one_page_of_text_at_a_time(
    tmp_path: Path, no_aws, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    # Regression: the digest read every text unit's text and translation
    # of the suffix into one list, every run (a current tree's and an
    # on_reset one's too), before the size check could refuse the tree.
    import gc
    import weakref

    from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
    from unified_kg_rag.application.ingestion import summary_tree_stage as sts

    config, providers, _, _ = setup(tmp_path)
    source = corpus(tmp_path)
    context = ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)

    class Row(dict):  # a dict a weak reference can follow
        pass

    pages = OpenSearchIndexer.scan_text_unit_pages
    scan = OpenSearchIndexer.scan_text_units
    held: list[int] = []
    listed: list[list[str]] = []

    def paged(self: Any, suffix: Any, *, fields: list[str], **kwargs: Any) -> Any:
        refs: list[Any] = []
        for page in pages(self, suffix, fields=fields, **kwargs):
            if "text_embedding" not in fields:
                gc.collect()
                held.append(sum(ref() is not None for ref in refs))
            rows = [Row(row) for row in page]
            refs.extend(weakref.ref(row) for row in rows)
            yield rows

    def scanned(self: Any, suffix: Any, *, fields: list[str], **kwargs: Any) -> Any:
        if "text_embedding" not in fields:
            listed.append(fields)
        return scan(self, suffix, fields=fields, **kwargs)

    monkeypatch.setattr(sts, "_DIGEST_PAGE", 1)
    monkeypatch.setattr(OpenSearchIndexer, "scan_text_unit_pages", paged)
    monkeypatch.setattr(OpenSearchIndexer, "scan_text_units", scanned)
    context = ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert listed == []
    # Read a unit at a time, at most the one before it is still held.
    assert len(held) > 2 and max(held) <= 1, held


def test_refresh_on_reset_builds_a_partial_tree_again(
    tmp_path: Path, no_aws, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    # A summary that failed leaves the tree partial, and its WARNING says
    # the next run builds it again: refresh: on_reset kept it until a reset.
    from unified_kg_rag.adapters.ingestion.tree_summarizer import TreeSummarizer

    config, providers, _, _ = setup(tmp_path)
    config.indexing.summary_tree.refresh = "on_reset"
    summarize = TreeSummarizer.summarize

    def first_fails(self: Any, clusters: Any, reusable: Any) -> Any:
        return [None, *summarize(self, clusters[1:], reusable)]

    source = corpus(tmp_path)
    with monkeypatch.context() as failing:
        failing.setattr(TreeSummarizer, "summarize", first_fails)
        context = ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert (tree_record(providers) or {}).get("partial") is True

    (source / "zeta.txt").write_text("Zeta repairs ships for Port.")
    context = ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert (tree_record(providers) or {}).get("partial") is False


def test_the_tree_is_served_while_it_is_rebuilt_and_refused_once_it_fails(
    tmp_path: Path, no_aws, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    from unified_kg_rag.application.ingestion import summary_tree_stage as sts
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
    from unified_kg_rag.domain.models import SearchStrategy
    from unified_kg_rag.shared import MethodologyNotIndexedError

    config, providers, _, _ = setup(tmp_path)
    source = corpus(tmp_path)
    context = ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    (source / "zeta.txt").write_text("Zeta repairs ships for Port.")
    (source / "mill.txt").unlink()
    during: list[str] = []

    def fail(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Indexing committed the run: the record still names the tree, and
        # a query reads it.
        during.extend(record(providers)["methodologies"])
        chain = GraphRAGChain(config, providers=providers)
        try:
            ask(chain, search_strategy=SearchStrategy.TREE)
        finally:
            chain.close()
        raise RuntimeError("simulated failure of the tree build")

    monkeypatch.setattr(sts.SummaryTreeStage, "_rebuild", fail)
    context = ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.FAILED
    assert during == ["raptor"]
    # The build failed: the tree no longer covers the stored text units.
    assert record(providers)["methodologies"] == []
    chain = GraphRAGChain(config, providers=providers)
    try:
        with pytest.raises(MethodologyNotIndexedError):
            ask(chain, search_strategy=SearchStrategy.TREE)
    finally:
        chain.close()


def test_a_failed_indexing_stage_builds_no_tree(
    tmp_path: Path, no_aws, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    # Run on (continue_on_error) past a failed indexing stage, the tree
    # stage must not summarize stores that may not hold the run's text.
    from unified_kg_rag.application.ingestion import pipeline_stages as ps

    config, providers, _, calls = setup(tmp_path)
    source = corpus(tmp_path)

    def fail(self: Any, context: Any) -> Any:
        raise RuntimeError("simulated indexing failure")

    monkeypatch.setattr(ps.IndexingStage, "execute", fail)
    pipeline = DataIngestionPipeline(
        config,
        PipelineConfig(
            stages_enabled=dict.fromkeys(PipelineStageType, True),
            local_directory=tmp_path / "run-1",
            continue_on_error=True,
        ),
        source_directory=source,
        providers=providers,
    )
    try:
        context = pipeline.run(source, pipeline_id="run-1")
    finally:
        pipeline.close()

    results = {r.stage_name: r for r in context.stage_results}
    assert results["indexing"].status is PipelineStageStatus.FAILED
    tree = results["summary_tree"]
    assert tree.status is PipelineStageStatus.COMPLETED, tree.error_message
    assert tree.metrics["tree_skipped"] == "indexing_failed"
    assert calls.kinds["summary"] == 0
    assert tree_record(providers) is None


def test_a_text_unit_without_a_vector_is_left_out_of_the_tree(
    tmp_path: Path,
    no_aws,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A stored unit whose vector (or text) is missing cannot be clustered:
    # it is skipped with a warning, the others are summarized. A vector
    # read as a JSON list is used like the store's float32 array.
    from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer

    config, providers, _, calls = setup(tmp_path)
    source = corpus(tmp_path)
    by_id = OpenSearchIndexer.read_text_unit_fields
    blank = "depot"

    def degraded(self: Any, ids: Any, suffix: Any, **kwargs: Any) -> Any:
        rows = by_id(self, ids, suffix, **kwargs)
        if "text_embedding" in kwargs["fields"]:
            for row in rows:
                if blank in str(row.get("attributes", {}).get("file_name", "")):
                    row["text_embedding"] = None
                elif row.get("text_embedding") is not None:
                    row["text_embedding"] = [float(v) for v in row["text_embedding"]]
        return rows

    monkeypatch.setattr(OpenSearchIndexer, "read_text_unit_fields", degraded)
    with caplog.at_level("WARNING"):
        context = ingest(config, providers, source, tmp_path / "run-1")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert "hold no vector or text" in caplog.text
    assert calls.kinds["summary"] > 0
    nodes = tree_nodes(providers)
    assert nodes
    cited = {unit for node in nodes for unit in node["text_unit_ids"]}
    indexer = providers.build_vector_indexer()
    try:
        units = indexer.scan_text_units(None, fields=["attributes"]) or []
    finally:
        indexer.close()
    skipped = {r["id"] for r in units if blank in r["attributes"].get("file_name", "")}
    assert len(skipped) == 2
    assert not cited & skipped
    assert cited


# --- the tree strategy ------------------------------------------------------


QUESTION = "Who stores the widgets for Vendor?"


def ask(chain: Any, query: str = QUESTION, **kwargs: Any) -> Any:
    from unified_kg_rag.application.retrieval.rag_chain import RAGInput, RAGOutput
    from unified_kg_rag.domain.models import SearchStrategy

    kwargs.setdefault("search_strategy", SearchStrategy.TREE)
    out = chain.invoke(RAGInput(query=query, **kwargs))
    assert isinstance(out, RAGOutput)
    return out


def by_type(out: Any, section: str) -> list[dict[str, Any]]:
    return [s for s in out.sources if s["metadata"]["section_type"] == section]


def test_tree_answers_from_summaries_and_their_passages(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
    from unified_kg_rag.domain.models import SearchStrategy
    from unified_kg_rag.shared import MethodologyNotIndexedError

    config, providers, server, _ = setup(tmp_path)
    # Fewer text units than the corpus holds, so the summaries' passages
    # are new ones (a passage already in the context is not repeated).
    config.search.tree_search.text_top_k = 2
    source = corpus(tmp_path)
    context = ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    stored = {n["id"]: n for n in tree_nodes(providers)}

    chain = GraphRAGChain(config, providers=providers)
    try:
        out = ask(chain)
        assert out.answer == e2e.ANSWER
        meta = out.search_results.metadata
        assert meta["graph_signal"] == "unavailable"
        assert meta["sources"]["tree_summaries"] > 0
        assert meta["sources"]["tree_text_units"] > 0
        summaries = by_type(out, "summary")
        assert summaries, out.sources
        for source_entry in summaries:
            m = source_entry["metadata"]
            node = stored[m["source_id"]]
            # Full provenance: the lineage down to the text units.
            assert m["text_unit_ids"] == node["text_unit_ids"]
            assert m["child_ids"] == node["child_ids"]
            assert m["level"] >= 1 and m["n_leaves"] >= 2
            assert not m.get("synthesized")
            assert source_entry["content"].startswith(f"Summary (level {m['level']}")
        # Each summary brought passages of its own lineage.
        texts = by_type(out, "text")
        expanded = [t for t in texts if t["metadata"].get("expanded_from")]
        assert expanded
        for passage in expanded:
            node = stored[passage["metadata"]["expanded_from"]]
            assert passage["metadata"]["chunk_id"] in node["text_unit_ids"]
            assert passage["metadata"]["attributes"]["file_name"].endswith(".txt")
        # Sources are deduplicated.
        ids = [s["metadata"]["source_id"] for s in out.sources]
        assert len(ids) == len(set(ids))

        # Expansion off: RAPTOR's collapsed tree, summaries and leaves only.
        flat = config.model_copy(deep=True)
        flat.search.tree_search.expand_children = 0
        flat_chain = GraphRAGChain(flat, providers=providers)
        try:
            plain = ask(flat_chain)
            assert plain.search_results.metadata["expanded_passages"] == 0
        finally:
            flat_chain.close()

        # The namespace holds raptor only: GraphRAG's and LightRAG's
        # strategies are refused before any model call; auto falls back to
        # tree, the one strategy reading what it holds.
        for strategy in (SearchStrategy.LOCAL, SearchStrategy.MIX):
            before = len(server.chat_models)
            with pytest.raises(MethodologyNotIndexedError, match="holds raptor"):
                ask(chain, search_strategy=strategy)
            assert len(server.chat_models) == before
        assert (
            ask(chain, search_strategy=SearchStrategy.AUTO).metadata["search_strategy"]
            == "tree"
        )
    finally:
        chain.close()


def test_tree_reads_what_the_namespace_holds_under_any_config(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
    from unified_kg_rag.domain.models import SearchStrategy

    config, providers, _, _ = setup(tmp_path)
    context = ingest(config, providers, corpus(tmp_path), tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    # A query config naming every methodology on the raptor-only namespace:
    # the graph buckets read the entities index only where it is held.
    wide = config.model_copy(deep=True)
    wide.indexing.methodologies = [
        Methodology.GRAPHRAG,
        Methodology.LIGHTRAG,
        Methodology.RAPTOR,
    ]
    chain = GraphRAGChain(wide, providers=providers)
    try:
        meta = ask(chain).search_results.metadata
        assert meta["graph_signal"] == "unavailable"
        assert "tree_graph_summaries" not in meta["sources"]
        # A key only the text units store leaves the summaries out (the
        # summary index would drop it and return them unfiltered).
        out = ask(chain, filters={"community_ids": "no-such-community"})
        meta = out.search_results.metadata
        assert meta["summaries_skipped_for_filters"] == ["community_ids"]
        assert "tree_summaries" not in meta["sources"]
        assert not by_type(out, "summary")
        # A key the summaries store filters them as before.
        meta = ask(chain, filters={"attr_category": "none"}).search_results.metadata
        assert "summaries_skipped_for_filters" not in meta
        assert meta["sources"]["tree_summaries"] == 0
        # naive and simple read what any namespace holds: served here.
        for strategy in (SearchStrategy.NAIVE, SearchStrategy.SIMPLE):
            out = ask(chain, search_strategy=strategy)
            assert by_type(out, "text"), strategy
        # simple reads the summary tree too, each summary with its lineage,
        # except under a key the summary index does not store.
        stored = {n["id"]: n for n in tree_nodes(providers)}
        out = ask(chain, search_strategy=SearchStrategy.SIMPLE)
        meta = out.search_results.metadata
        assert meta["sources"]["simple_summary_nodes"] > 0
        assert meta["sources"]["simple_text_units"] > 0
        summaries = by_type(out, "summary")
        assert summaries, out.sources
        for source_entry in summaries:
            m = source_entry["metadata"]
            assert m["text_unit_ids"] == stored[m["source_id"]]["text_unit_ids"]
            assert not m.get("synthesized")
        out = ask(
            chain,
            search_strategy=SearchStrategy.SIMPLE,
            filters={"community_ids": "no-such-community"},
        )
        assert "simple_summary_nodes" not in out.search_results.metadata["sources"]
        assert not by_type(out, "summary")
    finally:
        chain.close()


def test_a_filter_key_no_held_index_declares_is_refused(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    """On a raptor-only namespace, a key only the entities or relationships
    index declares is refused, under the raptor config and under one
    naming every methodology alike: no index the namespace holds would
    apply it, and the query ran unfiltered (simple swept the configured
    entities index the namespace lacks)."""
    from unified_kg_rag.application.retrieval.rag_chain import (
        GraphRAGChain,
        RAGInput,
    )
    from unified_kg_rag.domain.models import SearchStrategy
    from unified_kg_rag.shared import InvalidFilterError, MethodologyNotIndexedError

    config, providers, server, _ = setup(tmp_path)
    context = ingest(config, providers, corpus(tmp_path), tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    wide = config.model_copy(deep=True)
    wide.indexing.methodologies = [
        Methodology.GRAPHRAG,
        Methodology.LIGHTRAG,
        Methodology.RAPTOR,
    ]
    for query_config in (config, wide):
        chain = GraphRAGChain(query_config, providers=providers)
        try:
            before = len(server.chat_models)
            for strategy in (
                SearchStrategy.SIMPLE,
                SearchStrategy.TREE,
                SearchStrategy.AUTO,
            ):
                for key, value in (("type", "NO_SUCH_TYPE"), ("description", "x")):
                    with pytest.raises(InvalidFilterError, match=key):
                        ask(chain, search_strategy=strategy, filters={key: value})
                    stream = RAGInput(
                        query=QUESTION,
                        search_strategy=strategy,
                        filters={key: value},
                    )
                    with pytest.raises(InvalidFilterError, match=key):
                        list(chain.stream(stream))
            # Refused before any model call.
            assert len(server.chat_models) == before
            # A strategy the namespace does not serve is refused as such
            # first, whatever its filter keys.
            with pytest.raises(MethodologyNotIndexedError):
                ask(
                    chain,
                    search_strategy=SearchStrategy.LOCAL,
                    filters={"no_such_key": "x"},
                )
            # A key the text units declare still filters.
            for strategy in (SearchStrategy.SIMPLE, SearchStrategy.TREE):
                out = ask(
                    chain, search_strategy=strategy, filters={"attr_category": "none"}
                )
                assert not out.sources, strategy
        finally:
            chain.close()


def test_tree_is_refused_where_no_tree_was_built(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
    from unified_kg_rag.shared import MethodologyNotIndexedError

    config, providers, server, _ = setup(
        tmp_path, [Methodology.GRAPHRAG, Methodology.LIGHTRAG]
    )
    context = ingest(config, providers, corpus(tmp_path), tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    chain = GraphRAGChain(config, providers=providers)
    try:
        before = len(server.chat_models)
        with pytest.raises(MethodologyNotIndexedError, match="reads the raptor"):
            ask(chain)
        assert len(server.chat_models) == before
    finally:
        chain.close()


def test_tree_adds_the_graph_signal_where_entities_are_indexed(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain

    config, providers, _, _ = setup(
        tmp_path, [Methodology.GRAPHRAG, Methodology.LIGHTRAG, Methodology.RAPTOR]
    )
    context = ingest(config, providers, corpus(tmp_path), tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    chain = GraphRAGChain(config, providers=providers)
    try:
        out = ask(chain)
        meta = out.search_results.metadata
        assert meta["graph_signal"] == "on" and meta["graph_seeds"] > 0
        assert meta["sources"]["tree_graph_text_units"] > 0
        assert meta["sources"]["tree_graph_summaries"] > 0
        # A key the entities declare and the text units do not filters the
        # graph seeds; it leaves the text units and the summaries
        # unfiltered alike, so no summary is skipped for it.
        meta = ask(chain, filters={"type": "NO_SUCH_TYPE"}).search_results.metadata
        assert "summaries_skipped_for_filters" not in meta
        assert meta["sources"]["tree_summaries"] > 0
        assert meta["sources"]["tree_text_units"] > 0
        off = config.model_copy(deep=True)
        off.search.tree_search.graph_signal = "off"
        off_chain = GraphRAGChain(off, providers=providers)
        try:
            meta = ask(off_chain).search_results.metadata
            assert meta["graph_signal"] == "off"
            assert "tree_graph_summaries" not in meta["sources"]
        finally:
            off_chain.close()
    finally:
        chain.close()


def test_a_summary_passes_a_filter_only_when_all_its_documents_do(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    import json

    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain

    config, providers, _, _ = setup(tmp_path)
    # Text unit ids, hence the tree's clusters, are keyed by the source
    # scope, by default the corpus directory: pytest's tmp_path differs
    # from run to run, and some of its trees had no single-category node.
    # One fixed scope gives every run the tree that has them.
    config.processing.document_parsing.source_scope = "corpus"
    source = corpus(tmp_path)
    # Half the documents are of category "north", half "south", as
    # pre-parsed documents carrying their filters (read without a parsing
    # stage: parsed again, the JSON would be text without filters).
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    for index, path in enumerate(sorted(source.glob("*.txt"))):
        (parsed / f"{path.stem}.json").write_text(
            json.dumps(
                {
                    "document_id": path.stem,
                    "page_content": path.read_text(),
                    "file_name": path.name,
                    "file_path": path.name,
                    "metadata": {
                        "filters": {"category": "north" if index % 2 else "south"}
                    },
                }
            )
        )
        path.unlink()
    stages = set(PipelineStageType) - {PipelineStageType.DOCUMENT_PARSING}
    context = ingest(config, providers, parsed, tmp_path / "run-1", stages)
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    nodes = tree_nodes(providers)
    pure = [n for n in nodes if (n["attributes"].get("filters") or {}).get("category")]
    assert nodes and pure, nodes
    assert all(n["text"].startswith("Summary: ") for n in nodes)

    chain = GraphRAGChain(config, providers=providers)
    try:
        out = ask(chain, filters={"attr_category": "north"})
        assert by_type(out, "summary") and by_type(out, "text"), out.sources
        for entry in out.sources:
            if entry["metadata"]["section_type"] == "summary":
                filters = entry["metadata"]["attributes"]["filters"]
                assert filters["category"] == "north"
            else:
                assert entry["metadata"]["attributes"]["filters"] == {
                    "category": "north"
                }
    finally:
        chain.close()


def _parsed_with_filters(
    tmp_path: Path, filters: dict[str, str], parsed: Path | None = None
) -> Path:
    """The corpus as pre-parsed documents, each with ``filters`` (read
    without a parsing stage, see the test above)."""
    import json

    source = corpus(tmp_path)
    parsed = parsed or tmp_path / "parsed"
    parsed.mkdir(exist_ok=True)
    for path in sorted(source.glob("*.txt")):
        (parsed / f"{path.stem}.json").write_text(
            json.dumps(
                {
                    "document_id": path.stem,
                    "page_content": path.read_text(),
                    "file_name": path.name,
                    "file_path": path.name,
                    "metadata": {"filters": filters},
                }
            )
        )
    return parsed


def test_a_reset_after_the_filters_changed_rebuilds_the_tree(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # Regression: the leaf digest hashed the text-unit ids only, so a reset
    # after the documents' filters changed (same content, same ids) found
    # the tree current: its nodes kept the old uniform filters, and a query
    # filtered by the old value still got the summaries.
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain

    config, providers, _, _ = setup(tmp_path)
    config.processing.document_parsing.source_scope = "corpus"
    stages = set(PipelineStageType) - {PipelineStageType.DOCUMENT_PARSING}
    parsed = _parsed_with_filters(tmp_path, {"category": "north"})
    context = ingest(config, providers, parsed, tmp_path / "run-1", stages)
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert {n["attributes"]["filters"]["category"] for n in tree_nodes(providers)} == {
        "north"
    }

    _parsed_with_filters(tmp_path, {"category": "south"}, parsed)
    config.indexing.reset = True
    context = ingest(config, providers, parsed, tmp_path / "run-2", stages)
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)

    nodes = tree_nodes(providers)
    assert nodes
    assert {n["attributes"]["filters"]["category"] for n in nodes} == {"south"}
    chain = GraphRAGChain(config, providers=providers)
    try:
        assert not ask(chain, filters={"attr_category": "north"}).sources
        assert by_type(ask(chain, filters={"attr_category": "south"}), "summary")
    finally:
        chain.close()


def test_the_tree_is_built_over_the_translations_and_follows_them(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # The leaves are the translations into the index language, with their
    # own token counts (the stored count is the source text's), and a reset
    # that translates the same text units otherwise builds the tree again:
    # the leaf digest hashed the ids only, so the summaries of the old
    # translations stayed.
    from unified_kg_rag.application.ingestion.summary_tree_stage import (
        SummaryTreeStage,
    )
    from unified_kg_rag.domain.models.config import LanguageCode

    config, providers, server, calls = setup(tmp_path)
    config.processing.document_parsing.source_scope = "corpus"
    translation = config.processing.translation
    translation.enabled = True
    translation.source_language = LanguageCode.KO
    translation.target_language = LanguageCode.EN
    rendering = {"prefix": "Rendered"}

    def respond(system: str, human: str) -> str:
        if "professional translator" in system:
            text = re.search(r"<text>\n(.*?)\n</text>", human, re.S)
            assert text
            return f"{rendering['prefix']} as follows: {text.group(1)}"
        return e2e._respond(system, human)

    calls.respond = respond
    source = corpus(tmp_path)
    context = ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    level_1 = [n for n in tree_nodes(providers) if n["level"] == 1]
    assert level_1
    assert all(n["text"].startswith("Summary: Rendered as") for n in level_1)

    indexer = providers.build_vector_indexer()
    try:
        stage = SummaryTreeStage(config, providers=providers)
        rows = indexer.scan_text_units(None, fields=["attributes"]) or []
        leaves = [
            leaf
            for group in stage._leaves(indexer, None, {None}, rows).values()
            for leaf in group
        ]
        stored = {
            str(row["id"]): row
            for row in indexer.scan_text_units(None, fields=["text", "n_tokens"]) or []
        }
    finally:
        indexer.close()
    assert leaves
    for leaf in leaves:
        assert leaf.text.startswith("Rendered as follows: ")
        assert leaf.n_tokens == stage.summarizer.count_tokens(leaf.text)
        assert leaf.n_tokens != stored[leaf.id]["n_tokens"]

    # Another translation model: the reset translates the same text units
    # (same ids) otherwise.
    rendering["prefix"] = "Restated"
    config.models.roles.translation = "synthetic-translator"
    config.indexing.reset = True
    context = ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    level_1 = [n for n in tree_nodes(providers) if n["level"] == 1]
    assert level_1
    assert all(n["text"].startswith("Summary: Restated as") for n in level_1)


def test_a_korean_corpus_is_summarized_in_korean_with_cjk_counts(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
    from unified_kg_rag.domain.models.config import LanguageCode

    config, providers, _, calls = setup(tmp_path)
    config.processing.translation.source_language = LanguageCode.KO
    config.processing.translation.target_language = LanguageCode.KO
    topics = [
        ("창고", "벤더를 위해 위젯을 보관한다"),
        ("운송사", "창고를 위해 가젯을 옮긴다"),
        ("구매자", "벤더에게 위젯을 주문한다"),
        ("은행", "구매자에게 돈을 빌려준다"),
        ("보험사", "창고의 화재를 보장한다"),
        ("감사인", "벤더의 장부를 점검한다"),
        ("항구", "운송사를 위해 컨테이너를 싣는다"),
        ("제철소", "벤더를 위해 강철을 만든다"),
    ]
    source = tmp_path / "corpus"
    source.mkdir()
    for index, (name, fact) in enumerate(topics):
        (source / f"doc{index}.txt").write_text(
            f"{name}는 {fact}.\n\n{name}는 북쪽에 있다."
        )
    context = ingest(config, providers, source, tmp_path / "run-1")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    nodes = tree_nodes(providers)
    assert nodes and all(n["language"] == "Korean" for n in nodes)
    # The prompt asks for Korean, its length in characters.
    assert calls.tree_prompts
    assert all(
        "in Korean" in p and "characters at most" in p for p in calls.tree_prompts
    )
    chain = GraphRAGChain(config, providers=providers)
    try:
        out = ask(chain, query="누가 벤더를 위해 위젯을 보관하나?")
        assert by_type(out, "summary")
        assert any("창고" in s["content"] for s in out.sources)
    finally:
        chain.close()


def test_the_retrieval_evaluator_scores_tree_with_a_summary_row(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    import asyncio

    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
    from unified_kg_rag.domain.models import (
        EvaluationGroundTruth,
        EvaluationQuery,
        EvaluatorType,
    )
    from unified_kg_rag.evaluation import EvaluationManager

    config, providers, _, _ = setup(tmp_path)
    context = ingest(config, providers, corpus(tmp_path), tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    config.evaluation.enabled_evaluators = [EvaluatorType.RETRIEVAL]
    config.evaluation.outputs_directory = str(tmp_path / "eval")
    chain = GraphRAGChain(config, providers=providers)
    try:
        manager = EvaluationManager(config, rag_chain=chain)
        results, _, summary = asyncio.run(
            manager.evaluate_dataset(
                [
                    EvaluationQuery(
                        query_id="q1",
                        question=QUESTION,
                        metadata={"search_strategy": "tree"},
                    )
                ],
                [
                    EvaluationGroundTruth(
                        query_id="q1",
                        ground_truth=e2e.ANSWER,
                        reference_sources=["depot.txt"],
                    )
                ],
                show_progress=False,
            )
        )
    finally:
        chain.close()
    assert "summary" in results[0].retrieved_source_types
    assert summary.metric_statistics["hit_at_k"]["mean"] == 1.0
    assert "summary" in summary.grouped_statistics["source_type"]


def test_a_partial_tree_is_reported_to_the_metrics_sink(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # A failed summary moves its children up a level without failing the
    # run: the operator learns of it only from this metric (the dashboard's
    # summary-tree widget reads it).
    config, providers, server, calls = setup(tmp_path)
    failed: list[int] = []

    def flaky(system: str, human: str) -> str:
        if SUMMARY_TREE_PROMPT in system and not failed:
            failed.append(1)
            return ""  # answered nothing: the summary failed
        return calls(system, human)

    server.respond = flaky

    class Sink:
        def __init__(self) -> None:
            self.emitted: dict[str, float] = {}

        def emit(self, namespace: str, metrics: dict, dimensions: dict) -> None:
            self.emitted.update(metrics)

    sink = Sink()
    cache = tmp_path / "run-1"
    pipeline = DataIngestionPipeline(
        config,
        PipelineConfig(
            stages_enabled=dict.fromkeys(PipelineStageType, True),
            local_directory=cache,
        ),
        source_directory=corpus(tmp_path),
        providers=providers,
        metrics_sink=sink,
    )
    try:
        context = pipeline.run(corpus(tmp_path), pipeline_id=cache.name)
    finally:
        pipeline.close()

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert failed
    assert sink.emitted["total_tree_summaries_failed"] >= 1
    assert sink.emitted["total_tree_uncovered_leaves"] == 0
    tree = tree_record(providers)
    assert tree is not None and tree["partial"] is True
