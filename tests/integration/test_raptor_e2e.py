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

from tests.fixtures.fakes.chat_models import CountingResponder
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
    source = corpus(tmp_path)
    # Half the documents are of category "north", half "south", as
    # pre-parsed documents carrying their filters.
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    for index, path in enumerate(sorted(source.glob("*.txt"))):
        (parsed / f"{path.stem}.json").write_text(
            json.dumps(
                {
                    "content": path.read_text(),
                    "file_name": path.name,
                    "file_path": str(path),
                    "metadata": {
                        "filters": {"category": "north" if index % 2 else "south"}
                    },
                }
            )
        )
        path.unlink()
    context = ingest(config, providers, parsed, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    nodes = tree_nodes(providers)
    mixed = [
        n for n in nodes if "category" not in (n["attributes"].get("filters") or {})
    ]
    pure = [n for n in nodes if (n["attributes"].get("filters") or {}).get("category")]
    assert nodes and (mixed or pure)

    chain = GraphRAGChain(config, providers=providers)
    try:
        out = ask(chain, filters={"attr_category": "north"})
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
        if TREE_PROMPT in system and not failed:
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
