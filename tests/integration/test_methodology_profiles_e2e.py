# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Methodology profiles end to end on the embedded stores (no AWS, no
container; docs/design.md#46-methodology-profiles).

A LightRAG-only ingestion runs no community detection and makes no report
call; its strategies answer and GraphRAG's are refused before any model
call. Adding graphrag later rebuilds the namespace from the stored
extractions: the only new model calls are the community reports. Removing it
stops maintaining the reports without deleting them; a reset with the
remaining methodology replaces them.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from tests.fixtures.fakes.chat_models import CountingResponder
from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401 - fixture
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    IndexBuildRecord,
    Methodology,
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
    SearchStrategy,
)
from unified_kg_rag.shared import (
    ConfigurationError,
    ConfirmationDeclinedError,
    MethodologyNotIndexedError,
)

pytestmark = pytest.mark.integration

GRAPHRAG = (
    SearchStrategy.LOCAL,
    SearchStrategy.GLOBAL,
    SearchStrategy.DRIFT,
)
# iterative over its default base, mix, reads lightrag; simple sweeps the
# indexes the namespace holds.
LIGHTRAG = (
    SearchStrategy.MIX,
    SearchStrategy.HYBRID,
    SearchStrategy.NAIVE,
    SearchStrategy.ITERATIVE,
    SearchStrategy.SIMPLE,
)


def _corpus(root: Path) -> Path:
    source = root / "corpus"
    source.mkdir(parents=True)
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    (source / "freight.txt").write_text("Carrier moves gadgets for Depot.")
    return source


def _setup(
    tmp_path: Path, *, registry: bool = True
) -> tuple[Config, Providers, FakeOllama, CountingResponder]:
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.providers.doc_status.enabled = registry
    calls = CountingResponder(e2e._respond)
    server = FakeOllama(calls, e2e.ANSWER, e2e.DIM)
    encoder = e2e.FakeCrossEncoder("synthetic-reranker")
    return config, e2e._providers(config, server, encoder), server, calls


def _ran(context: PipelineContext) -> set[str]:
    return {
        r.stage_name
        for r in context.stage_results
        if r.status is PipelineStageStatus.COMPLETED
    }


def _record(providers: Providers) -> dict:
    indexer = providers.build_vector_indexer()
    try:
        record = indexer.read_build_record(None)
    finally:
        indexer.close()
    assert record is not None
    return record


def _reports(config: Config, providers: Providers) -> list[dict]:
    prefix = config.indexing.opensearch.community_reports_index_prefix
    try:
        return e2e._stored(config, providers, prefix)
    except Exception:  # noqa: BLE001 - no such index
        return []


def _community_vertices(config: Config) -> int:
    """The community vertices the SQLite graph holds."""
    prefix = config.indexing.neptune.community_label_prefix
    with sqlite3.connect(config.providers.graph.sqlite_path) as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM vertices WHERE label LIKE ?", (f"{prefix}%",)
        ).fetchone()[0]


def _ask(chain: GraphRAGChain, strategy: SearchStrategy) -> RAGOutput:
    out = chain.invoke(
        RAGInput(query="Who stores the gadgets for Vendor?", search_strategy=strategy)
    )
    assert isinstance(out, RAGOutput)
    return out


def test_a_lightrag_only_namespace_builds_less_and_gains_graphrag_later(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    config, providers, server, calls = _setup(tmp_path)
    config.indexing.methodologies = [Methodology.LIGHTRAG]
    source = _corpus(tmp_path)

    # LightRAG only: no community detection, no report call.
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert "community_detection" not in _ran(context)
    light_calls = calls.since(Counter())
    assert light_calls["extraction"] > 0 and light_calls["report"] == 0
    record = _record(providers)
    assert record["methodologies"] == ["lightrag", "hipporag"]
    assert "community_reports" not in record["artifacts"]

    # The same corpus with the default methodologies in another namespace
    # pays the reports on top: the LightRAG-only build cost less.
    full_config, full_providers, _, full_calls = _setup(tmp_path / "full")
    full_source = _corpus(tmp_path / "full")
    context = e2e._ingest(full_config, full_providers, full_source, tmp_path / "f")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert full_calls.kinds["report"] > 0
    assert full_calls.kinds["extraction"] == light_calls["extraction"]
    assert sum(full_calls.kinds.values()) > sum(light_calls.values())

    # Its strategies answer; GraphRAG's are refused before any model call.
    chain = GraphRAGChain(config, providers=providers)
    try:
        for strategy in LIGHTRAG:
            out = _ask(chain, strategy)
            assert out.answer == e2e.ANSWER, strategy
            assert out.sources, strategy
        for strategy in GRAPHRAG:
            before = len(server.chat_models)
            with pytest.raises(MethodologyNotIndexedError) as refused:
                _ask(chain, strategy)
            assert len(server.chat_models) == before, strategy
            message = str(refused.value)
            assert "reads the graphrag methodology" in message
            assert "holds lightrag, hipporag" in message
            assert "mix, hybrid, naive" in message
            assert "indexing.methodologies" in message
        # auto has one strategy left of local, mix and global: no router call.
        out = _ask(chain, SearchStrategy.AUTO)
        assert out.metadata["search_strategy"] == "mix"
        # iterative over local reads graphrag: refused there.
        over_local = config.model_copy(deep=True)
        over_local.search.iterative_search.base = "local"
        local_chain = GraphRAGChain(over_local, providers=providers)
        try:
            before = len(server.chat_models)
            with pytest.raises(MethodologyNotIndexedError, match="'iterative'"):
                _ask(local_chain, SearchStrategy.ITERATIVE)
            assert len(server.chat_models) == before
        finally:
            local_chain.close()

        # Adding graphrag: the run rebuilds the namespace from the stored
        # extractions; the community reports are its only new model calls.
        config.indexing.methodologies = [Methodology.GRAPHRAG, Methodology.LIGHTRAG]
        before = Counter(calls.kinds)
        context = e2e._ingest(config, providers, source, tmp_path / "run-2")
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        assert "community_detection" in _ran(context)
        added = calls.since(before)
        assert added["report"] > 0
        assert added["extraction"] == 0
        assert set(added) == {"report"}
        assert config.indexing.reset is False  # set for that run only
        assert _record(providers)["methodologies"] == [
            "graphrag",
            "lightrag",
            "hipporag",
        ]
        assert _reports(config, providers)
        assert _community_vertices(config) > 0
        # The chain reads the record again before refusing: global answers.
        for strategy in (SearchStrategy.GLOBAL, SearchStrategy.LOCAL):
            assert _ask(chain, strategy).answer == e2e.ANSWER, strategy

        # Removing graphrag: the next run that writes the namespace stops
        # maintaining the reports and records them gone, deleting nothing.
        config.indexing.methodologies = [Methodology.LIGHTRAG]
        (source / "freight.txt").write_text("Carrier moves gadgets for Vendor.")
        context = e2e._ingest(config, providers, source, tmp_path / "run-3")
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        assert "community_detection" not in _ran(context)
        assert _record(providers)["methodologies"] == ["lightrag", "hipporag"]
        assert _reports(config, providers)
        assert _community_vertices(config) > 0
    finally:
        chain.close()

    # A new chain reads the record: GraphRAG's strategies are refused again.
    chain = GraphRAGChain(config, providers=providers)
    try:
        with pytest.raises(MethodologyNotIndexedError):
            _ask(chain, SearchStrategy.GLOBAL)
    finally:
        chain.close()

    # A reset with the remaining methodology replaces the reports index with
    # an empty one.
    config.indexing.reset = True
    context = e2e._ingest(config, providers, source, tmp_path / "run-4")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    prefix = config.indexing.opensearch.community_reports_index_prefix
    assert e2e._stored(config, providers, prefix) == []
    assert _community_vertices(config) == 0


def test_a_resumed_pipeline_id_runs_only_the_added_stages(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # Without the registry, a methodology added to a pipeline id runs only
    # the stages it adds (and indexing, downstream of them) from the stage
    # cache.
    config, providers, _, calls = _setup(tmp_path, registry=False)
    config.indexing.methodologies = [Methodology.LIGHTRAG]
    source = _corpus(tmp_path)
    cache = tmp_path / "run"
    context = e2e._ingest(config, providers, source, cache)
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert "community_detection" not in _ran(context)

    config.indexing.methodologies = [Methodology.GRAPHRAG, Methodology.LIGHTRAG]
    before = Counter(calls.kinds)
    pipeline = DataIngestionPipeline(
        config,
        PipelineConfig(
            stages_enabled=dict.fromkeys(PipelineStageType, True),
            local_directory=cache,
        ),
        source_directory=source,
        providers=providers,
    )
    try:
        context = pipeline.run(source, pipeline_id=cache.name)
    finally:
        pipeline.close()
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert set(calls.since(before)) == {"report"}
    assert pipeline.executed_stages == {"community_detection", "indexing"}
    assert _record(providers)["methodologies"] == ["graphrag", "lightrag", "hipporag"]


def test_a_lightrag_only_run_exports_its_graph_without_communities(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    import json

    config, providers, _, _ = _setup(tmp_path)
    config.indexing.methodologies = [Methodology.LIGHTRAG]
    context = e2e._ingest(config, providers, _corpus(tmp_path), tmp_path / "run")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    (path,) = tmp_path.rglob("visualization_data.json")
    data = json.loads(path.read_text())
    assert data["nodes"] and data["edges"]
    assert data["communities"] == {}


def test_report_generation_off_keeps_incremental_runs_incremental(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # A run that never builds community reports does not take the
    # namespace for missing them: the next run is incremental, not a rebuild.
    config, providers, _, calls = _setup(tmp_path)
    config.graph.community_detection.report_generation.enabled = False
    source = _corpus(tmp_path)
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    record = _record(providers)
    assert "community_reports" not in record["artifacts"]
    # It holds graphrag: local answers, global and drift are refused before
    # any model call, naming the switch.
    assert "graphrag" in record["methodologies"]
    chain = GraphRAGChain(config, providers=providers)
    try:
        out = _ask(chain, SearchStrategy.LOCAL)
        assert out.answer == e2e.ANSWER and out.sources
        for strategy in (SearchStrategy.GLOBAL, SearchStrategy.DRIFT):
            with pytest.raises(
                MethodologyNotIndexedError, match="report_generation.enabled: true"
            ):
                _ask(chain, strategy)
    finally:
        chain.close()

    (source / "notes.txt").write_text("Bank lends money to Vendor.")
    before = Counter(calls.kinds)
    context = e2e._ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert _record(providers)["mode"] == "incremental"
    # Only the new document is extracted.
    assert calls.since(before) == Counter({"extraction": 1})


def test_adding_a_methodology_asks_before_its_rebuild_deletes_most_documents(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # The rebuild that adds graphrag deletes the documents gone from the
    # corpus (a partial sync, say) as an incremental run would: it is asked
    # first, past indexing.max_deletion_share, and nothing is written.
    config, providers, server, _ = _setup(tmp_path)
    config.indexing.methodologies = [Methodology.LIGHTRAG]
    source = tmp_path / "corpus"
    source.mkdir()
    names = [f"party{i}" for i in range(12)]
    for i, name in enumerate(names):
        (source / f"{name}.txt").write_text(
            f"Party{i} trades widgets with Party{(i + 1) % 12}."
        )
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    record = _record(providers)
    for name in names[3:]:
        (source / f"{name}.txt").unlink()

    config.indexing.methodologies = [Methodology.GRAPHRAG, Methodology.LIGHTRAG]
    calls = len(server.chat_models)
    with pytest.raises(ConfirmationDeclinedError, match="deletes 9 of the 12"):
        e2e._ingest(config, providers, source, tmp_path / "add")
    assert len(server.chat_models) == calls
    assert config.indexing.reset is False
    assert _record(providers) == record
    assert len(e2e._registry(providers)) == 12


def test_adding_a_methodology_never_rebuilds_over_another_scope(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # A rebuild from one corpus deletes the namespace's other source scopes:
    # the run is refused before any model call instead, and writes nothing.
    config, providers, server, _ = _setup(tmp_path)
    config.indexing.methodologies = [Methodology.LIGHTRAG]
    first, second = _corpus(tmp_path / "a"), tmp_path / "b" / "corpus"
    second.mkdir(parents=True)
    (second / "notes.txt").write_text("Bank lends money to Vendor.")
    for source in (first, second):
        cache = tmp_path / f"run-{source.parent.name}"
        context = e2e._ingest(config, providers, source, cache)
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    record = _record(providers)
    scopes = sorted(str(source.resolve()) for source in (first, second))
    assert record["source_scopes"] == scopes

    config.indexing.methodologies = [Methodology.GRAPHRAG, Methodology.LIGHTRAG]
    # The query-time advice for such a namespace is no rebuild either.
    chain = GraphRAGChain(config, providers=providers)
    try:
        with pytest.raises(MethodologyNotIndexedError) as refused:
            _ask(chain, SearchStrategy.GLOBAL)
    finally:
        chain.close()
    assert f"source scopes {', '.join(scopes)}" in str(refused.value)
    assert "rebuilt once" not in str(refused.value)
    calls = len(server.chat_models)
    with pytest.raises(ConfigurationError, match="also holds documents of source"):
        e2e._ingest(config, providers, first, tmp_path / "add")
    assert len(server.chat_models) == calls
    assert config.indexing.reset is False
    assert _record(providers) == record
    assert e2e._registry(providers) == {
        "contract.txt": DocStatus.PROCESSED,
        "logistics.txt": DocStatus.PROCESSED,
        "freight.txt": DocStatus.PROCESSED,
        "notes.txt": DocStatus.PROCESSED,
    }


def test_fewer_methodologies_beside_another_scope_are_refused(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # Regression: a run of a second corpus with fewer methodologies dropped
    # graphrag from the namespace's build record, for the first corpus's
    # documents too; the query advised a rebuild, which was then refused.
    # The run is refused before any model call instead, and writes nothing.
    config, providers, server, _ = _setup(tmp_path)
    first, second = _corpus(tmp_path / "a"), tmp_path / "b" / "corpus"
    second.mkdir(parents=True)
    (second / "notes.txt").write_text("Bank lends money to Vendor.")
    context = e2e._ingest(config, providers, first, tmp_path / "run-a")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    record = _record(providers)
    assert record["source_scopes"] == [str(first.resolve())]

    def refused() -> None:
        calls = len(server.chat_models)
        with pytest.raises(ConfigurationError, match="leaves out graphrag") as error:
            e2e._ingest(config, providers, second, tmp_path / "run-b")
        message = str(error.value)
        assert str(first.resolve()) in message
        for way_out in (
            "indexing.methodologies: graphrag, lightrag, hipporag",
            "processing.document_parsing.index_value",
            "indexing.reset: true",
        ):
            assert way_out in message
        assert len(server.chat_models) == calls
        assert set(e2e._registry(providers)) == {
            "contract.txt",
            "logistics.txt",
            "freight.txt",
        }

    config.indexing.methodologies = [Methodology.LIGHTRAG]
    refused()
    assert _record(providers) == record
    # A record written before source scopes were recorded: the registry
    # tells the scopes, so it is refused all the same.
    indexer = providers.build_vector_indexer()
    try:
        old = IndexBuildRecord.model_validate(record)
        indexer.record_build(old.model_copy(update={"source_scopes": None}), None)
    finally:
        indexer.close()
    refused()

    # With the namespace's methodologies, the second corpus is added.
    config.indexing.methodologies = [Methodology.GRAPHRAG, Methodology.LIGHTRAG]
    context = e2e._ingest(config, providers, second, tmp_path / "run-b2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    record = _record(providers)
    assert "graphrag" in record["methodologies"]
    assert record["source_scopes"] == sorted(
        str(source.resolve()) for source in (first, second)
    )
    # A run that retires the other scope narrows its own alone.
    config.indexing.methodologies = [Methodology.LIGHTRAG]
    config.indexing.retire_source_scopes = [str(first.resolve())]
    context = e2e._ingest(config, providers, second, tmp_path / "run-b3")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    record = _record(providers)
    assert record["methodologies"] == ["lightrag", "hipporag"]
    assert record["source_scopes"] == [str(second.resolve())]
    assert set(e2e._registry(providers)) == {"notes.txt"}


def test_without_the_registry_a_new_run_reuses_the_stored_extractions(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # Every run is a full build: one under a new pipeline id that adds
    # graphrag takes the extraction store's records, so its only new model
    # calls are the community reports.
    config, providers, _, calls = _setup(tmp_path, registry=False)
    config.indexing.methodologies = [Methodology.LIGHTRAG]
    source = _corpus(tmp_path)
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    config.indexing.methodologies = [Methodology.GRAPHRAG, Methodology.LIGHTRAG]
    before = Counter(calls.kinds)
    context = e2e._ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert set(calls.since(before)) == {"report"}
    assert _record(providers)["methodologies"] == ["graphrag", "lightrag", "hipporag"]


def test_removing_a_methodology_is_recorded_by_a_run_with_no_change(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # Regression (real AWS): a state-machine run under a narrower profile
    # with an unchanged corpus wrote no build record, so the removed
    # methodology's strategies kept being served on artifacts no run
    # maintains any more, until a run with a delta.
    config, providers, server, calls = _setup(tmp_path)
    source = _corpus(tmp_path)
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert "graphrag" in _record(providers)["methodologies"]

    config.indexing.methodologies = [Methodology.LIGHTRAG]
    before = Counter(calls.kinds)
    context = e2e._ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert calls.since(before) == Counter()  # nothing changed, nothing asked
    record = _record(providers)
    assert record["methodologies"] == ["lightrag", "hipporag"]
    assert "community_reports" not in record["artifacts"]
    # One source scope: narrowing is allowed, and the advice is a re-run.
    assert record["source_scopes"] == [str(source.resolve())]
    chain = GraphRAGChain(config, providers=providers)
    try:
        with pytest.raises(MethodologyNotIndexedError) as refused:
            _ask(chain, SearchStrategy.LOCAL)
        assert f"source scope '{source.resolve()}'" in str(refused.value)
        assert "rebuilt once from its stored extractions" in str(refused.value)
        assert _ask(chain, SearchStrategy.MIX).answer == e2e.ANSWER
    finally:
        chain.close()

    # A run with no change under the same profile leaves the record as it is.
    context = e2e._ingest(config, providers, source, tmp_path / "run-3")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert _record(providers) == record


def test_a_reset_to_fewer_methodologies_keeps_the_extraction_records(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # Regression (real AWS): a raptor-only reset of a LightRAG-built
    # namespace emptied the extraction-records index (8 -> 0), so adding
    # lightrag back extracted every document again instead of rebuilding
    # from the stored extractions.
    config, providers, _, calls = _setup(tmp_path)
    config.indexing.methodologies = [Methodology.LIGHTRAG]
    source = _corpus(tmp_path)
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    prefix = config.indexing.opensearch.extraction_records_index_prefix
    records = len(e2e._stored(config, providers, prefix))
    assert records > 0

    config.indexing.methodologies = [Methodology.RAPTOR]
    config.indexing.reset = True
    context = e2e._ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert _record(providers)["methodologies"] == ["raptor"]
    assert len(e2e._stored(config, providers, prefix)) == records

    # Adding lightrag back rebuilds from the stored extractions: no
    # extraction call.
    config.indexing.reset = False
    config.indexing.methodologies = [Methodology.LIGHTRAG, Methodology.RAPTOR]
    before = Counter(calls.kinds)
    context = e2e._ingest(config, providers, source, tmp_path / "run-3")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert calls.since(before)["extraction"] == 0
    assert "lightrag" in _record(providers)["methodologies"]

    # A reset that builds what the namespace holds still drops the records
    # no stored text unit names (a deleted document's).
    (source / "freight.txt").unlink()
    config.indexing.reset = True
    context = e2e._ingest(config, providers, source, tmp_path / "run-4")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert 0 < len(e2e._stored(config, providers, prefix)) < records


def test_a_second_raptor_only_reset_keeps_the_extraction_records(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # Regression: the first raptor-only reset kept the records (its stored
    # build record named lightrag's artifacts), the second one found only
    # raptor in the record and pruned every one of them (3 -> 3 -> 0).
    config, providers, _, calls = _setup(tmp_path)
    config.indexing.methodologies = [Methodology.LIGHTRAG]
    source = _corpus(tmp_path)
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    prefix = config.indexing.opensearch.extraction_records_index_prefix
    records = len(e2e._stored(config, providers, prefix))
    assert records > 0

    config.indexing.methodologies = [Methodology.RAPTOR]
    config.indexing.reset = True
    for run in ("run-2", "run-3"):
        context = e2e._ingest(config, providers, source, tmp_path / run)
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        assert _record(providers)["methodologies"] == ["raptor"]
        assert len(e2e._stored(config, providers, prefix)) == records

    config.indexing.reset = False
    config.indexing.methodologies = [Methodology.LIGHTRAG, Methodology.RAPTOR]
    before = Counter(calls.kinds)
    context = e2e._ingest(config, providers, source, tmp_path / "run-4")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert calls.since(before)["extraction"] == 0


def test_a_reset_after_narrowing_with_no_change_keeps_the_extraction_records(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # Regression: an incremental raptor-only run with no change narrows the
    # build record to raptor; the reset after it then pruned every record
    # (3 -> 0), so adding lightrag back extracted every document again.
    config, providers, _, calls = _setup(tmp_path)
    config.indexing.methodologies = [Methodology.LIGHTRAG]
    source = _corpus(tmp_path)
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    prefix = config.indexing.opensearch.extraction_records_index_prefix
    records = len(e2e._stored(config, providers, prefix))
    assert records > 0

    config.indexing.methodologies = [Methodology.RAPTOR]
    context = e2e._ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert _record(providers)["methodologies"] == ["raptor"]

    config.indexing.reset = True
    context = e2e._ingest(config, providers, source, tmp_path / "run-3")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert len(e2e._stored(config, providers, prefix)) == records

    config.indexing.reset = False
    config.indexing.methodologies = [Methodology.LIGHTRAG, Methodology.RAPTOR]
    before = Counter(calls.kinds)
    context = e2e._ingest(config, providers, source, tmp_path / "run-4")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert calls.since(before)["extraction"] == 0
    assert "lightrag" in _record(providers)["methodologies"]


def test_two_lightrag_only_resets_keep_the_report_answers(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # Regression: the first lightrag-only reset kept every record (its stored
    # build record named the reports), the second found no report in the
    # record and pruned the records to what its text units name, the report
    # answers too: adding graphrag back asked a report again.
    config, providers, _, calls = _setup(tmp_path)
    source = _corpus(tmp_path)
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert calls.kinds["report"] > 0
    prefix = config.indexing.opensearch.extraction_records_index_prefix
    records = len(e2e._stored(config, providers, prefix))

    config.indexing.methodologies = [Methodology.LIGHTRAG]
    config.indexing.reset = True
    for run in ("run-2", "run-3"):
        context = e2e._ingest(config, providers, source, tmp_path / run)
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        assert "graphrag" not in _record(providers)["methodologies"]
        assert len(e2e._stored(config, providers, prefix)) == records

    config.indexing.reset = False
    config.indexing.methodologies = [Methodology.GRAPHRAG, Methodology.LIGHTRAG]
    before = Counter(calls.kinds)
    context = e2e._ingest(config, providers, source, tmp_path / "run-4")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert calls.since(before)["extraction"] == 0
    assert calls.since(before)["report"] == 0
    assert _reports(config, providers)


def test_two_claims_off_resets_keep_the_claim_records(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # Regression: as above for claims: the second claims-off reset pruned
    # the claim records, so turning claims back on asked every unit again.
    from tests.integration.test_claim_convergence import _respond

    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.processing.claim_extraction.enabled = True
    asked = Counter[str]()

    def respond(system: str, human: str) -> str:
        if "claim extraction specialist" in system:
            asked["claims"] += 1
        return _respond(system, human)

    server = FakeOllama(CountingResponder(respond), e2e.ANSWER, e2e.DIM)
    providers = e2e._providers(config, server, e2e.FakeCrossEncoder("r"))
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "a.txt").write_text("Acme supplies Buyer.")
    (source / "b.txt").write_text("Globex pays Depot and Acme.")
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert asked["claims"] > 0

    config.processing.claim_extraction.enabled = False
    config.indexing.reset = True
    for run in ("run-2", "run-3"):
        context = e2e._ingest(config, providers, source, tmp_path / run)
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)

    config.indexing.reset = False
    config.processing.claim_extraction.enabled = True
    before = Counter(asked)
    context = e2e._ingest(config, providers, source, tmp_path / "run-4")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert asked - before == Counter()
