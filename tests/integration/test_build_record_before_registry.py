# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A run killed after its registry write leaves its build record (AWS-free).

The indexing stage wrote the suffix's build record after the registry
recorded the documents. A run killed in between (a first build, or a reset,
whose new indices hold no record) left the registry naming every document
indexed and no record: the next run found nothing to index, wrote no record
either, and every later query read the namespace as holding whatever its
config builds (the methodology checks fail open without a record). The
record is now written once the writes are judged recorded, before the
registry write.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401 - a fixture
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.application.cli.run_ingestion_pipeline import _build_record
from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
from unified_kg_rag.application.storage.inspection import describe_namespace
from unified_kg_rag.domain.models import Methodology, PipelineStageStatus

pytestmark = pytest.mark.integration


class _Killed(BaseException):
    """A process death: no ``except Exception`` handler absorbs it."""


def _record(config, providers) -> dict | None:  # noqa: ANN001
    vectors = providers.build_vector_indexer()
    try:
        return vectors.read_build_record(None)
    finally:
        vectors.close()


@pytest.mark.usefixtures("no_aws")
@pytest.mark.parametrize("reset", [False, True], ids=["first-build", "reset"])
def test_a_run_killed_after_its_registry_write_leaves_its_record(
    tmp_path: Path, mocker, reset: bool
) -> None:
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    providers = e2e._providers(
        config, FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM), e2e.FakeCrossEncoder("r")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    if reset:
        context = e2e._ingest(config, providers, source, tmp_path / "run-0")
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        (source / "carrier.txt").write_text("Carrier ships widgets for Depot.")
        config.indexing.reset = True
    record_processed = IncrementalIndexer._record_processed

    def then_die(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        record_processed(self, *args, **kwargs)
        raise _Killed

    mocker.patch.object(IncrementalIndexer, "_record_processed", then_die)
    with pytest.raises(_Killed):
        e2e._ingest(config, providers, source, tmp_path / "run-1")
    mocker.stopall()

    killed = _record(config, providers)
    assert killed is not None
    assert killed["pipeline_id"] == "run-1"
    assert killed["mode"] == "full"
    config.indexing.reset = False
    context = e2e._ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert context.incremental_delta is not None
    assert context.incremental_delta.to_process == []
    assert _record(config, providers) == killed


def _setup(tmp_path: Path):  # noqa: ANN202
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    # A narrower profile than a query's default: without the record, a
    # query reads the namespace as holding GraphRAG's artifacts too.
    config.indexing.methodologies = [Methodology.LIGHTRAG]
    providers = e2e._providers(
        config, FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM), e2e.FakeCrossEncoder("r")
    )
    source = tmp_path / "corpus"
    source.mkdir(parents=True)
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    return config, providers, source


def _failing_record_writes(mocker, attempts: int) -> None:  # noqa: ANN001
    # The first ``attempts`` record writes fail on every index; the adapter
    # logs each and goes on (the record is best-effort).
    record_build = OpenSearchIndexer.record_build
    calls = {"n": 0}

    def denied(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        raise RuntimeError("mapping update denied")

    def flaky(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        calls["n"] += 1
        if calls["n"] > attempts:
            return record_build(self, *args, **kwargs)
        with mock.patch.object(OpenSearchIndexer, "_put_index_meta", denied):
            return record_build(self, *args, **kwargs)

    mocker.patch.object(OpenSearchIndexer, "record_build", flaky)


@pytest.mark.usefixtures("no_aws")
def test_a_record_the_store_rejected_is_written_again_after_the_registry(
    tmp_path: Path, mocker
) -> None:
    config, providers, source = _setup(tmp_path)
    # Every index of the first attempt fails; the second attempt succeeds.
    _failing_record_writes(mocker, attempts=1)

    context = e2e._ingest(config, providers, source, tmp_path / "run-1")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    record = _record(config, providers)
    assert record is not None
    assert record["pipeline_id"] == "run-1"


@pytest.mark.usefixtures("no_aws")
def test_a_missing_record_is_written_by_a_run_with_nothing_to_index(
    tmp_path: Path, mocker
) -> None:
    # A store that never took the record (or a build from before records):
    # without one, every query reads the namespace as holding what its
    # config builds. The next run writes it, even with nothing to index.
    config, providers, source = _setup(tmp_path / "control")
    e2e._ingest(config, providers, source, tmp_path / "control" / "run-1")
    want = _record(config, providers)
    assert want is not None
    config, providers, source = _setup(tmp_path)
    _failing_record_writes(mocker, attempts=1000)
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert _record(config, providers) is None
    # Visible to automation: the run result and run-describe say so.
    assert _build_record(context) == "failed"
    assert describe_namespace(config, providers=providers).build_record == "missing"
    mocker.stopall()

    context = e2e._ingest(config, providers, source, tmp_path / "run-2")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert context.incremental_delta is not None
    assert context.incremental_delta.to_process == []
    assert _build_record(context) == "written"
    assert describe_namespace(config, providers=providers).build_record == "present"
    record = _record(config, providers)
    assert record is not None
    assert _build_record(e2e._ingest(config, providers, source, tmp_path / "r3")) == (
        "kept"
    )
    assert record["methodologies"] == want["methodologies"] == ["lightrag", "hipporag"]
    assert record["artifacts"] == want["artifacts"]
    # Written by a delta that indexed nothing: the build the stores hold,
    # counting the documents the registry records.
    assert (record["mode"], record["documents"]) == ("full", 2)


@pytest.mark.usefixtures("no_aws")
def test_a_rewritten_record_names_only_what_the_stores_hold(
    tmp_path: Path, mocker
) -> None:
    # Built under LightRAG alone, the record rejected; the next run, with
    # nothing to index, runs the default profile. Its record must not name
    # GraphRAG's communities and reports, which no run built: a global query
    # would then search an empty reports index instead of being refused.
    config, providers, source = _setup(tmp_path)
    _failing_record_writes(mocker, attempts=1000)
    e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert _record(config, providers) is None
    mocker.stopall()
    config.indexing.methodologies = [Methodology.GRAPHRAG, Methodology.LIGHTRAG]

    context = e2e._ingest(config, providers, source, tmp_path / "run-2")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert context.incremental_delta is not None
    assert context.incremental_delta.to_process == []
    record = _record(config, providers)
    assert record is not None
    assert "graphrag" not in record["methodologies"]
    assert not {"communities", "community_reports"} & set(record["artifacts"])
    assert {"text_units", "entities", "relationship_vectors", "graph"} <= set(
        record["artifacts"]
    )
    # The documents the registry records, not the none this run indexed.
    assert (record["mode"], record["documents"]) == ("full", 2)


@pytest.mark.usefixtures("no_aws")
def test_a_rewritten_record_is_the_full_build_later_records_carry(
    tmp_path: Path, mocker
) -> None:
    # Under the default profile: the rewritten record starts the community
    # counters from the reports stored, and the next incremental record
    # carries it as its full build (the corpus size survives).
    config, providers, source = _setup(tmp_path)
    config.indexing.methodologies = [Methodology.GRAPHRAG, Methodology.LIGHTRAG]
    _failing_record_writes(mocker, attempts=1000)
    e2e._ingest(config, providers, source, tmp_path / "run-1")
    mocker.stopall()
    e2e._ingest(config, providers, source, tmp_path / "run-2")
    rewritten = _record(config, providers)
    assert rewritten is not None
    assert (rewritten["mode"], rewritten["documents"]) == ("full", 2)
    reports = next(
        entry["documents"]
        for entry in providers.build_vector_indexer().describe_indices(None)
        if entry["kind"] == "community_reports"
    )
    assert reports
    assert rewritten["communities"]["communities_now"] == reports
    assert rewritten["communities"]["runs_since"] == 0
    (source / "carrier.txt").write_text("Carrier ships widgets for Depot.")

    e2e._ingest(config, providers, source, tmp_path / "run-3")

    record = _record(config, providers)
    assert record is not None
    assert (record["mode"], record["documents"]) == ("incremental", 1)
    assert record["full_build"]["pipeline_id"] == "run-2"
    assert record["full_build"]["documents"] == 2
    assert record["communities"]["runs_since"] == 1


@pytest.mark.usefixtures("no_aws")
def test_a_rewritten_record_keeps_the_claims_a_build_found_none_of(
    tmp_path: Path, mocker
) -> None:
    # Claim extraction on, a corpus stating no claim: no claims index. The
    # rewritten record left claims out, so the next delta rebuilt the whole
    # namespace for them (3 documents processed instead of 1).
    config, providers, source = _setup(tmp_path)
    config.indexing.methodologies = [Methodology.GRAPHRAG, Methodology.LIGHTRAG]
    config.processing.claim_extraction.enabled = True
    _failing_record_writes(mocker, attempts=1000)
    e2e._ingest(config, providers, source, tmp_path / "run-1")
    mocker.stopall()
    e2e._ingest(config, providers, source, tmp_path / "run-2")
    rewritten = _record(config, providers)
    assert rewritten is not None
    assert {"claims", "relationship_vectors"} <= set(rewritten["artifacts"])
    (source / "carrier.txt").write_text("Carrier ships widgets for Depot.")

    context = e2e._ingest(config, providers, source, tmp_path / "run-3")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert context.incremental_delta is not None
    assert len(context.incremental_delta.to_process) == 1
    record = _record(config, providers)
    assert record is not None
    assert (record["mode"], record["documents"]) == ("incremental", 1)


@pytest.mark.usefixtures("no_aws")
def test_a_record_one_index_rejected_is_written_again_in_the_run(
    tmp_path: Path, mocker
) -> None:
    # One index refuses the first write; the others take it. The run writes
    # it again after the registry, as its log says, so no index is left
    # holding no record (or an earlier one).
    config, providers, source = _setup(tmp_path)
    put = OpenSearchIndexer._put_index_meta
    refused: list[str] = []

    def once(self, name, meta):  # noqa: ANN001, ANN202
        if "entities" in name and "build" in meta and not refused:
            refused.append(name)
            raise RuntimeError("mapping update denied")
        return put(self, name, meta)

    mocker.patch.object(OpenSearchIndexer, "_put_index_meta", once)

    context = e2e._ingest(config, providers, source, tmp_path / "run-1")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert refused
    assert _build_record(context) == "written"
    vectors = providers.build_vector_indexer()
    try:
        builds = {
            entry["kind"]: (entry["meta"].get("build") or {}).get("pipeline_id")
            for entry in vectors.describe_indices(None)
            if entry["index"]
        }
    finally:
        vectors.close()
    assert builds and set(builds.values()) == {"run-1"}, builds
