# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""When ingestion writes ``visualization_data.json`` (AWS-free).

Every run that ran the indexing stage exported the namespace's whole graph
while it held the namespace's lease, a run that found nothing to index
included: at 10k text units a no-change run went from 2 s to 41 s and
rewrote (and uploaded) a 59 MB file. The export now runs once the lease is
given up, a run that wrote nothing keeps an export of the build the stores
hold, and only a full build writes one (rv24 perf 2: a one-document
incremental run spent 36% of its time exporting 3,200 entities);
run-visualization builds it from the stores after an incremental run.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401 - a fixture
from unified_kg_rag.application.ingestion import pipeline as pipeline_module
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.ingestion.namespace_lease import NamespaceLease
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.application.storage.visualization_export import (
    export_build_time,
)
from unified_kg_rag.domain.ingestion.visualization_data import (
    VISUALIZATION_DATA_FILENAME,
)
from unified_kg_rag.domain.models import (
    PipelineConfig,
    PipelineStageStatus,
    PipelineStageType,
)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]


def _setup(tmp_path: Path):  # noqa: ANN202
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    providers = e2e._providers(
        config, FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM), e2e.FakeCrossEncoder("r")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    return config, providers, source


def _exports(mocker) -> list[Path]:  # noqa: ANN001
    write = pipeline_module.write_visualization_data
    paths: list[Path] = []

    def recording(providers, suffix, path):  # noqa: ANN001, ANN202
        paths.append(path)
        return write(providers, suffix, path)

    mocker.patch.object(pipeline_module, "write_visualization_data", recording)
    return paths


def test_a_run_that_wrote_nothing_keeps_the_export(tmp_path: Path, mocker) -> None:
    config, providers, source = _setup(tmp_path)
    exports = _exports(mocker)
    e2e._ingest(config, providers, source, tmp_path / "run-1")
    path = tmp_path / "viz" / VISUALIZATION_DATA_FILENAME
    assert exports == [path] and export_build_time(path) is not None
    before = path.stat().st_mtime_ns

    context = e2e._ingest(config, providers, source, tmp_path / "run-2")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert exports == [path]
    assert path.stat().st_mtime_ns == before
    assert context.visualization_data_path == str(path)


def test_an_incremental_run_that_wrote_exports_nothing(tmp_path: Path, mocker) -> None:
    config, providers, source = _setup(tmp_path)
    exports = _exports(mocker)
    e2e._ingest(config, providers, source, tmp_path / "run-1")
    path = tmp_path / "viz" / VISUALIZATION_DATA_FILENAME
    first = export_build_time(path)
    (source / "carrier.txt").write_text("Carrier ships widgets for Depot.")

    context = e2e._ingest(config, providers, source, tmp_path / "run-2")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert exports == [path]
    assert export_build_time(path) == first
    assert context.visualization_data_path is None


def test_a_reset_exports_again(tmp_path: Path, mocker) -> None:
    config, providers, source = _setup(tmp_path)
    exports = _exports(mocker)
    e2e._ingest(config, providers, source, tmp_path / "run-1")
    path = tmp_path / "viz" / VISUALIZATION_DATA_FILENAME
    first = export_build_time(path)
    (source / "carrier.txt").write_text("Carrier ships widgets for Depot.")
    config.indexing.reset = True

    e2e._ingest(config, providers, source, tmp_path / "run-2")

    assert exports == [path, path]
    assert export_build_time(path) not in (None, first)


def test_an_export_of_an_earlier_build_is_not_reported(tmp_path: Path, mocker) -> None:
    # Another run (another cache directory, say) wrote the namespace since
    # this export: a run that writes nothing itself reports no export, and
    # run-visualization builds one from the stores.
    config, providers, source = _setup(tmp_path)
    exports = _exports(mocker)
    e2e._ingest(config, providers, source, tmp_path / "run-1")
    path = tmp_path / "viz" / VISUALIZATION_DATA_FILENAME
    text = path.read_text(encoding="utf-8")
    stamp = export_build_time(path)
    assert stamp is not None
    path.write_text(text.replace(stamp, "2000-01-01T00:00:00Z", 1), encoding="utf-8")

    context = e2e._ingest(config, providers, source, tmp_path / "run-2")

    assert exports == [path]
    assert context.visualization_data_path is None


def test_the_export_runs_once_the_lease_is_given_up(tmp_path: Path, mocker) -> None:
    config, providers, source = _setup(tmp_path)
    events: list[str] = []
    write = pipeline_module.write_visualization_data
    leave = NamespaceLease.__exit__

    def exporting(*args):  # noqa: ANN002, ANN202
        events.append("export")
        return write(*args)

    def released(self, *args):  # noqa: ANN001, ANN002, ANN202
        events.append("release")
        return leave(self, *args)

    mocker.patch.object(pipeline_module, "write_visualization_data", exporting)
    mocker.patch.object(NamespaceLease, "__exit__", released)

    context = e2e._ingest(config, providers, source, tmp_path / "run-1")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert events == ["release", "export"]


def test_a_run_under_a_new_pipeline_id_keeps_the_current_export(
    tmp_path: Path, mocker
) -> None:
    # The CLI's default pipeline id is new each run: its cache directory
    # holds no export yet, but the one of the run before shows the stores.
    config, providers, source = _setup(tmp_path)
    config.graph.visualization.outputs_directory = None
    exports = _exports(mocker)
    cache = tmp_path / "cache"

    def ingest(pipeline_id: str):  # noqa: ANN202
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
            return pipeline.run(source, pipeline_id=pipeline_id)
        finally:
            pipeline.close()

    ingest("first")
    first = cache / "first" / "visualization" / VISUALIZATION_DATA_FILENAME
    assert exports == [first]

    context = ingest("second")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert exports == [first]
    assert context.visualization_data_path == str(first)


class _Bucket:
    """The S3 cache sync over a directory: ``<root>/<pipeline id>/<path>``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def upload_file(self, pipeline_id: str, cache_dir: Path, path: Path) -> None:
        target = self.root / pipeline_id / path.relative_to(cache_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    def sync_pipeline_to_s3(self, pipeline_id: str, cache_dir: Path) -> None:
        for path in cache_dir.rglob("*.json"):
            self.upload_file(pipeline_id, cache_dir, path)

    def sync_pipeline_from_s3(self, pipeline_id: str, cache_dir: Path) -> None:
        source = self.root / pipeline_id
        if source.is_dir():
            shutil.copytree(source, cache_dir, dirs_exist_ok=True)

    def read_head(self, pipeline_id: str, relative_path: str, size: int) -> str | None:
        path = self.root / pipeline_id / relative_path
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return handle.read(size)

    def uri(self, pipeline_id: str, relative_path: str) -> str:
        return f"s3://bucket/{pipeline_id}/{relative_path}"


@pytest.mark.parametrize("uploaded", [True, False], ids=["uploaded", "lost"])
def test_a_new_task_on_an_empty_disk_keeps_the_uploaded_export(
    tmp_path: Path, mocker, uploaded: bool
) -> None:
    # A deployed task: a new pipeline id each run, an empty disk, the cache
    # synced to S3. The build record names the run whose upload shows the
    # stores, so a run that writes nothing finds it there (17 s and a 59 MB
    # upload at 10k text units); an upload that is gone is not written again
    # (an incremental run exports none).
    config, providers, source = _setup(tmp_path)
    config.graph.visualization.outputs_directory = None
    exports = _exports(mocker)
    bucket = _Bucket(tmp_path / "bucket")

    def ingest(pipeline_id: str):  # noqa: ANN202
        cache = tmp_path / "disk" / pipeline_id
        pipeline = DataIngestionPipeline(
            config,
            PipelineConfig(
                stages_enabled=dict.fromkeys(PipelineStageType, True),
                local_directory=cache,
            ),
            source_directory=source,
            providers=providers,
        )
        pipeline.pipeline_config.s3_sync_enabled = True
        pipeline.s3_cache_manager = bucket
        for stage in pipeline.stages:
            if isinstance(stage, ps.GraphResolutionStage):
                stage.resolver.entity_resolver.use_process_pool = False
                stage.resolver.relationship_resolver.use_process_pool = False
        try:
            return pipeline.run(source, pipeline_id=pipeline_id)
        finally:
            pipeline.close()

    ingest("first")
    relative = Path("visualization") / VISUALIZATION_DATA_FILENAME
    assert (bucket.root / "first" / relative).is_file()
    assert len(exports) == 1
    if not uploaded:
        (bucket.root / "first" / relative).unlink()

    context = ingest("second")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert len(exports) == 1
    if uploaded:
        assert context.visualization_data_path == (
            f"s3://bucket/first/{relative.as_posix()}"
        )
    else:
        # An incremental run writes none (run-visualization builds it).
        assert context.visualization_data_path is None
        assert not (bucket.root / "second" / relative).exists()


def test_the_console_builds_the_graph_an_incremental_run_did_not_export(
    tmp_path: Path, mocker
) -> None:
    # No console job exports one (rv25 perf 4): the Graph page, once
    # opened, builds it from the stores when no export shows them.
    from unified_kg_rag.application.cli.console import graph_api

    config, providers, source = _setup(tmp_path)
    config.graph.visualization.outputs_directory = None
    cache = tmp_path / "cache"
    # An empty namespace (no build recorded): nothing to build.
    assert graph_api.current_graph(config, cache).state == "unrecorded"
    exports = _exports(mocker)
    e2e._ingest(config, providers, source, tmp_path / "run-1")
    (source / "carrier.txt").write_text("Carrier ships widgets for Depot.")
    e2e._ingest(config, providers, source, tmp_path / "run-2")

    current = graph_api.current_graph(config, cache)
    path = current.path

    assert current.state == "current" and current.built_at
    assert path == cache / "console-graphs" / "default" / VISUALIZATION_DATA_FILENAME
    names = {n["attributes"]["name"] for n in json.loads(path.read_text())["nodes"]}
    assert "Carrier" in names and "Vendor" in names
    # It shows the stores now: found again, not built.
    written = path.stat().st_mtime_ns
    assert graph_api.current_graph(config, cache).path == path
    assert path.stat().st_mtime_ns == written
    assert len(exports) == 1  # the full build's
    config.graph.visualization.export_data = False
    assert graph_api.current_graph(config, cache).state == "export_off"
