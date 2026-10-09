# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An unreadable doc-status registry fails the run instead of rebuilding.

The loading stage used to swallow every registry error and process all
documents with no delta, so the indexing stage took the full-rebuild path: it
replaced the suffix's live index content with this run's documents and never
recorded them. A missing table, AccessDenied or throttling then turned every
incremental run into a destructive full rebuild. AWS-free: an in-memory
registry, a recording indexing manager and the moto-backed DynamoDB adapter.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from unified_kg_rag.adapters.aws import DynamoDBDocStatusStore
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.domain.models import (
    Config,
    Document,
    DocumentDelta,
    PipelineContext,
    PipelineStageResult,
    PipelineStageStatus,
    PipelineStageType,
)
from unified_kg_rag.shared import DocStatusRegistryError, PipelineStageError

pytestmark = pytest.mark.unit


class _FailingRegistry(FakeDocStatusStore):
    def diff(self, incoming: dict[str, str], scope: str | None = None):
        raise RuntimeError("registry offline")


def _config() -> Config:
    config = Config()
    config.aws.dynamodb.enabled = True
    return config


def _document(path: str) -> Document:
    return Document(
        page_content="Vendor supplies Buyer.",
        document_id=f"run-{path}",
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
        file_type="txt",
        total_pages=1,
    )


def _parsed_context(root: Path) -> PipelineContext:
    """A context whose parsing stage completed with one document."""
    context = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=root,
        documents=[_document(str(root / "a.txt"))],
    )
    context.add_stage_result(
        PipelineStageResult(
            stage_name=PipelineStageType.DOCUMENT_PARSING.value,
            status=PipelineStageStatus.COMPLETED,
            start_time=datetime(2026, 1, 1),
        )
    )
    return context


def _loading_stage(config: Config, root: Path, store, mocker):
    mocker.patch("boto3.Session")
    return ps.DocumentLoadingStage(config, source_directory=root, doc_status=store)


def _indexing_stage(config: Config, store, mocker):
    manager = mocker.MagicMock()
    manager.config = config
    manager.initialize.return_value = True
    mocker.patch.object(ps, "IndexingManager", return_value=manager)
    stage = ps.IndexingStage(
        config=config, boto_session=mocker.MagicMock(), doc_status=store
    )
    return stage, manager


def test_registry_diff_error_fails_the_loading_stage(tmp_path, mocker) -> None:
    config = _config()
    stage = _loading_stage(config, tmp_path, _FailingRegistry(), mocker)
    context = _parsed_context(tmp_path)

    result = stage.execute(context)

    assert result.status is PipelineStageStatus.FAILED
    assert "registry offline" in (result.error_message or "")
    assert "aws.dynamodb.enabled" in (result.error_message or "")
    assert context.incremental_delta is None

    with pytest.raises(DocStatusRegistryError):
        stage._apply_incremental_filter(list(context.documents), context)


def test_indexing_does_not_rebuild_when_the_delta_is_missing(tmp_path, mocker) -> None:
    # continue_on_error lets the pipeline reach indexing after the loading
    # stage failed; the indexing stage must still refuse the full rebuild.
    config = _config()
    store = _FailingRegistry()
    context = _parsed_context(tmp_path)
    _loading_stage(config, tmp_path, store, mocker).execute(context)
    indexing, manager = _indexing_stage(config, store, mocker)

    with pytest.raises(PipelineStageError, match="no document delta"):
        indexing._execute_core(context)

    result = indexing.execute(context)
    assert result.status is PipelineStageStatus.FAILED
    manager.index_all_data.assert_not_called()
    manager.index_delta.assert_not_called()
    manager.clear_all_data.assert_not_called()


def test_indexing_without_the_registry_still_runs_the_full_path(mocker) -> None:
    config = Config()  # registry disabled: a full rebuild is the intended path
    indexing, manager = _indexing_stage(config, None, mocker)
    manager.index_all_data.return_value = {}
    context = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory="/",
    )

    indexing._execute_core(context)

    manager.index_all_data.assert_called_once()


def test_loading_stage_drops_a_delta_restored_from_previous_run_metadata(
    tmp_path, mocker
) -> None:
    # A reused pipeline id restores the previous run's delta and fingerprints
    # from its metadata. Re-running the loading stage with the registry now
    # disabled must not leave them on the context for the indexing stage.
    stage = _loading_stage(Config(), tmp_path, None, mocker)
    context = _parsed_context(tmp_path)
    context.incremental_delta = DocumentDelta(changed=["old-doc"])
    context.incremental_fingerprints = {"old-doc": "old-hash"}
    context.incremental_scope = "old-scope"

    result = stage.execute(context)

    assert result.status is PipelineStageStatus.COMPLETED
    assert context.incremental_delta is None
    assert context.incremental_fingerprints == {}
    assert context.incremental_scope is None


def test_loading_stage_replaces_a_restored_delta_with_a_fresh_diff(
    tmp_path, mocker
) -> None:
    stage = _loading_stage(_config(), tmp_path, FakeDocStatusStore(), mocker)
    context = _parsed_context(tmp_path)
    context.incremental_delta = DocumentDelta(changed=["old-doc"])
    context.incremental_fingerprints = {"old-doc": "old-hash"}

    stage.execute(context)

    assert context.incremental_delta is not None
    assert "old-doc" not in context.incremental_fingerprints
    assert len(context.incremental_delta.new) == 1
    assert context.incremental_delta.changed == []


def test_missing_dynamodb_table_fails_fast(tmp_path, mocker) -> None:
    with mock_aws():
        config = _config()
        config.aws.dynamodb.table_name = "missing-doc-status"
        config.aws.dynamodb.create_table_if_missing = False
        session = boto3.Session(region_name="us-east-1")
        store = DynamoDBDocStatusStore(config, boto_session=session)

        with pytest.raises(DocStatusRegistryError) as raised:
            store.diff({"doc": "hash"})
        message = str(raised.value)
        assert "'missing-doc-status'" in message
        assert "ResourceNotFoundException" in message
        assert "create_table_if_missing" in message

        stage = _loading_stage(config, tmp_path, store, mocker)
        context = _parsed_context(tmp_path)
        result = stage.execute(context)
        assert result.status is PipelineStageStatus.FAILED
        assert "missing-doc-status" in (result.error_message or "")
        assert context.incremental_delta is None


def test_dynamodb_client_retries_throttling() -> None:
    with mock_aws():
        config = _config()
        session = boto3.Session(region_name="us-east-1")
        client = DynamoDBDocStatusStore(config, boto_session=session).client
        retries = client.meta.config.retries
        assert retries["mode"] == "standard"
        assert retries["total_max_attempts"] == 10
