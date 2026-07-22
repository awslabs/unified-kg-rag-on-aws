# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test: DocumentLoadingStage incremental filter (moto DynamoDB)."""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from unified_kg_rag.adapters.aws import DynamoDBDocStatusStore
from unified_kg_rag.application.ingestion.pipeline_stages import DocumentLoadingStage
from unified_kg_rag.domain.ingestion.delta_detector import (
    compute_content_hash,
    compute_doc_id,
)
from unified_kg_rag.domain.models import Config, DocStatusRecord, Document

pytestmark = pytest.mark.integration


class _Ctx:
    """Minimal stand-in for PipelineContext (only the incremental attrs are set)."""

    incremental_delta = None
    incremental_fingerprints: dict[str, str] = {}
    documents: list[Document] = []


def test_deletion_only_run_allows_empty_output() -> None:
    # Regression: a deletion-only / all-unchanged incremental delta yields zero
    # documents to (re)extract. DOCUMENT_LOADING is critical + must-have-input,
    # so without the incremental opt-out the run would crash before deletions
    # are propagated. _allows_empty_output must permit it.
    from unified_kg_rag.domain.models import DocumentDelta

    config = Config()
    config.aws.dynamodb.enabled = True
    stage = _stage(config, boto3.Session(region_name="us-east-1"))

    ctx = _Ctx()
    ctx.incremental_delta = DocumentDelta(deleted=["doc-removed"])
    ctx.documents = []  # nothing survived the filter
    assert stage._allows_empty_output(ctx) is True

    # Non-incremental run with no documents must still fail the check.
    plain = _Ctx()
    plain.incremental_delta = None
    plain.documents = []
    assert stage._allows_empty_output(plain) is False


def _doc(path: str, text: str) -> Document:
    return Document(
        page_content=text,
        document_id="x",
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
        file_type="txt",
        total_pages=1,
    )


def _stage(config: Config, session: boto3.Session) -> DocumentLoadingStage:
    stage = DocumentLoadingStage.__new__(DocumentLoadingStage)
    stage.config = config  # type: ignore[attr-defined]
    stage.boto_session = session  # type: ignore[attr-defined]
    # No injected port -> the stage builds the default DynamoDB adapter itself.
    stage._doc_status = None  # type: ignore[attr-defined]
    return stage


def test_filter_skips_unchanged_documents() -> None:
    with mock_aws():
        config = Config()
        config.aws.dynamodb.enabled = True
        config.aws.dynamodb.table_name = "test-doc-status"
        session = boto3.Session(region_name="us-east-1")
        store = DynamoDBDocStatusStore(config, boto_session=session)
        _ = store.client  # create table

        # Seed registry: /a.txt already processed (unchanged), /b.txt absent.
        a = _doc("/a.txt", "A")
        store.put(
            DocStatusRecord(
                doc_id=compute_doc_id("/a.txt"),
                content_hash=compute_content_hash(a),
            )
        )

        stage = _stage(config, session)
        ctx = _Ctx()
        kept, skipped = stage._apply_incremental_filter([a, _doc("/b.txt", "B")], ctx)

        assert [d.file_path for d in kept] == ["/b.txt"]
        assert skipped == 1
        # The delta is stashed on the context for the IndexingStage.
        assert ctx.incremental_delta is not None
        assert compute_doc_id("/b.txt") in ctx.incremental_delta.new


def test_filter_degrades_gracefully_on_error() -> None:
    # No moto context -> store calls fail -> process everything, no crash.
    config = Config()
    config.aws.dynamodb.enabled = True
    config.aws.dynamodb.create_table_if_missing = False
    stage = _stage(config, boto3.Session(region_name="us-east-1"))
    docs = [_doc("/a.txt", "A")]
    kept, skipped = stage._apply_incremental_filter(docs, _Ctx())
    assert kept == docs and skipped == 0
