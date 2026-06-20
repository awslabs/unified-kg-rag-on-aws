# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Integration test: DocumentLoadingStage incremental filter (moto DynamoDB)."""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from aws_graphrag.adapters.aws import DynamoDBDocStatusStore
from aws_graphrag.application.ingestion.pipeline_stages import DocumentLoadingStage
from aws_graphrag.domain.ingestion.delta_detector import (
    compute_content_hash,
    compute_doc_id,
)
from aws_graphrag.domain.models import Config, DocStatusRecord, Document

pytestmark = pytest.mark.integration


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
        kept, skipped = stage._apply_incremental_filter([a, _doc("/b.txt", "B")])

        assert [d.file_path for d in kept] == ["/b.txt"]
        assert skipped == 1


def test_filter_degrades_gracefully_on_error() -> None:
    # No moto context -> store calls fail -> process everything, no crash.
    config = Config()
    config.aws.dynamodb.enabled = True
    config.aws.dynamodb.create_table_if_missing = False
    stage = _stage(config, boto3.Session(region_name="us-east-1"))
    docs = [_doc("/a.txt", "A")]
    kept, skipped = stage._apply_incremental_filter(docs)
    assert kept == docs and skipped == 0
