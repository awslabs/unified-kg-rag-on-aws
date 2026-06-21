# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""The pipeline builds the DocStatusPort once and injects it into the stages that
need it, rather than each stage constructing its own (hexagonal injection)."""

from __future__ import annotations

import pytest

from aws_graphrag.application.ingestion.pipeline_stages import (
    DocumentLoadingStage,
    IndexingStage,
)
from aws_graphrag.domain.models import Config

pytestmark = pytest.mark.unit


def test_injected_store_is_used_not_reconstructed() -> None:
    sentinel = object()  # stand-in DocStatusPort

    loading = DocumentLoadingStage.__new__(DocumentLoadingStage)
    loading._doc_status = sentinel  # type: ignore[attr-defined]
    assert loading._build_doc_status_store() is sentinel

    indexing = IndexingStage.__new__(IndexingStage)
    indexing._doc_status = sentinel  # type: ignore[attr-defined]
    assert indexing._build_doc_status_store() is sentinel


def test_stage_accepts_doc_status_kwarg() -> None:
    # Constructor wiring: the kwarg the pipeline injects is accepted and stored.
    sentinel = object()
    indexing = IndexingStage(config=Config(), doc_status=sentinel)
    assert indexing._doc_status is sentinel
