# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Resume-strategy unit tests for PipelineResumeManager.

These cover the phased-execution handoff used by the Step Functions ingestion
pipeline: each phase runs in its own Fargate task and resumes at a stage that
the *previous* phase did not record, so the upstream completed stages must still
be restored from the (S3-synced) cache.
"""

import pytest

from aws_graphrag.shared.pipeline_manager import PipelineResumeManager

pytestmark = pytest.mark.unit


def _results(*names_statuses: tuple[str, str]) -> list[dict]:
    return [{"stage_name": n, "status": s} for n, s in names_statuses]


def test_explicit_resume_loads_all_upstream_when_target_not_recorded() -> None:
    # Prep phase recorded only its own stages; GraphBuild phase resumes at
    # graph_extraction, which is NOT in the recorded history. All recorded
    # completed stages are upstream and must be restored (the bug: returned []).
    stage_results = _results(
        ("document_parsing", "completed"),
        ("document_loading", "completed"),
        ("text_chunking", "completed"),
        ("translation", "completed"),
    )

    resume, completed = PipelineResumeManager._handle_explicit_stage_resume(
        stage_results, "graph_extraction"
    )

    assert resume == "graph_extraction"
    assert completed == [
        "document_parsing",
        "document_loading",
        "text_chunking",
        "translation",
    ]


def test_explicit_resume_skips_failed_upstream_stages() -> None:
    stage_results = _results(
        ("document_parsing", "completed"),
        ("document_loading", "failed"),
    )

    resume, completed = PipelineResumeManager._handle_explicit_stage_resume(
        stage_results, "graph_extraction"
    )

    assert resume == "graph_extraction"
    assert completed == ["document_parsing"]


def test_explicit_resume_within_recorded_history_loads_only_prior() -> None:
    # When the target IS recorded, only stages before it count as completed.
    stage_results = _results(
        ("document_parsing", "completed"),
        ("text_chunking", "completed"),
        ("translation", "failed"),
    )

    resume, completed = PipelineResumeManager._handle_explicit_stage_resume(
        stage_results, "text_chunking"
    )

    assert resume == "text_chunking"
    assert completed == ["document_parsing"]


def test_explicit_resume_empty_history() -> None:
    resume, completed = PipelineResumeManager._handle_explicit_stage_resume(
        [], "graph_extraction"
    )
    assert resume == "graph_extraction"
    assert completed == []
