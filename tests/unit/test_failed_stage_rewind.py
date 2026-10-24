# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The failed-stage rewind of a resume (``DataIngestionPipeline``).

A resume moves back to the earliest completed, enabled stage before its
start that recorded failed text units, so the stage retries them instead of
the indexing stage recording their documents FAILED again; a completed stage
run again on the failures of a stage the run does not enable is refused.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.models.pipeline import PipelineContext
from unified_kg_rag.shared import PipelineResumeError

pytestmark = pytest.mark.unit

ORDER = [stage.value for stage in DataIngestionPipeline.STAGE_CLASSES]
DONE = ORDER[: ORDER.index("indexing")]


def _pipeline(enabled: list[str]) -> DataIngestionPipeline:
    pipeline = object.__new__(DataIngestionPipeline)
    pipeline.stages = [SimpleNamespace(name=name) for name in enabled]  # type: ignore[misc]
    pipeline._doc_status = FakeDocStatusStore()
    return pipeline


def _context(failed: dict[str, list[str]]) -> PipelineContext:
    return PipelineContext.model_construct(failed_text_unit_ids=failed)


def test_two_failing_stages_rewind_to_the_earliest() -> None:
    got = DataIngestionPipeline._rewind_to_failed_stage(
        _pipeline(ORDER),
        _context({"claim_extraction": ["u2"], "graph_extraction": ["u1"]}),
        "indexing",
        DONE,
    )
    assert got == ("graph_extraction", DONE[: DONE.index("graph_extraction")])


def test_a_disabled_or_uncompleted_stage_is_not_a_rewind_target() -> None:
    enabled = [name for name in ORDER if name != "graph_extraction"]
    failed = _context({"graph_extraction": ["u1"]})
    rewind = DataIngestionPipeline._rewind_to_failed_stage
    assert rewind(_pipeline(enabled), failed, "indexing", DONE) is None
    partial = [name for name in DONE if name != "graph_extraction"]
    assert rewind(_pipeline(ORDER), failed, "indexing", partial) is None


def test_a_completed_stage_run_again_on_a_disabled_stages_failures_is_refused(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # An indexing-only rerun of a finished pipeline: indexing completed and
    # recorded the documents FAILED; running it again would count them once
    # more without retrying them.
    finished = [*DONE, "indexing"]
    failed = _context({"graph_extraction": ["u1"]})
    rewind = DataIngestionPipeline._rewind_to_failed_stage
    with pytest.raises(PipelineResumeError, match="'graph_extraction' failed on 1"):
        rewind(_pipeline(["indexing"]), failed, "indexing", finished)
    with caplog.at_level("WARNING"):
        assert (
            rewind(_pipeline(["indexing"]), failed, "indexing", finished, explicit=True)
            is None
        )
    assert "'graph_extraction' failed on 1" in caplog.text
    # Its first run (a phased run's indexing window) records them once.
    assert rewind(_pipeline(["indexing"]), failed, "indexing", DONE) is None
    # A stage at or after the start is not restored (_drop_unrun_failures).
    late = _context({"claim_extraction": ["u1"]})
    assert rewind(_pipeline([]), late, "graph_resolution", finished) is None
