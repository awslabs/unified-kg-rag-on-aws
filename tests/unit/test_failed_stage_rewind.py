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
    # An explicit resume restores only the stages before its start: whether
    # indexing ran before comes from the recorded stages (rerun).
    caplog.clear()
    with caplog.at_level("WARNING"):
        rewind(_pipeline(["indexing"]), failed, "indexing", DONE, explicit=True)
        assert "'graph_extraction' failed on 1" not in caplog.text
        rewind(
            _pipeline(["indexing"]),
            failed,
            "indexing",
            DONE,
            explicit=True,
            rerun=True,
        )
    assert "'graph_extraction' failed on 1" in caplog.text
    # Its first run (a phased run's indexing window) records them once.
    assert rewind(_pipeline(["indexing"]), failed, "indexing", DONE) is None
    # A stage at or after the start is not restored (_drop_unrun_failures).
    late = _context({"claim_extraction": ["u1"]})
    assert rewind(_pipeline([]), late, "graph_resolution", finished) is None


def test_the_refusal_holds_when_an_enabled_stage_also_failed() -> None:
    # graph_extraction (not enabled) and claim_extraction (enabled) failed:
    # the rewind to claim_extraction skipped the refusal, and the indexing
    # stage recorded graph_extraction's documents FAILED once more without
    # retrying them.
    finished = [*DONE, "indexing"]
    enabled = ORDER[ORDER.index("claim_extraction") :]
    both = _context({"graph_extraction": ["u1"], "claim_extraction": ["u2"]})
    rewind = DataIngestionPipeline._rewind_to_failed_stage
    with pytest.raises(PipelineResumeError, match="'graph_extraction' failed on 1"):
        rewind(_pipeline(enabled), both, "indexing", finished)
    # Its first run still rewinds: the indexing stage records them once.
    assert rewind(_pipeline(enabled), both, "indexing", DONE) == (
        "claim_extraction",
        DONE[: DONE.index("claim_extraction")],
    )
    # A failed stage after the rewind target is dropped, not refused.
    late = _context({"claim_extraction": ["u2"], "community_detection": ["u3"]})
    enabled = [name for name in ORDER if name != "community_detection"]
    assert rewind(_pipeline(enabled), late, "indexing", finished) == (
        "claim_extraction",
        DONE[: DONE.index("claim_extraction")],
    )


def test_failures_of_stages_the_resume_neither_runs_nor_restores_are_dropped() -> None:
    # A resume at graph_resolution with graph_resolution, the stage after
    # claim_extraction and indexing enabled: graph_extraction is restored
    # (before the start), so its failures stay; claim_extraction (after the
    # start, not enabled) does not run, so its failures, own errors
    # included, go; the enabled later stage keeps its own.
    resolution, claims = "graph_resolution", "claim_extraction"
    later = ORDER[ORDER.index(claims) + 1]
    assert ORDER.index("graph_extraction") < ORDER.index(resolution)
    assert ORDER.index(resolution) < ORDER.index(claims)
    context = PipelineContext.model_construct(
        failed_text_unit_ids={
            "graph_extraction": ["u1"],
            claims: ["u2"],
            later: ["u3"],
        },
        item_failed_text_unit_ids={"graph_extraction": ["u1"], claims: ["u2"]},
    )
    _pipeline([resolution, later, "indexing"])._drop_unrun_failures(context, resolution)
    assert context.failed_text_unit_ids == {"graph_extraction": ["u1"], later: ["u3"]}
    assert context.item_failed_text_unit_ids == {"graph_extraction": ["u1"]}


def test_a_rewind_to_an_unrun_stage_restores_only_the_stages_before_it() -> None:
    # Gleaning was off when the pipeline ran: claim extraction completed
    # after it. Resumed from gleaning, the later stages run again and are
    # not restored (their output would stand for a graph gleaning changes).
    gleaning = "gleaning"
    completed = [name for name in DONE if name != gleaning]
    later = [n for n in completed if ORDER.index(n) > ORDER.index(gleaning)]
    assert later, "a completed stage after gleaning"
    start, restored = DataIngestionPipeline._rewind_to_unrun_stage(
        _pipeline(ORDER), "indexing", completed
    )
    assert start == gleaning
    assert restored == completed[: completed.index(later[0])]
