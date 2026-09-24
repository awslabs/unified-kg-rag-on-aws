# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The resume point must follow the cache, not the recorded status (AWS-free).

A stage's recorded ``completed`` status is only useful if the cache still holds
the output it stands for. Once cache keys carry a fingerprint of the inputs that
produced a stage's output, a changed model or prompt, or a cache written under
the pre-fingerprint keys, leaves the status in place while the output becomes
unreachable. Scheduling the resume from the status alone then starts downstream
of the invalidated stage on an empty context.

These tests drive the real ``CacheManager``, ``PipelineStateManager`` and
``PipelineResumeManager`` through ``DataIngestionPipeline._prepare_resume`` and
into ``_execute_pipeline_stages`` (stage execution stubbed), and pin that a run
either recomputes from the first stage the cache no longer backs or fails before
any downstream stage executes. Never an empty restored context.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    LanguageModelId,
    PipelineContext,
    PipelineStageResult,
    PipelineStageStatus,
    PipelineStageType,
    Relationship,
)
from unified_kg_rag.shared.cache_manager import CacheManager
from unified_kg_rag.shared.exceptions import PipelineExecutionError
from unified_kg_rag.shared.pipeline_manager import (
    PipelineResumeManager,
    PipelineStateManager,
)

pytestmark = pytest.mark.unit

PIPELINE_ID = "pid"
EXTRACTION = "graph_extraction"
GLEANING = "gleaning"


def _result(stage_name: str, status: PipelineStageStatus) -> PipelineStageResult:
    return PipelineStageResult(
        stage_name=stage_name, status=status, start_time=datetime(2026, 1, 1)
    )


def _context() -> PipelineContext:
    """graph_extraction produced one entity and completed; gleaning then failed."""
    context = PipelineContext(
        pipeline_id=PIPELINE_ID,
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=Path("/tmp/src"),
    )
    context.entities = [Entity(id="e1", name="Vendor")]
    context.relationships = [
        Relationship(id="r1", source_id="e1", target_id="e1", type="SELF")
    ]
    context.stage_results = [
        _result(EXTRACTION, PipelineStageStatus.COMPLETED),
        _result(GLEANING, PipelineStageStatus.FAILED),
    ]
    return context


def _pipeline(config: Config, root: Path) -> DataIngestionPipeline:
    """A pipeline with ``__init__`` bypassed, wired to real managers on tmp_path.

    Holds what the save path, ``_prepare_resume`` and
    ``_execute_pipeline_stages`` read; no adapters or boto session are built.
    """
    cache_manager = CacheManager(config=config, cache_directory=root)
    pipeline = object.__new__(DataIngestionPipeline)
    pipeline.config = config
    pipeline.pipeline_config = SimpleNamespace(cache_enabled=True)
    pipeline.cache_manager = cache_manager
    pipeline.state_manager = PipelineStateManager(cache_manager)
    pipeline.resume_manager = PipelineResumeManager(pipeline.state_manager, config)
    pipeline.name_to_type_map = {stage.value: stage for stage in PipelineStageType}
    pipeline.stages = [
        SimpleNamespace(name=stage.value)
        for stage in DataIngestionPipeline.STAGE_CLASSES
    ]
    return pipeline


def _record_run(pipeline: DataIngestionPipeline) -> None:
    """Persist the history and cache graph_extraction's output the way a run does."""
    context = _context()
    pipeline.state_manager.save_pipeline_metadata(context)
    pipeline._save_stage_outputs_to_cache(context, EXTRACTION)


def _record_legacy_run(pipeline: DataIngestionPipeline) -> None:
    """Persist the history and cache the output under the pre-fingerprint keys.

    This is what any cache directory written before stage keys carried an input
    fingerprint looks like: the key is the bare context attribute name.
    """
    context = _context()
    pipeline.state_manager.save_pipeline_metadata(context)
    for attr in ("entities", "relationships"):
        pipeline.cache_manager.save_stage_result(
            data=getattr(context, attr),
            cache_key=attr,
            stage_name=EXTRACTION,
            pipeline_id=PIPELINE_ID,
        )


def _resume(
    pipeline: DataIngestionPipeline, mocker, resume_from: str | None = None
) -> tuple[str | None, PipelineContext, list[str]]:
    """Run the orchestrator's resume path and record which stages would execute."""
    executed: list[str] = []

    def fake_execute_stage(stage, context):
        executed.append(stage.name)
        return _result(stage.name, PipelineStageStatus.COMPLETED)

    mocker.patch.object(pipeline, "_execute_stage", side_effect=fake_execute_stage)
    mocker.patch.object(pipeline, "_should_stop_pipeline", return_value=False)

    context, start = pipeline._prepare_resume(PIPELINE_ID, resume_from)
    pipeline._execute_pipeline_stages(context, start)
    return start, context, executed


def _changed_model_config() -> Config:
    config = Config()
    current = config.processing.graph_extraction.extraction_model_id
    config.processing.graph_extraction.extraction_model_id = next(
        model for model in LanguageModelId if model != current
    )
    return config


def _changed_prompt_config() -> Config:
    config = Config()
    config.custom_prompts.graph_extraction_system = "Extract only monetary amounts."
    return config


CHANGED_CONFIGS = pytest.mark.parametrize(
    "changed_config",
    [_changed_model_config(), _changed_prompt_config()],
    ids=["model", "prompt"],
)


class TestAutoResume:
    def test_unchanged_config_resumes_downstream_on_restored_output(
        self, tmp_path, mocker
    ) -> None:
        _record_run(_pipeline(Config(), tmp_path))

        start, context, executed = _resume(_pipeline(Config(), tmp_path), mocker)

        assert start == GLEANING
        assert [entity.id for entity in context.entities] == ["e1"]
        assert executed[0] == GLEANING

    @CHANGED_CONFIGS
    def test_changed_inputs_recompute_from_the_invalidated_stage(
        self, tmp_path, mocker, changed_config: Config
    ) -> None:
        _record_run(_pipeline(Config(), tmp_path))

        start, context, executed = _resume(_pipeline(changed_config, tmp_path), mocker)

        # The start stage moves back to the stage whose output the cache no
        # longer backs, its recorded completion is gone from the context so the
        # scheduler cannot skip it, and it is the first stage to execute.
        assert start == EXTRACTION
        assert [r.stage_name for r in context.stage_results] == []
        assert executed[0] == EXTRACTION

    def test_legacy_cache_never_continues_on_an_empty_context(
        self, tmp_path, mocker
    ) -> None:
        # A cache written under the pre-fingerprint keys, resumed with an
        # unchanged configuration. Either outcome is safe: the output is read
        # (the pre-fingerprint behaviour) or the stage is recomputed. Starting
        # downstream with no entities is the one outcome that is not.
        _record_legacy_run(_pipeline(Config(), tmp_path))

        start, context, executed = _resume(_pipeline(Config(), tmp_path), mocker)

        recomputed = executed[0] == EXTRACTION
        restored = [entity.id for entity in context.entities] == ["e1"]
        assert (
            recomputed or restored
        ), f"resumed at '{start}' with entities={context.entities!r}"

    def test_legacy_cache_is_recomputed_and_the_legacy_key_is_named(
        self, tmp_path, mocker, caplog
    ) -> None:
        # A legacy entry carries no record of the inputs that produced it, so it
        # cannot be verified against the current configuration: treat it as a
        # miss, and say which key was found so the one-time recomputation on
        # upgrade is explained rather than silent.
        _record_legacy_run(_pipeline(Config(), tmp_path))

        with caplog.at_level(logging.WARNING):
            start, context, executed = _resume(_pipeline(Config(), tmp_path), mocker)

        assert start == EXTRACTION
        assert executed[0] == EXTRACTION
        assert "'entities'" in caplog.text

    def test_downstream_completion_is_invalidated_with_its_upstream(
        self, tmp_path, mocker
    ) -> None:
        # gleaning completed too, and its output IS cached under the new
        # configuration's key. It consumed the invalidated extraction output, so
        # it is recomputed regardless.
        changed = _changed_model_config()
        baseline = _pipeline(Config(), tmp_path)
        _record_run(baseline)
        context = _context()
        context.stage_results = [
            _result(EXTRACTION, PipelineStageStatus.COMPLETED),
            _result(GLEANING, PipelineStageStatus.COMPLETED),
            _result("graph_resolution", PipelineStageStatus.FAILED),
        ]
        baseline.state_manager.save_pipeline_metadata(context)
        _pipeline(changed, tmp_path)._save_stage_outputs_to_cache(context, GLEANING)

        start, context, executed = _resume(_pipeline(changed, tmp_path), mocker)

        assert start == EXTRACTION
        assert executed[:2] == [EXTRACTION, GLEANING]


class TestExplicitResume:
    """A phased run names its start stage, and its window may not contain the
    upstream stages, so a missing prerequisite cannot be recomputed here: the
    resume has to fail before any downstream stage runs on empty input."""

    def test_unchanged_config_restores_the_prerequisite(self, tmp_path, mocker) -> None:
        _record_run(_pipeline(Config(), tmp_path))

        start, context, executed = _resume(
            _pipeline(Config(), tmp_path), mocker, resume_from=GLEANING
        )

        assert start == GLEANING
        assert [entity.id for entity in context.entities] == ["e1"]
        assert executed[0] == GLEANING

    @CHANGED_CONFIGS
    def test_changed_inputs_fail_before_downstream_execution(
        self, tmp_path, mocker, changed_config: Config
    ) -> None:
        _record_run(_pipeline(Config(), tmp_path))
        pipeline = _pipeline(changed_config, tmp_path)
        executed = mocker.patch.object(pipeline, "_execute_stage")

        with pytest.raises(PipelineExecutionError, match=EXTRACTION):
            pipeline._prepare_resume(PIPELINE_ID, GLEANING)

        executed.assert_not_called()

    def test_legacy_cache_fails_before_downstream_execution(
        self, tmp_path, mocker
    ) -> None:
        _record_legacy_run(_pipeline(Config(), tmp_path))
        pipeline = _pipeline(Config(), tmp_path)
        executed = mocker.patch.object(pipeline, "_execute_stage")

        with pytest.raises(PipelineExecutionError, match=EXTRACTION):
            pipeline._prepare_resume(PIPELINE_ID, GLEANING)

        executed.assert_not_called()


class TestEmptyOutputIsStillOutput:
    def test_stage_that_produced_nothing_is_not_recomputed(
        self, tmp_path, mocker
    ) -> None:
        # Translation is off by default: the stage completes and leaves
        # translated_units empty. That is its output under these inputs, so the
        # resume must not read the empty cache as "not produced" and rewind to
        # it on every run.
        pipeline = _pipeline(Config(), tmp_path)
        context = _context()
        context.stage_results = [
            _result("translation", PipelineStageStatus.COMPLETED),
            _result(EXTRACTION, PipelineStageStatus.FAILED),
        ]
        pipeline.state_manager.save_pipeline_metadata(context)
        pipeline._save_stage_outputs_to_cache(context, "translation")

        start, _, executed = _resume(_pipeline(Config(), tmp_path), mocker)

        assert start == EXTRACTION
        assert executed[0] == EXTRACTION
