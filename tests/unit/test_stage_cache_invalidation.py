# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-cache invalidation on a configuration change (AWS-free).

A resumed run must not reuse stage output produced by a different prompt, model
or processing rule. These tests drive the real save path
(``DataIngestionPipeline._save_stage_outputs_to_cache``) and the real restore
path (``PipelineResumeManager.restore_pipeline_context``) over a real
``CacheManager`` on tmp_path, so they assert the round-trip rather than the
key-building helper in isolation.
"""

from __future__ import annotations

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
from unified_kg_rag.shared.pipeline_manager import (
    PipelineResumeManager,
    PipelineStateManager,
)
from unified_kg_rag.shared.utils import (
    cache_keys,
    stage_cache_key,
    stage_input_fingerprint,
)

pytestmark = pytest.mark.unit

PIPELINE_ID = "pid"
STAGE = "graph_extraction"


def _cache_manager(root: Path) -> CacheManager:
    return CacheManager(config=Config(), cache_directory=root)


def _context() -> PipelineContext:
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
        PipelineStageResult(
            stage_name=STAGE,
            status=PipelineStageStatus.COMPLETED,
            start_time=datetime(2026, 1, 1),
        )
    ]
    return context


def _pipeline(config: Config, cache_manager: CacheManager) -> DataIngestionPipeline:
    """A pipeline with ``__init__`` bypassed, holding only what the save path reads.

    Avoids building adapters or a boto session; ``STAGE_OUTPUT_MAPPING`` is a
    class attribute and comes along for free.
    """
    pipeline = object.__new__(DataIngestionPipeline)
    pipeline.config = config
    pipeline.pipeline_config = SimpleNamespace(cache_enabled=True)
    pipeline.cache_manager = cache_manager
    pipeline.name_to_type_map = {STAGE: PipelineStageType.GRAPH_EXTRACTION}
    return pipeline


def _save_and_list_keys(config: Config, root: Path) -> set[str]:
    cache_manager = _cache_manager(root)
    _pipeline(config, cache_manager)._save_stage_outputs_to_cache(_context(), STAGE)
    return set(cache_manager.load_cache_index(PIPELINE_ID).entries)


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


class TestSavedKeyTracksInputs:
    """The key written to the cache index must move when the inputs move."""

    def test_unchanged_config_reuses_the_same_key(self, tmp_path) -> None:
        first = _save_and_list_keys(Config(), tmp_path / "first")
        second = _save_and_list_keys(Config(), tmp_path / "second")
        assert first
        assert first == second

    def test_changed_model_id_writes_a_different_key(self, tmp_path) -> None:
        baseline = _save_and_list_keys(Config(), tmp_path / "baseline")
        changed = _save_and_list_keys(_changed_model_config(), tmp_path / "changed")
        assert baseline and changed
        assert baseline.isdisjoint(changed)

    def test_changed_prompt_override_writes_a_different_key(self, tmp_path) -> None:
        baseline = _save_and_list_keys(Config(), tmp_path / "baseline")
        changed = _save_and_list_keys(_changed_prompt_config(), tmp_path / "changed")
        assert baseline and changed
        assert baseline.isdisjoint(changed)

    def test_throughput_knob_does_not_invalidate(self, tmp_path) -> None:
        # Concurrency and batch size change how long a stage takes, not what it
        # produces, so retuning them must not discard an expensive cache.
        retuned = Config()
        retuned.processing.max_concurrency += 7
        retuned.processing.batch_size += 3
        retuned.processing.max_retries += 1
        assert _save_and_list_keys(Config(), tmp_path / "base") == _save_and_list_keys(
            retuned, tmp_path / "retuned"
        )


class TestResumeRoundTrip:
    """Resume must load its own run's output and refuse another config's."""

    @staticmethod
    def _restore(config: Config, cache_manager: CacheManager) -> PipelineContext:
        state = PipelineStateManager(cache_manager)
        state.save_pipeline_metadata(_context())
        resume = PipelineResumeManager(state, config)
        return resume.restore_pipeline_context(PIPELINE_ID, [STAGE])

    def test_unchanged_config_resumes_from_cache(self, tmp_path) -> None:
        cache_manager = _cache_manager(tmp_path)
        _pipeline(Config(), cache_manager)._save_stage_outputs_to_cache(
            _context(), STAGE
        )

        restored = self._restore(Config(), cache_manager)
        assert [entity.id for entity in restored.entities] == ["e1"]

    @pytest.mark.parametrize(
        "changed_config", [_changed_model_config(), _changed_prompt_config()]
    )
    def test_changed_inputs_do_not_resume_stale_output(
        self, tmp_path, changed_config: Config
    ) -> None:
        cache_manager = _cache_manager(tmp_path)
        _pipeline(Config(), cache_manager)._save_stage_outputs_to_cache(
            _context(), STAGE
        )

        restored = self._restore(changed_config, cache_manager)
        # Pins the key-level miss only: restore_pipeline_context is asked for
        # the stage directly, bypassing determine_resume_strategy, which is what
        # keeps a run from continuing on this empty context (see
        # test_resume_point_follows_cache for the orchestration-level cases).
        assert restored.entities == []

    def test_integrity_check_reports_the_stale_entry_as_missing(self, tmp_path) -> None:
        # validate_pipeline_integrity must judge against the same key the save
        # path wrote, or it would call every completed stage's cache missing.
        cache_manager = _cache_manager(tmp_path)
        _pipeline(Config(), cache_manager)._save_stage_outputs_to_cache(
            _context(), STAGE
        )
        state = PipelineStateManager(cache_manager)
        state.save_pipeline_metadata(_context())

        ok, errors = PipelineResumeManager(state, Config()).validate_pipeline_integrity(
            PIPELINE_ID
        )
        assert ok is True
        assert errors == []

        ok, errors = PipelineResumeManager(
            state, _changed_model_config()
        ).validate_pipeline_integrity(PIPELINE_ID)
        assert ok is False
        assert errors


class TestFingerprintScope:
    """A stage's fingerprint covers its own inputs and every upstream stage's."""

    def test_downstream_stage_sees_an_upstream_change(self) -> None:
        baseline, changed = Config(), Config()
        changed.processing.chunking.max_chunk_size += 100

        # Chunking is upstream of extraction, so both must move.
        assert stage_input_fingerprint(
            baseline, PipelineStageType.TEXT_CHUNKING
        ) != stage_input_fingerprint(changed, PipelineStageType.TEXT_CHUNKING)
        assert stage_input_fingerprint(
            baseline, PipelineStageType.GRAPH_EXTRACTION
        ) != stage_input_fingerprint(changed, PipelineStageType.GRAPH_EXTRACTION)

    def test_upstream_stage_ignores_a_downstream_change(self) -> None:
        # Otherwise editing a community-report prompt would force a re-parse and
        # re-chunk of the whole corpus.
        baseline, changed = Config(), Config()
        changed.custom_prompts.community_report_system = "Summarise in one line."

        assert stage_input_fingerprint(
            baseline, PipelineStageType.TEXT_CHUNKING
        ) == stage_input_fingerprint(changed, PipelineStageType.TEXT_CHUNKING)
        assert stage_input_fingerprint(
            baseline, PipelineStageType.COMMUNITY_DETECTION
        ) != stage_input_fingerprint(changed, PipelineStageType.COMMUNITY_DETECTION)

    def test_stage_outside_the_canonical_order_folds_in_every_input(self) -> None:
        # A stage added without extending the canonical order must not get a
        # narrower fingerprint than its predecessors.
        assert cache_keys._input_paths_through(
            "not_a_pipeline_stage"  # type: ignore[arg-type]
        ) == cache_keys._input_paths_through(PipelineStageType.INDEXING)

    def test_key_is_the_attribute_plus_the_fingerprint(self) -> None:
        config = Config()
        expected = stage_input_fingerprint(config, PipelineStageType.DOCUMENT_LOADING)
        key = stage_cache_key(config, PipelineStageType.DOCUMENT_LOADING, "documents")
        assert key == f"documents-{expected}"
