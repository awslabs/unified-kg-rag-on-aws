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
        retuned.processing.max_attempts += 1
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

    def test_graph_analysis_knobs_change_no_stage_key(self) -> None:
        # Centrality and statistics settings shape only values no later stage
        # reads, so changing them must not discard the community reports
        # (LLM output) or the indexing cache.
        baseline = Config()
        changed = Config()
        changed.graph.analysis.centrality.pagerank_alpha = 0.5
        changed.graph.analysis.centrality.calculate_betweenness = (
            not baseline.graph.analysis.centrality.calculate_betweenness
        )
        changed.graph.analysis.statistics.calculate_diameter = (
            not baseline.graph.analysis.statistics.calculate_diameter
        )
        assert changed.graph.analysis != baseline.graph.analysis

        for stage in (
            PipelineStageType.GRAPH_ANALYSIS,
            PipelineStageType.COMMUNITY_DETECTION,
            PipelineStageType.INDEXING,
        ):
            assert stage_input_fingerprint(baseline, stage) == (
                stage_input_fingerprint(changed, stage)
            ), stage

    def test_graph_analysis_key_follows_its_upstream_inputs(self) -> None:
        # The stage passes the resolved graph through, so its cached output
        # must still go stale when resolution changes.
        changed = Config()
        changed.processing.similarity_threshold = 0.5
        assert stage_input_fingerprint(
            Config(), PipelineStageType.GRAPH_ANALYSIS
        ) != stage_input_fingerprint(changed, PipelineStageType.GRAPH_ANALYSIS)

    # Default-config fingerprints of every cached stage as released before the
    # per-tier effort split (#128) and before the fields added since. They must
    # not move, or every existing stage cache turns into a miss on upgrade.
    _RELEASED_DEFAULT_FINGERPRINTS = {
        PipelineStageType.DOCUMENT_PARSING: "1035b7a62512",
        PipelineStageType.DOCUMENT_LOADING: "ac616352a4e7",
        # The fast tier's default model moved to Claude Haiku 5.5, so every
        # stage from chunking on (each has a fast-tier role upstream) re-keys.
        PipelineStageType.TEXT_CHUNKING: "7d24d450ee75",
        PipelineStageType.TRANSLATION: "5abcca511058",
        PipelineStageType.GRAPH_EXTRACTION: "5cecc719e0a7",
        PipelineStageType.GLEANING: "7daa823fa004",
        PipelineStageType.GRAPH_RESOLUTION: "5709e17d032a",
        PipelineStageType.CLAIM_EXTRACTION: "99e66b287398",
        PipelineStageType.CLAIM_RESOLUTION: "99e66b287398",
        # graph.analysis is no longer an input (it shapes nothing a later
        # stage reads), so graph analysis shares claim resolution's inputs.
        PipelineStageType.GRAPH_ANALYSIS: "99e66b287398",
        PipelineStageType.COMMUNITY_DETECTION: "a35b6a8701d2",
    }

    def test_default_config_fingerprints_are_unchanged(self) -> None:
        # Derived effort (#128) and the omit-when-default fields
        # (fast_effort, default_max_output_tokens, model_overrides) and the
        # non-output source_scope keep the released keys for a default config.
        config = Config()
        assert {
            stage: stage_input_fingerprint(config, stage)
            for stage in self._RELEASED_DEFAULT_FINGERPRINTS
        } == self._RELEASED_DEFAULT_FINGERPRINTS

    def test_corpus_manifest_changes_every_cached_stage_key(self) -> None:
        config = Config()
        for stage in self._RELEASED_DEFAULT_FINGERPRINTS:
            unchanged = stage_input_fingerprint(config, stage, "corpus-v1")
            assert unchanged == stage_input_fingerprint(config, stage, "corpus-v1")
            assert unchanged != stage_input_fingerprint(config, stage, "corpus-v2")
            assert unchanged != stage_input_fingerprint(config, stage)

    def test_source_scope_does_not_change_the_fingerprint(self) -> None:
        changed = Config()
        changed.processing.document_parsing.source_scope = "s3://example/corpus/"
        assert stage_input_fingerprint(
            changed, PipelineStageType.DOCUMENT_PARSING
        ) == stage_input_fingerprint(Config(), PipelineStageType.DOCUMENT_PARSING)

    def test_tier_efforts_fold_into_the_fingerprint(self) -> None:
        stage = PipelineStageType.DOCUMENT_PARSING

        def fingerprint(bedrock: dict[str, str]) -> str:
            config = Config.model_validate({"aws": {"bedrock": bedrock}})
            return stage_input_fingerprint(config, stage)

        baseline = fingerprint({})
        # The legacy key and default_effort are the same input.
        assert fingerprint({"effort": "medium"}) == fingerprint(
            {"default_effort": "medium"}
        )
        assert fingerprint({"default_effort": "medium"}) != baseline
        assert fingerprint({"fast_effort": "low"}) == baseline
        assert fingerprint({"fast_effort": "medium"}) != baseline

    def test_key_is_the_attribute_plus_the_fingerprint(self) -> None:
        config = Config()
        expected = stage_input_fingerprint(config, PipelineStageType.DOCUMENT_LOADING)
        key = stage_cache_key(config, PipelineStageType.DOCUMENT_LOADING, "documents")
        assert key == f"documents-{expected}"

    @pytest.mark.parametrize(
        "path, value",
        [
            ("default_max_output_tokens", 1024),
            ("model_overrides", {"vendor.synthetic-model": {"max_output_tokens": 1}}),
        ],
    )
    def test_bedrock_output_shaping_fields_are_inputs(self, path, value) -> None:
        changed = Config()
        setattr(changed.aws.bedrock, path, value)
        assert stage_input_fingerprint(
            Config(), PipelineStageType.DOCUMENT_LOADING
        ) != stage_input_fingerprint(changed, PipelineStageType.DOCUMENT_LOADING)


def _write_corpus(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


_EXTENSIONS = {".txt", ".json"}


class TestCorpusManifest:
    """A changed corpus must be a cache miss even under a fixed pipeline_id."""

    def _fingerprint(self, root: Path, exclude: tuple[Path, ...] = ()) -> str:
        return cache_keys.corpus_manifest_fingerprint(root, _EXTENSIONS, exclude)

    def test_content_edit_add_and_remove_change_the_fingerprint(self, tmp_path) -> None:
        root = _write_corpus(tmp_path, {"a.txt": "Vendor pays Buyer 100."})
        baseline = self._fingerprint(root)

        (root / "a.txt").write_text("Vendor pays Buyer 900.", encoding="utf-8")
        edited = self._fingerprint(root)  # same size, different bytes
        assert edited != baseline

        _write_corpus(root, {"sub/b.txt": "Buyer ships goods."})
        added = self._fingerprint(root)
        assert added != edited

        (root / "sub/b.txt").unlink()
        assert self._fingerprint(root) == edited

    def test_mtime_hidden_files_and_owned_dirs_do_not_count(self, tmp_path) -> None:
        import os

        root = _write_corpus(tmp_path / "src", {"a.txt": "Vendor pays Buyer."})
        baseline = self._fingerprint(root, (root / "cache",))

        os.utime(root / "a.txt", (1_000_000, 1_000_000))
        _write_corpus(
            root,
            {
                ".hidden/notes.txt": "scratch",
                "cache/pid/documents.json": "[]",
                "image.png": "binary",
            },
        )
        assert self._fingerprint(root, (root / "cache",)) == baseline

    def test_changed_corpus_does_not_resume_stale_output(self, tmp_path) -> None:
        cache_manager = _cache_manager(tmp_path / "cache")
        pipeline = _pipeline(Config(), cache_manager)
        pipeline.corpus_fingerprint = "corpus-v1"
        pipeline._save_stage_outputs_to_cache(_context(), STAGE)

        state = PipelineStateManager(cache_manager)
        state.save_pipeline_metadata(_context())

        same = PipelineResumeManager(state, Config())
        same.corpus_fingerprint = "corpus-v1"
        restored = same.restore_pipeline_context(PIPELINE_ID, [STAGE])
        assert [entity.id for entity in restored.entities] == ["e1"]

        changed = PipelineResumeManager(state, Config())
        changed.corpus_fingerprint = "corpus-v2"
        assert changed.restore_pipeline_context(PIPELINE_ID, [STAGE]).entities == []
        ok, errors = changed.validate_pipeline_integrity(PIPELINE_ID)
        assert ok is False and errors

    def test_pipeline_fingerprint_skips_its_cache_and_export_dirs(
        self, tmp_path
    ) -> None:
        root = _write_corpus(tmp_path, {"a.txt": "Vendor pays Buyer."})
        pipeline = object.__new__(DataIngestionPipeline)
        pipeline.pipeline_config = SimpleNamespace(local_directory=root / "cache")
        pipeline.target_directory = root / "parsed"
        baseline = pipeline._compute_corpus_fingerprint(root)

        # The pipeline's own JSON output (cache entries, parsed export) lives
        # under the source root here; it must not move the fingerprint mid-run.
        _write_corpus(root, {"cache/pid/entities.json": "[]", "parsed/a.json": "{}"})
        assert pipeline._compute_corpus_fingerprint(root) == baseline
        _write_corpus(root, {"b.json": "{}"})
        assert pipeline._compute_corpus_fingerprint(root) != baseline
