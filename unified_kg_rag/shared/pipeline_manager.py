# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from unified_kg_rag.domain.models import (
    Claim,
    Community,
    CommunityReport,
    Config,
    Document,
    Entity,
    PipelineContext,
    PipelineStageResult,
    PipelineStageStatus,
    PipelineStageType,
    Relationship,
    TextUnit,
)

from .exceptions import PipelineResumeError, PipelineStateError
from .logging import get_logger
from .utils.cache_keys import stage_cache_key

if TYPE_CHECKING:
    from unified_kg_rag.ports.cache import CachePort


logger = get_logger(__name__)


# Canonical ingestion stage order (mirrors DataIngestionPipeline.STAGE_CLASSES).
# Used to keep stage_results sorted when merging across phased executions so the
# accumulated history reads in pipeline order regardless of which phase wrote it.
_CANONICAL_STAGE_ORDER: tuple[str, ...] = (
    "document_parsing",
    "document_loading",
    "text_chunking",
    "translation",
    "graph_extraction",
    "gleaning",
    "graph_resolution",
    "claim_extraction",
    "claim_resolution",
    "graph_analysis",
    "community_detection",
    "indexing",
)


def _canonical_index(stage_name: str) -> int:
    """Position of a stage in the canonical order; unknown stages sort last."""
    if stage_name in _CANONICAL_STAGE_ORDER:
        return _CANONICAL_STAGE_ORDER.index(stage_name)
    return len(_CANONICAL_STAGE_ORDER)


class PipelineStateManager:
    PIPELINE_METADATA_FILE = "pipeline_metadata.json"
    EXCLUDED_DATA_FIELDS = {
        "centrality_metrics",
        "claims",
        "communities",
        "community_reports",
        "documents",
        "entities",
        "graph",
        "graph_statistics",
        "knowledge_graph",
        "relationships",
        "resolved_claims",
        "resolved_entities",
        "resolved_relationships",
        "text_units",
        "translated_units",
    }

    def __init__(self, cache_manager: "CachePort") -> None:
        self.cache_manager = cache_manager

    def create_context_from_metadata(self, metadata: dict) -> PipelineContext:
        context_data = {
            k: v for k, v in metadata.items() if k not in self.EXCLUDED_DATA_FIELDS
        }
        context = PipelineContext.model_validate(context_data)
        original_status = context.status
        context.status = PipelineStageStatus.RUNNING
        logger.info(
            "Created context from metadata for pipeline '%s' (status: %s -> %s)",
            context.pipeline_id,
            original_status,
            context.status.value,
        )
        return context

    def pipeline_exists(self, pipeline_id: str) -> bool:
        metadata_path = (
            self.cache_manager.get_pipeline_cache_dir(pipeline_id)
            / self.PIPELINE_METADATA_FILE
        )
        return metadata_path.exists() and metadata_path.stat().st_size > 0

    def load_pipeline_metadata(self, pipeline_id: str) -> dict[str, Any]:
        metadata_path = (
            self.cache_manager.get_pipeline_cache_dir(pipeline_id)
            / self.PIPELINE_METADATA_FILE
        )

        try:
            if not metadata_path.exists():
                raise PipelineStateError(f"Metadata file not found: '{metadata_path}'")

            if metadata_path.stat().st_size == 0:
                raise PipelineStateError(f"Metadata file is empty: '{metadata_path}'")

            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

            if not isinstance(metadata, dict):
                raise PipelineStateError(
                    f"Metadata file does not contain a JSON object: '{metadata_path}'"
                )

            required_fields = ["pipeline_id", "start_time", "status"]
            if not all(field in metadata for field in required_fields):
                raise PipelineStateError(
                    f"Metadata missing one of required fields: '{required_fields}'"
                )
            return metadata

        except json.JSONDecodeError as e:
            logger.error("Corrupted metadata file, invalid JSON: '%s'", metadata_path)
            raise PipelineStateError(
                f"Corrupted metadata file, invalid JSON: '{metadata_path}'"
            ) from e
        except Exception as e:
            logger.error(
                "Failed to load pipeline metadata from '%s': %s", metadata_path, e
            )
            raise PipelineStateError(
                f"Failed to load pipeline metadata from '{metadata_path}': {e}"
            ) from e

    def save_pipeline_metadata(self, context: PipelineContext) -> None:
        metadata_directory = self.cache_manager.get_pipeline_cache_dir(
            context.pipeline_id
        )
        metadata_path = metadata_directory / self.PIPELINE_METADATA_FILE
        temp_path = metadata_path.with_suffix(".tmp")

        try:
            metadata_directory.mkdir(parents=True, exist_ok=True)
            # Phased executions (each a separate task sharing one pipeline_id via
            # S3) run a stage window and would otherwise overwrite the on-disk
            # metadata with only their own stage_results, dropping the upstream
            # phases' history. Merge the existing results in first so the resume
            # of a later phase can see — and restore the cache of — every
            # completed upstream stage (e.g. graph_resolution's relationships,
            # claim_resolution's claims), not just the immediately prior phase.
            self._merge_persisted_stage_results(context, metadata_path)
            self._update_context_status(context)

            if context.status in {
                PipelineStageStatus.COMPLETED,
                PipelineStageStatus.FAILED,
            }:
                context.end_time = datetime.now()

            json_data = context.model_dump_json(
                exclude=self.EXCLUDED_DATA_FIELDS, exclude_none=True, indent=2
            )

            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(json_data)
                f.flush()
                os.fsync(f.fileno())

            temp_path.replace(metadata_path)

            if not self._verify_saved_metadata(metadata_path):
                raise PipelineStateError("Metadata verification failed after save")

            logger.info(
                "Pipeline metadata saved for '%s' (status: %s)",
                context.pipeline_id,
                context.status.value,
            )

        except Exception as e:
            logger.error(
                "Failed to save pipeline metadata for '%s': %s", context.pipeline_id, e
            )
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    logger.warning(
                        "Failed to clean up temporary file: %s", cleanup_error
                    )
            raise PipelineStateError(
                f"Critical error saving pipeline metadata: {e}"
            ) from e

    def _merge_persisted_stage_results(
        self, context: PipelineContext, metadata_path: Path
    ) -> None:
        """Union the context's stage_results with any already on disk.

        Keyed by stage_name: the in-memory result (this phase's fresh run) wins
        over a previously persisted one for the same stage; stages only present
        on disk (earlier phases) are preserved. The merged list is sorted into
        canonical pipeline order so downstream resume logic sees a coherent,
        cumulative history across phases.
        """
        if not metadata_path.exists() or metadata_path.stat().st_size == 0:
            return

        try:
            with open(metadata_path, encoding="utf-8") as f:
                persisted = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Could not read existing metadata to merge stage results "
                "(treating as fresh): %s",
                e,
            )
            return

        merged: dict[str, PipelineStageResult] = {}
        for raw in persisted.get("stage_results", []):
            try:
                result = PipelineStageResult.model_validate(raw)
            except Exception as e:  # noqa: BLE001 - skip unparseable legacy entries
                logger.warning("Skipping unparseable persisted stage result: %s", e)
                continue
            merged[result.stage_name] = result

        # This phase's in-memory results take precedence for the same stage.
        for result in context.stage_results:
            merged[result.stage_name] = result

        context.stage_results = sorted(
            merged.values(), key=lambda r: _canonical_index(r.stage_name)
        )

    @staticmethod
    def _update_context_status(context: PipelineContext) -> None:
        old_status = context.status

        if not context.stage_results:
            context.status = PipelineStageStatus.PENDING
        elif any(r.status == PipelineStageStatus.FAILED for r in context.stage_results):
            context.status = PipelineStageStatus.FAILED
        elif all(
            r.status == PipelineStageStatus.COMPLETED for r in context.stage_results
        ):
            context.status = PipelineStageStatus.COMPLETED
        else:
            context.status = PipelineStageStatus.RUNNING

        if old_status != context.status:
            logger.info(
                "Pipeline status updated: '%s' -> '%s'",
                old_status.value,
                context.status.value,
            )

    @staticmethod
    def _verify_saved_metadata(metadata_path: Path) -> bool:
        try:
            if not metadata_path.exists() or metadata_path.stat().st_size == 0:
                logger.error(
                    "Metadata file is missing or empty after save: '%s'", metadata_path
                )
                return False

            with open(metadata_path, encoding="utf-8") as f:
                json.load(f)
            return True

        except Exception as e:
            logger.error("Metadata verification failed: %s", e)
            return False

    def repair_pipeline_metadata(self, context: PipelineContext) -> bool:
        logger.info(
            "Attempting to repair metadata for pipeline: '%s'", context.pipeline_id
        )
        try:
            self.save_pipeline_metadata(context)
            is_verified = self.verify_pipeline_metadata(context.pipeline_id)

            if is_verified:
                logger.info(
                    "Successfully repaired metadata for pipeline: '%s'",
                    context.pipeline_id,
                )
            else:
                logger.error(
                    "Failed to repair metadata for pipeline: '%s'", context.pipeline_id
                )
            return is_verified

        except Exception as e:
            logger.error(
                "Error during metadata repair for '%s': %s", context.pipeline_id, e
            )
            return False

    def verify_pipeline_metadata(self, pipeline_id: str) -> bool:
        try:
            self.load_pipeline_metadata(pipeline_id)
            return True
        except PipelineStateError:
            return False

    def update_stage_result(
        self, context: PipelineContext, stage_result: PipelineStageResult
    ) -> None:
        try:
            index = next(
                i
                for i, r in enumerate(context.stage_results)
                if r.stage_name == stage_result.stage_name
            )
            context.stage_results[index] = stage_result
        except StopIteration:
            context.add_stage_result(stage_result)

        self.save_pipeline_metadata(context)


class PipelineResumeManager:
    STAGE_CACHE_MAPPING = {
        PipelineStageType.CLAIM_EXTRACTION: {"claims": "claims.json"},
        PipelineStageType.CLAIM_RESOLUTION: {"resolved_claims": "resolved_claims.json"},
        PipelineStageType.COMMUNITY_DETECTION: {
            "communities": "communities.json",
            "community_reports": "community_reports.json",
        },
        PipelineStageType.DOCUMENT_LOADING: {"documents": "documents.json"},
        PipelineStageType.GLEANING: {"entities": "entities.json"},
        PipelineStageType.GRAPH_ANALYSIS: {
            "resolved_entities": "resolved_entities.json"
        },
        PipelineStageType.GRAPH_EXTRACTION: {
            "entities": "entities.json",
            "relationships": "relationships.json",
        },
        PipelineStageType.GRAPH_RESOLUTION: {
            "resolved_entities": "resolved_entities.json",
            "resolved_relationships": "resolved_relationships.json",
        },
        PipelineStageType.TEXT_CHUNKING: {"text_units": "text_units.json"},
        PipelineStageType.TRANSLATION: {"translated_units": "translated_units.json"},
    }
    CONTEXT_ATTR_TO_MODEL = {
        "claims": Claim,
        "communities": Community,
        "community_reports": CommunityReport,
        "documents": Document,
        "entities": Entity,
        "relationships": Relationship,
        "resolved_claims": Claim,
        "resolved_entities": Entity,
        "resolved_relationships": Relationship,
        "text_units": TextUnit,
        "translated_units": TextUnit,
    }

    def __init__(
        self, state_manager: PipelineStateManager, config: Config | None = None
    ) -> None:
        self.state_manager = state_manager
        self.cache_manager = state_manager.cache_manager
        # Cache keys carry a fingerprint of the config that produced the stage
        # output, so resuming needs the same config the save path used. Omitting
        # it falls back to framework defaults, which fingerprints to a different
        # key for any customised run — a cache miss (recompute), never a stale
        # hit.
        self.config = config or Config()

    def determine_resume_strategy(
        self, pipeline_id: str, explicit_stage: str | None = None
    ) -> tuple[str, list[str]]:
        logger.info(
            "Determining resume strategy for pipeline '%s'%s",
            pipeline_id,
            f" from stage: {explicit_stage}" if explicit_stage else "",
        )

        try:
            metadata = self.state_manager.load_pipeline_metadata(pipeline_id)
            stage_results = metadata.get("stage_results", [])

            if explicit_stage:
                start_stage, completed_stages = self._handle_explicit_stage_resume(
                    stage_results, explicit_stage
                )
            else:
                start_stage, completed_stages = self._handle_auto_resume(stage_results)

            # A recorded "completed" only stands for output the cache still
            # holds. Cache keys carry a fingerprint of the inputs that produced
            # the output, so a changed model or prompt (or a cache written
            # under the pre-fingerprint keys) leaves the status in place while
            # the output becomes unreachable. Scheduling from the status alone
            # would then start downstream of the stale stage on an empty
            # context, so the resume point is derived from what the cache
            # verifiably holds.
            result = self._verify_completed_stages_against_cache(
                pipeline_id, start_stage, completed_stages, explicit_stage
            )

            logger.info(
                "Resume strategy determined - Start from: '%s', Load %s completed stages",
                result[0],
                len(result[1]),
            )
            return result

        except PipelineResumeError:
            raise
        except Exception as e:
            logger.error(
                "Failed to determine resume strategy for pipeline '%s': %s",
                pipeline_id,
                e,
            )
            raise PipelineResumeError(
                f"Failed to determine resume strategy for pipeline '{pipeline_id}': {e}"
            ) from e

    @staticmethod
    def _handle_explicit_stage_resume(
        stage_results: list[dict], explicit_stage: str
    ) -> tuple[str, list[str]]:
        stage_names = [result["stage_name"] for result in stage_results]

        if explicit_stage not in stage_names:
            # The resume target hasn't run yet — typical of phased execution where
            # an earlier phase (separate task) recorded only the stages it ran and
            # this phase resumes at the next one. Every recorded *completed* stage
            # is by construction upstream of the resume target, so load them all
            # to restore the upstream outputs (e.g. translated text units).
            completed_stages = [
                result["stage_name"]
                for result in stage_results
                if result["status"] == "completed"
            ]
            logger.warning(
                "Stage '%s' not in pipeline history (phased resume); restoring %d "
                "completed upstream stage(s) and starting from it.",
                explicit_stage,
                len(completed_stages),
            )
            return explicit_stage, completed_stages

        explicit_index = stage_names.index(explicit_stage)
        completed_stages = [
            result["stage_name"]
            for result in stage_results[:explicit_index]
            if result["status"] == "completed"
        ]
        return explicit_stage, completed_stages

    @staticmethod
    def _handle_auto_resume(stage_results: list[dict]) -> tuple[str, list[str]]:
        completed_stages = []
        resume_stage = None

        for result in stage_results:
            if result["status"] == "completed":
                completed_stages.append(result["stage_name"])
            else:
                resume_stage = result["stage_name"]
                break

        if resume_stage is None:
            resume_stage = (
                stage_results[-1]["stage_name"] if stage_results else "document_loading"
            )

        return resume_stage, completed_stages

    def _verify_completed_stages_against_cache(
        self,
        pipeline_id: str,
        start_stage: str,
        completed_stages: list[str],
        explicit_stage: str | None,
    ) -> tuple[str, list[str]]:
        """Keep only the completed prefix whose output the cache still holds.

        Walks the completed stages in canonical order and stops at the first
        one with no cached output under the current inputs. An auto resume
        moves its start back to that stage, so it and every stage downstream
        of it are recomputed. An explicit resume names its start and, in a
        phased run, may not have the upstream stages in its window, so it
        fails here instead of running downstream stages on an empty context.
        """
        for stage_name in sorted(completed_stages, key=_canonical_index):
            missing = self._missing_cache_keys(pipeline_id, stage_name)
            if not missing:
                continue

            self._log_unbacked_stage(pipeline_id, stage_name, missing)

            if explicit_stage:
                missing_keys = ", ".join(f"'{key}'" for _, key in missing)
                message = (
                    f"Cannot resume from '{explicit_stage}': upstream stage "
                    f"'{stage_name}' is recorded completed, but the cache does not "
                    f"hold its output under the current inputs (missing "
                    f"{missing_keys}). Resume from '{stage_name}' to recompute it, "
                    f"or force a rebuild."
                )
                logger.error(message)
                raise PipelineResumeError(message)

            verified = [
                name
                for name in completed_stages
                if _canonical_index(name) < _canonical_index(stage_name)
            ]
            logger.warning(
                "Rewinding resume point from '%s' to '%s': it and every stage "
                "downstream of it will be recomputed",
                start_stage,
                stage_name,
            )
            return stage_name, verified

        return start_stage, completed_stages

    def _missing_cache_keys(
        self, pipeline_id: str, stage_name: str
    ) -> list[tuple[str, str]]:
        """List ``(context_attr, cache_key)`` for each output of ``stage_name``
        the cache does not hold under the current inputs.

        A stage with no cache mapping (document_parsing, indexing) produces
        nothing the resume reads back, so there is nothing to verify.
        """
        try:
            stage_type = PipelineStageType(stage_name)
        except ValueError:
            return []

        missing = []
        for context_attr in self.STAGE_CACHE_MAPPING.get(stage_type, {}):
            cache_key = stage_cache_key(self.config, stage_type, context_attr)
            if not self.cache_manager.cache_exists(cache_key, pipeline_id):
                missing.append((context_attr, cache_key))
        return missing

    def _log_unbacked_stage(
        self, pipeline_id: str, stage_name: str, missing: list[tuple[str, str]]
    ) -> None:
        for context_attr, cache_key in missing:
            # Before keys carried an input fingerprint, the key was the bare
            # attribute name. Such an entry records nothing about the inputs
            # that produced it, so it cannot be checked against the current
            # configuration: it is a miss, and the stage is recomputed once.
            if self.cache_manager.cache_exists(context_attr, pipeline_id):
                logger.warning(
                    "Stage '%s' is recorded completed and its output is cached "
                    "under the pre-fingerprint key '%s', which carries no record "
                    "of the inputs that produced it; treating it as a miss "
                    "(expected key '%s')",
                    stage_name,
                    context_attr,
                    cache_key,
                )
            else:
                logger.warning(
                    "Stage '%s' is recorded completed, but the cache holds no '%s' "
                    "for it under the current inputs (key '%s'): the inputs "
                    "changed since it ran, or the cache is incomplete",
                    stage_name,
                    context_attr,
                    cache_key,
                )

    def restore_pipeline_context(
        self, pipeline_id: str, completed_stages: list[str]
    ) -> PipelineContext:
        logger.info(
            "Restoring pipeline context for '%s' with %s completed stages",
            pipeline_id,
            len(completed_stages),
        )

        try:
            metadata = self.state_manager.load_pipeline_metadata(pipeline_id)
            context = self.state_manager.create_context_from_metadata(metadata)
            loaded_count = 0

            for stage_name in completed_stages:
                try:
                    if self._load_stage_cache_data(pipeline_id, stage_name, context):
                        loaded_count += 1
                except Exception as e:
                    logger.error(
                        "Error loading cache for stage '%s': %s", stage_name, e
                    )

            logger.info(
                "Context restoration completed for '%s': %s/%s stages loaded",
                pipeline_id,
                loaded_count,
                len(completed_stages),
            )
            return context

        except Exception as e:
            logger.error(
                "Failed to restore pipeline context for '%s': %s", pipeline_id, e
            )
            raise PipelineResumeError(
                f"Failed to restore pipeline context for '{pipeline_id}': {e}"
            ) from e

    def _load_stage_cache_data(
        self, pipeline_id: str, stage_name: str, context: PipelineContext
    ) -> bool:
        try:
            stage_type = PipelineStageType(stage_name)
            cache_mapping = self.STAGE_CACHE_MAPPING.get(stage_type, {})
        except ValueError:
            return True

        if not cache_mapping:
            return True

        loaded_files_count = 0

        for context_attr, _ in cache_mapping.items():
            try:
                model_class = self.CONTEXT_ATTR_TO_MODEL.get(context_attr)
                data: Any = self.cache_manager.load_stage_result(
                    cache_key=stage_cache_key(self.config, stage_type, context_attr),
                    pipeline_id=pipeline_id,
                    data_type=model_class if model_class else None,
                )

                if data is not None:
                    setattr(context, context_attr, data)
                    loaded_files_count += 1
                else:
                    logger.warning(
                        "Cache data not found for stage '%s': '%s'",
                        stage_name,
                        context_attr,
                    )
            except Exception as e:
                logger.error(
                    "Failed to load cache data '%s' for stage '%s': %s",
                    context_attr,
                    stage_name,
                    e,
                )

        return loaded_files_count > 0

    def validate_pipeline_integrity(self, pipeline_id: str) -> tuple[bool, list[str]]:
        logger.info("Validating pipeline integrity for '%s'", pipeline_id)
        errors = []

        try:
            metadata = self.state_manager.load_pipeline_metadata(pipeline_id)
        except PipelineStateError as e:
            logger.error("Pipeline integrity validation failed: %s", e)
            return False, [str(e)]

        for result in metadata.get("stage_results", []):
            if result.get("status") != "completed":
                continue

            stage_name = result["stage_name"]
            for _, cache_key in self._missing_cache_keys(pipeline_id, stage_name):
                error_msg = (
                    f"Missing cache for completed stage '{stage_name}', "
                    f"key: '{cache_key}'"
                )
                errors.append(error_msg)
                logger.warning(error_msg)

        is_valid = not errors
        logger.info(
            "Pipeline integrity validation %s for '%s'%s",
            "passed" if is_valid else "failed",
            pipeline_id,
            f" with {len(errors)} errors" if not is_valid else "",
        )
        return is_valid, errors
