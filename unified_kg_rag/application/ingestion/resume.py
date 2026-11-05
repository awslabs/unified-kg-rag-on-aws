# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a resumed ingestion run starts, and the context it starts from.

``prepare_resume`` restores a pipeline id's context from the stage cache
(``PipelineResumeManager``) and moves the resume point back, or refuses it,
when the restored stages cannot stand for this run (see its table). The
functions take the ``DataIngestionPipeline`` they plan the resume of: it
holds the enabled stages, the managers, the doc-status registry and the
config they read, and it keeps a delegating method for each public one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from unified_kg_rag.application.ingestion.incremental import delta_still_applies
from unified_kg_rag.application.ingestion.pipeline_stages import _with_record_keys
from unified_kg_rag.domain.ingestion.cache_keys import (
    changed_settings,
    extraction_settings,
)
from unified_kg_rag.domain.ingestion.contributions import record_keys
from unified_kg_rag.domain.models import (
    ExtractionSettings,
    PipelineContext,
    PipelineStageType,
)
from unified_kg_rag.ports import VectorIndexer
from unified_kg_rag.ports.indexer import BaseIndexer
from unified_kg_rag.shared import (
    NamespaceLeaseError,
    PipelineExecutionError,
    PipelineResumeError,
    PipelineStateError,
    get_logger,
)

if TYPE_CHECKING:
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline

# The pipeline's logger: the records keep the name they had before this
# module was split out of it.
logger = get_logger("unified_kg_rag.application.ingestion.pipeline")

_STALE_DELTA = (
    "the doc-status registry changed since this pipeline computed its "
    "incremental delta (another run committed some of its documents, a "
    "commit of this pipeline did not finish, runs died processing it "
    "often enough to isolate the document that kills them, or a document "
    "failed indexing.max_document_failures times)"
)


def prepare_resume(
    pipeline: DataIngestionPipeline, pipeline_id: str, resume_from_stage: str | None
) -> tuple[PipelineContext, str | None]:
    """Restore a pipeline id's context and the stage its resume starts at.

    The recorded stages give the resume point (after the last completed
    stage, or ``resume_from_stage``). These then move it back or refuse
    it, in this order:

    ================================  ================================  ==================
    Step                              When                              Explicit point
    ================================  ================================  ==================
    ``rewind_to_unrun_stage``         an enabled stage never completed  not applied
    ``_rewind_on_extraction_change``  the extraction settings or the    not applied
                                      registry changed since the delta
                                      (to document_loading)
    ``_check_upstream_complete``      the point is upstream of every    applied
                                      enabled stage: refused
    ``rewind_to_failed_stage``        a completed, enabled stage        kept, with a
                                      recorded failed units; a rerun    WARNING (for the
                                      on a disabled stage's failures    refusal too); the
                                      is refused                        failures count
                                                                        again
    ``rewind_on_missing_records``     a reset's stored extraction       applied
                                      records are gone
    ``delta_is_stale``                the registry no longer matches    applied
                                      the restored delta (to
                                      document_loading)
    ================================  ================================  ==================

    Each rewind restores the context again for the stages it keeps.
    ``drop_unrun_failures`` then forgets the failures of the stages the
    run neither runs nor restores.
    """
    logger.info(
        "Preparing pipeline resume for '%s' from stage: '%s'",
        pipeline_id,
        resume_from_stage or "auto-detect",
    )

    try:
        start_stage_name, completed_stages = (
            pipeline.resume_manager.determine_resume_strategy(
                pipeline_id, resume_from_stage
            )
        )
        if resume_from_stage is None:
            start_stage_name, completed_stages = pipeline._rewind_to_unrun_stage(
                start_stage_name, completed_stages
            )
            start_stage_name, completed_stages = _rewind_on_extraction_change(
                pipeline, pipeline_id, start_stage_name, completed_stages
            )

        _check_upstream_complete(pipeline, pipeline_id, start_stage_name)
        context = pipeline.resume_manager.restore_pipeline_context(
            pipeline_id, completed_stages
        )
        rewound = pipeline._rewind_to_failed_stage(
            context,
            start_stage_name,
            completed_stages,
            explicit=resume_from_stage is not None,
            rerun=resume_from_stage is not None
            and _recorded_completed(pipeline, pipeline_id, start_stage_name),
        )
        if rewound is not None:
            start_stage_name, completed_stages = rewound
            context = pipeline.resume_manager.restore_pipeline_context(
                pipeline_id, completed_stages
            )
        rewound = pipeline._rewind_on_missing_records(
            context, start_stage_name, completed_stages
        )
        if rewound is not None:
            start_stage_name, completed_stages = rewound
            context = pipeline.resume_manager.restore_pipeline_context(
                pipeline_id, completed_stages
            )
        if pipeline._delta_is_stale(context, start_stage_name):
            start_stage_name, completed_stages = _rewind_to_loading(
                pipeline, start_stage_name, completed_stages, _STALE_DELTA
            )
            context = pipeline.resume_manager.restore_pipeline_context(
                pipeline_id, completed_stages
            )

        pipeline._drop_unrun_failures(context, start_stage_name)
        pipeline._prepare_context_for_resume(context, start_stage_name)
        # What an earlier run recorded is not this run's.
        context.recorded_failed_doc_ids = []
        context.retrying_doc_ids = []
        context.claims_missing_doc_ids = []

        logger.info(
            "Pipeline resume prepared successfully - will start from stage: '%s'",
            start_stage_name,
        )
        return context, start_stage_name
    except (PipelineResumeError, PipelineStateError) as e:
        logger.error("Failed to prepare resume for pipeline '%s': %s", pipeline_id, e)
        raise PipelineExecutionError(f"Pipeline resume failed: {e}") from e


def _recorded_completed(
    pipeline: DataIngestionPipeline, pipeline_id: str, stage_name: str
) -> bool:
    """Whether ``pipeline_id`` recorded ``stage_name`` completed."""
    stage_results = pipeline.state_manager.load_pipeline_metadata(pipeline_id).get(
        "stage_results", []
    )
    return any(
        result.get("stage_name") == stage_name and result.get("status") == "completed"
        for result in stage_results
    )


def rewind_to_unrun_stage(
    pipeline: DataIngestionPipeline, start_stage_name: str, completed_stages: list[str]
) -> tuple[str, list[str]]:
    """Move an auto resume back to an enabled stage that never completed.

    A stage switched on since the pipeline last ran (gleaning or claim
    extraction, say) has no recorded result, so the resume point the
    recorded stages give lies past it and the stage would never run.
    The resume starts at the first such stage instead, and only the
    completed stages upstream of it are restored.
    """
    if start_stage_name not in pipeline._ORDER:
        return start_stage_name, completed_stages
    start_position = pipeline._ORDER.index(start_stage_name)
    completed = set(completed_stages)
    for stage in pipeline.stages:
        position = pipeline._ORDER.index(stage.name)
        if position >= start_position:
            break
        if stage.name not in completed:
            logger.warning(
                "Stage '%s' is enabled but has not completed in this "
                "pipeline (it was off when the pipeline last ran); resuming "
                "from it instead of '%s'",
                stage.name,
                start_stage_name,
            )
            return stage.name, pipeline._before(completed_stages, stage.name)
    return start_stage_name, completed_stages


def rewind_to_failed_stage(
    pipeline: DataIngestionPipeline,
    context: PipelineContext,
    start_stage_name: str | None,
    completed_stages: list[str],
    *,
    explicit: bool = False,
    rerun: bool = False,
) -> tuple[str, list[str]] | None:
    """Move a resume back to the earliest completed, enabled stage that
    recorded failed text units (``context.failed_text_unit_ids``), or
    None to keep it.

    Its restored output would make the indexing stage record their
    documents FAILED once more without extracting them again, so every
    re-run of the pipeline id would count one more failure. Run again,
    the stage retries only the failed units (the extraction store keeps
    what succeeded). A document that reached
    ``indexing.max_document_failures`` meanwhile is not retried: the
    delta no longer applies (``incremental.delta_still_applies``), and
    the resume diffs again. An ``explicit`` resume point
    (``--resume-from-stage``) is kept, with a warning, as the other
    rewinds of an auto resume do.
    """
    if start_stage_name not in pipeline._ORDER:
        return None
    enabled = {stage.name for stage in pipeline.stages}
    failed = pipeline._before(
        (
            name
            for name, unit_ids in context.failed_text_unit_ids.items()
            if unit_ids and name in completed_stages and name in enabled
        ),
        start_stage_name,
    )
    first = min(failed, key=pipeline._ORDER.index) if failed else None
    # Also when the resume rewinds to ``first``: a stage not enabled
    # that failed before it is not retried there either.
    _refuse_rerun_of_unrun_failures(
        pipeline,
        context,
        start_stage_name,
        completed_stages,
        explicit=explicit,
        resume_at=None if explicit else first,
        rerun=rerun,
    )
    if first is None:
        return None
    if explicit:
        logger.warning(
            "Stage '%s' failed on %d text units; resuming from '%s' as "
            "asked records their documents FAILED again without retrying "
            "them, one more run toward indexing.max_document_failures "
            "(resume from '%s' to retry them while they are under it: "
            "past it they are excluded, not retried)",
            first,
            len(context.failed_text_unit_ids[first]),
            start_stage_name,
            first,
        )
        return None
    logger.warning(
        "Stage '%s' failed on %d text units; resuming from it instead of '%s'",
        first,
        len(context.failed_text_unit_ids[first]),
        start_stage_name,
    )
    return first, pipeline._before(completed_stages, first)


def rewind_on_missing_records(
    pipeline: DataIngestionPipeline,
    context: PipelineContext,
    start_stage_name: str | None,
    completed_stages: list[str],
) -> tuple[str, list[str]] | None:
    """Move a reset resumed past the stages that store extraction records
    back to the first of them whose records the store no longer holds, or
    None to keep it.

    The restored text units name the records this pipeline's stages
    stored (graph records at graph_resolution, claim records at
    claim_extraction). Another pipeline id may have deleted them since
    (an incremental run deletes the records of the units it replaces, a
    full build keeps only the ones its units name). Indexed anyway, the
    units name records that are gone: the full build keeps none of them,
    and the next incremental run cannot derive the items those units
    cite. Run again, the stage extracts only the units whose records are
    gone and stores them, and every later stage runs again: the LLM
    response cache of the pipeline id answers their model calls, but with
    ``cache.llm_responses.enabled`` off those units, every description
    summary and every community report are generated (and billed) again. An
    incremental resume needs no check: another pipeline id's write makes
    its delta stale (``incremental.delta_still_applies``).

    Raises:
        PipelineResumeError: Records are missing and the stage that
            stores them is not enabled in this run.
    """
    if not pipeline.config.indexing.reset or start_stage_name not in pipeline._ORDER:
        return None
    stores = {
        "graph": PipelineStageType.GRAPH_RESOLUTION.value,
        "claims": PipelineStageType.CLAIM_EXTRACTION.value,
    }
    restored = pipeline._before(completed_stages, start_stage_name)
    checked = {kind: stage for kind, stage in stores.items() if stage in restored}
    if not checked:
        return None
    units = _with_record_keys(
        pipeline.config, context, context.translated_units or context.text_units
    )
    named: dict[str, dict[str, str]] = {}
    for unit in units:
        for kind, key in record_keys(unit).items():
            if kind in checked:
                named.setdefault(BaseIndexer.get_suffix(unit), {})[key] = kind
    if not named:
        return None
    missing: set[str] = set()
    try:
        indexer = pipeline.providers.build_vector_indexer()
        try:
            read = indexer.read_extraction_records
            if getattr(read, "__func__", read) is VectorIndexer.read_extraction_records:
                return None  # the backend keeps no records
            for suffix, kinds in named.items():
                stored = indexer.stored_extraction_record_keys(
                    sorted(kinds), suffix=suffix
                )
                missing |= {kinds[key] for key in kinds.keys() - stored}
        finally:
            indexer.close()
    except NamespaceLeaseError:
        raise
    except Exception as e:  # noqa: BLE001 - the indexing stage meets the store
        logger.warning("Checking the stored extraction records failed: %s", e)
        return None
    if not missing:
        return None
    first = min(
        (
            (
                PipelineStageType.GRAPH_EXTRACTION.value
                if kind == "graph"
                else checked[kind]
            )
            for kind in missing
        ),
        key=pipeline._ORDER.index,
    )
    if first not in {stage.name for stage in pipeline.stages}:
        raise PipelineResumeError(
            f"Extraction records this pipeline stored are gone (another "
            f"run replaced them), so '{first}' must run again. Re-run with "
            f"'{first}' enabled (--resume-from-stage {first})."
        )
    logger.warning(
        "Extraction records this pipeline stored are gone (another run "
        "replaced them); resuming from '%s' instead of '%s'",
        first,
        start_stage_name,
    )
    return first, pipeline._before(completed_stages, first)


def _refuse_rerun_of_unrun_failures(
    pipeline: DataIngestionPipeline,
    context: PipelineContext,
    start_stage_name: str,
    completed_stages: list[str],
    *,
    explicit: bool,
    resume_at: str | None = None,
    rerun: bool = False,
) -> None:
    """Refuse to run a completed stage again on the restored output of a
    stage that failed on text units and is not enabled in this run (an
    indexing-only phase run again, say), restored before ``resume_at``
    (the stage the resume rewinds to; default ``start_stage_name``).

    The completed run recorded those documents FAILED already; run
    again, the indexing stage would record them FAILED once more
    without retrying them, until ``indexing.max_document_failures``
    excludes them. Only the failed stage can retry them. An ``explicit``
    resume point is kept with a warning, like the rewind it replaces:
    run as asked, the indexing stage counts those failures again.
    ``rerun`` says the start stage completed in this pipeline id before
    (an explicit resume's ``completed_stages`` stop before its start).

    Raises:
        PipelineResumeError: An auto resume would count those failures
            again.
    """
    if pipeline._doc_status is None or not (
        rerun or start_stage_name in completed_stages
    ):
        return
    enabled = {stage.name for stage in pipeline.stages}
    unrun = pipeline._before(
        (
            name
            for name, unit_ids in context.failed_text_unit_ids.items()
            if unit_ids and name not in enabled
        ),
        resume_at or start_stage_name,
    )
    if not unrun:
        return
    first = min(unrun, key=pipeline._ORDER.index)
    message = (
        f"Stage '{first}' failed on "
        f"{len(context.failed_text_unit_ids[first])} text units and is not "
        f"enabled in this run: running '{start_stage_name}' again would "
        f"record their documents FAILED once more without retrying them"
    )
    if explicit:
        logger.warning("%s", message)
        return
    raise PipelineResumeError(
        f"{message}. Re-run with '{first}' enabled to retry them."
    )


def drop_unrun_failures(
    pipeline: DataIngestionPipeline,
    context: PipelineContext,
    start_stage_name: str | None,
) -> None:
    """Forget the failed text units the restored context holds for the
    stages this resumed run neither runs nor restores (not enabled, at or
    after ``start_stage_name``; gleaning switched off since, say): the run
    does not use their output, so the indexing stage must not record
    their documents FAILED again. A stage before the start point keeps
    them: its restored output is the run's (a phased run's earlier
    window)."""
    if start_stage_name not in pipeline._ORDER:
        return
    start = pipeline._ORDER.index(start_stage_name)
    enabled = {stage.name for stage in pipeline.stages}
    for name in list(context.failed_text_unit_ids):
        if (
            name in pipeline._ORDER
            and name not in enabled
            and pipeline._ORDER.index(name) >= start
        ):
            logger.info(
                "Stage '%s' is not enabled in this run: its failed text "
                "units of an earlier run are not recorded",
                name,
            )
            del context.failed_text_unit_ids[name]
            context.item_failed_text_unit_ids.pop(name, None)


def _rewind_on_extraction_change(
    pipeline: DataIngestionPipeline,
    pipeline_id: str,
    start_stage_name: str,
    completed_stages: list[str],
) -> tuple[str, list[str]]:
    """Move an auto resume back to document_loading when the extraction
    settings changed since its delta was computed.

    document_loading's cache key does not cover the extraction settings,
    so a resume past it keeps the previous run's delta and extraction
    fingerprint: the stages that rerun (a newly enabled stage, or one
    whose inputs changed) see only that delta's documents, and the
    commit records the old fingerprint, so the next run re-extracts
    every document. The same holds when the doc-status registry was
    turned off since: the restored delta would write records to the
    registry the user disabled. Rerunning document_loading diffs the
    corpus with this run's settings (or skips the diff without a
    registry); document_parsing stays cached.
    """
    loading = PipelineStageType.DOCUMENT_LOADING.value
    if (
        start_stage_name not in pipeline._ORDER
        or pipeline._ORDER.index(start_stage_name) <= pipeline._ORDER.index(loading)
        or loading not in {stage.name for stage in pipeline.stages}
    ):
        return start_stage_name, completed_stages
    metadata = pipeline.state_manager.load_pipeline_metadata(pipeline_id)
    recorded = metadata.get("incremental_extraction_settings")
    if recorded is None:
        return start_stage_name, completed_stages
    changed = changed_settings(
        ExtractionSettings.model_validate(recorded),
        extraction_settings(pipeline.config),
    )
    if not pipeline.providers.doc_status_enabled:
        reason = "the doc-status registry was disabled"
    elif changed:
        reason = f"the extraction settings changed ({', '.join(changed)})"
    else:
        return start_stage_name, completed_stages
    return _rewind_to_loading(
        pipeline,
        start_stage_name,
        completed_stages,
        f"{reason} since this pipeline computed its incremental delta",
    )


def _rewind_to_loading(
    pipeline: DataIngestionPipeline,
    start_stage_name: str,
    completed_stages: list[str],
    reason: str,
) -> tuple[str, list[str]]:
    """Resume from document_loading instead (it diffs the corpus again;
    document_parsing stays cached), logging ``reason``."""
    loading = PipelineStageType.DOCUMENT_LOADING.value
    logger.warning(
        "%s; resuming from '%s' instead of '%s'",
        reason[:1].upper() + reason[1:],
        loading,
        start_stage_name,
    )
    return loading, pipeline._before(completed_stages, loading)


def delta_is_stale(
    pipeline: DataIngestionPipeline,
    context: PipelineContext,
    start_stage_name: str | None,
) -> bool:
    """Whether a resume past document_loading would index a delta the
    registry no longer matches (``incremental.delta_still_applies``).

    Raises:
        PipelineResumeError: It would, and document_loading is not
            enabled in this run to compute the delta again.
    """
    if not pipeline._incremental_delta(context) or not pipeline._resumes_past_loading(
        start_stage_name
    ):
        return False
    assert context.incremental_delta is not None and pipeline._doc_status is not None
    if delta_still_applies(
        pipeline._doc_status,
        context.incremental_delta,
        context.incremental_fingerprints,
        pipeline.config.indexing.max_document_attempts,
        context.incremental_extraction_fingerprint,
        context.incremental_scope,
        context.pipeline_id,
        pipeline.config.indexing.max_document_failures,
        context.incremental_retired_scopes or (),
    ):
        return False
    loading = PipelineStageType.DOCUMENT_LOADING.value
    if loading not in {stage.name for stage in pipeline.stages}:
        raise PipelineResumeError(
            f"{_STALE_DELTA}, so it must be computed again. Re-run with "
            f"'{loading}' enabled (--resume-from-stage {loading})."
        )
    return True


def _check_upstream_complete(
    pipeline: DataIngestionPipeline, pipeline_id: str, start_stage_name: str
) -> None:
    """Fail when the resume point is a recorded stage upstream of every
    enabled stage: an earlier stage failed or lost its cached output, so
    this run's stages (a later phase, say) would run on missing inputs.
    A resume point downstream of the enabled stages means they are all
    done, and the run skips them."""
    if not pipeline.stages:
        return
    if start_stage_name not in pipeline._ORDER or start_stage_name in {
        stage.name for stage in pipeline.stages
    }:
        return
    first_enabled = pipeline._ORDER.index(pipeline.stages[0].name)
    if pipeline._ORDER.index(start_stage_name) > first_enabled:
        return
    metadata = pipeline.state_manager.load_pipeline_metadata(pipeline_id)
    recorded = {r.get("stage_name") for r in metadata.get("stage_results", [])}
    if start_stage_name in recorded:
        raise PipelineResumeError(
            f"Stage '{start_stage_name}', upstream of this run's enabled "
            f"stages, has not completed (or its cached output is gone). "
            f"Run it first, e.g. with --enabled-stages including "
            f"'{start_stage_name}'."
        )


def prepare_context_for_resume(
    pipeline: DataIngestionPipeline, context: PipelineContext, start_stage_name: str
) -> None:
    # Order by the canonical pipeline sequence (STAGE_CLASSES), NOT the
    # enabled-stage subset: phased execution narrows pipeline.stages to one
    # phase's window (e.g. just ["indexing"]), so judging "is this an
    # upstream result?" against that subset would wrongly discard every
    # earlier phase's results — silently dropping resolved_relationships /
    # resolved_claims before indexing. The full order keeps them.
    try:
        kept = set(
            pipeline._before(
                (res.stage_name for res in context.stage_results),
                start_stage_name,
            )
        )
    except ValueError:
        logger.warning(
            "Start stage '%s' not in canonical pipeline order; "
            "keeping all %s previous stage results",
            start_stage_name,
            len(context.stage_results),
        )
        return

    valid_results = [res for res in context.stage_results if res.stage_name in kept]
    context.stage_results = valid_results
    logger.info(
        "Context prepared for resume from '%s' - keeping %s previous stage results",
        start_stage_name,
        len(valid_results),
    )
