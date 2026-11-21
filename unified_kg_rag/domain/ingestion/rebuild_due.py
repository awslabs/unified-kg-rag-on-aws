# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""When an index namespace is due for a rebuild: its build record against
the configuration, for what the next incremental run cannot apply.

An incremental run writes its delta: it clusters only the delta's entities
(``community_drift``), writes the reports and description summaries of the
delta alone, and builds a corpus-wide artifact the namespace lacks only by
rebuilding it. :func:`rebuild_reasons` reads, from the build record the
stores keep (``IndexBuildRecord``), what a run under ``config`` would leave
stale, each with what applies it:

- ``communities`` (``RebuildRemedy.RESET``): the community counters crossed
  an ``indexing.community_compaction`` threshold, or the run that wrote the
  record measured a drift that recommends one
  (``community_drift.reset_reasons``).
- ``outdated_reports`` (``RebuildRemedy.NEXT_RUN``), whatever the
  thresholds: the record lists community reports kept after a run removed
  text they were written from and could not write them again from what
  remains (``IndexBuildRecord.outdated_reports``): they may state it, and
  the next incremental run tries again (``report_refresh``).
- ``community_depths`` (reset): the community reports were written by a
  release before 0.4.0 (the full build the record is or keeps,
  ``package_version``), whose depths count from the coarsest root, not from
  the first Leiden pass as global search's ``community_depth`` cut now
  reads them; incremental runs write their delta's reports at the new
  depths beside them.
- ``answer_settings`` (reset): the community reports or description
  summaries the last full build wrote were made under other settings
  (prompt, model, language; ``cache_keys.corpus_answer_fingerprints``)
  than the configuration's; incremental runs keep them.
- ``extraction_settings`` (``RebuildRemedy.NEXT_RUN``): the extraction
  settings changed (``cache_keys.changed_settings``): the next run
  re-extracts every document.
- ``methodologies`` (next run): the configuration builds a corpus-wide
  artifact the namespace does not hold (``methodologies.
  missing_artifacts``): the next run rebuilds the namespace to add it.

A record from before a field was recorded tells nothing of it (no reason).
``run-describe``, the console and the MCP ``kg_describe_schema`` tool report
the reasons of a stored record; the indexing stage those of the record it
writes (its WARNING and the run's ``rebuild_due``, which the console's job
card reads). Nothing here reads a store or calls a model.
"""

import re
from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, ValidationError

from unified_kg_rag.domain.ingestion.cache_keys import (
    COMMUNITY_REPORT_ANSWERS,
    DESCRIPTION_SUMMARY_ANSWERS,
    changed_settings,
    corpus_answer_fingerprints,
    extraction_settings,
)
from unified_kg_rag.domain.ingestion.community_drift import reset_reasons
from unified_kg_rag.domain.ingestion.methodologies import (
    artifacts_to_build,
    held_artifacts,
    missing_artifacts,
)
from unified_kg_rag.domain.models import (
    CommunityCounters,
    Config,
    ExtractionSettings,
    IndexArtifact,
)


class RebuildRemedy(str, Enum):
    """What applies a rebuild reason."""

    # Only a run with indexing.reset: true (a full rebuild, reusing the
    # stored extractions and every answer made the way it would make it).
    RESET = "reset"
    # The next ingestion run, at the cost the reason names.
    NEXT_RUN = "next_run"


class RebuildReasonKind(str, Enum):
    COMMUNITIES = "communities"
    OUTDATED_REPORTS = "outdated_reports"
    COMMUNITY_DEPTHS = "community_depths"
    ANSWER_SETTINGS = "answer_settings"
    EXTRACTION_SETTINGS = "extraction_settings"
    METHODOLOGIES = "methodologies"


class RebuildReason(BaseModel):
    """One reason an index namespace is due for a rebuild."""

    kind: RebuildReasonKind = Field(description="What is stale")
    remedy: RebuildRemedy = Field(
        description="What applies it: reset (a run with indexing.reset: "
        "true) or next_run (the next ingestion run)"
    )
    detail: str = Field(description="Why, and what applying it costs")
    settings: list[str] = Field(
        default_factory=list,
        description="The configuration paths or artifacts behind it",
    )


# The ``rebuild`` field of every surface that describes a namespace
# (run-describe, the MCP kg_describe_schema tool, the graph lookup), declared
# once.
RebuildReasons = Annotated[
    list[RebuildReason],
    Field(
        default_factory=list,
        description="Why the namespace is due for a rebuild under the "
        "configuration (rebuild_due.rebuild_reasons): what the next "
        "incremental run cannot apply (stale communities, reports or "
        "summaries: remedy reset) or applies at a cost (changed extraction "
        "settings, a methodology it lacks: remedy next_run); empty when "
        "nothing is, or without a build record",
    ),
]


# The first release whose community depths count from the first Leiden pass
# (community_clustering._set_depths): a full build of an earlier one stored
# them counted from the coarsest root.
FIRST_PASS_DEPTH_RELEASE = (0, 4, 0)


_ANSWERS = {
    COMMUNITY_REPORT_ANSWERS: "community reports",
    DESCRIPTION_SUMMARY_ANSWERS: "description summaries",
}


def rebuild_reasons(
    record: Mapping[str, Any] | None,
    config: Config,
    *,
    drift_recommended: bool = False,
) -> list[RebuildReason]:
    """Why the namespace whose build record is ``record`` (an
    ``IndexBuildRecord`` dump; None without one) is due for a rebuild
    under ``config``, in the order of the module docstring; empty when
    nothing is (or without a record). ``drift_recommended``: the run that
    writes ``record`` measured a community drift that recommends a rebuild
    (a reason only under ``on_drift_recommendation``; the record keeps
    the reports it kept outdated, which are one whatever the setting)."""
    if not isinstance(record, Mapping):
        return []
    return [
        *_community_reasons(record, config, drift_recommended),
        *_outdated_report_reasons(record, config),
        *_community_depth_reasons(record, config),
        *_answer_reasons(record, config),
        *_extraction_reasons(record, config),
        *_methodology_reasons(record, config),
    ]


def _community_reasons(
    record: Mapping[str, Any], config: Config, drift_recommended: bool
) -> list[RebuildReason]:
    try:
        counters = CommunityCounters.model_validate(record.get("communities"))
    except ValidationError:
        return []
    crossed = reset_reasons(
        counters,
        config.indexing.community_compaction,
        drift_recommended=drift_recommended,
    )
    if not crossed:
        return []
    return [
        RebuildReason(
            kind=RebuildReasonKind.COMMUNITIES,
            remedy=RebuildRemedy.RESET,
            detail=f"its communities are due for re-clustering: "
            f"{'; '.join(crossed)}. Incremental runs append their delta's "
            "communities instead of re-clustering the graph; a run with "
            "indexing.reset: true re-clusters it, reusing the unchanged "
            "summaries and reports",
            settings=["indexing.community_compaction"],
        )
    ]


def _outdated_report_reasons(
    record: Mapping[str, Any], config: Config
) -> list[RebuildReason]:
    outdated = record.get("outdated_reports")
    if not isinstance(outdated, list) or not outdated:
        return []
    if IndexArtifact.COMMUNITY_REPORTS not in held_artifacts(record, config):
        return []
    one = len(outdated) == 1
    kept = (
        "1 community report was kept after documents it was"
        if one
        else f"{len(outdated)} community reports were kept after documents they were"
    )
    return [
        RebuildReason(
            kind=RebuildReasonKind.OUTDATED_REPORTS,
            remedy=RebuildRemedy.NEXT_RUN,
            detail=f"{kept} written from were deleted or changed, and the run "
            "could not write "
            + ("it" if one else "them")
            + " again from what remains: such a report may still state the "
            "removed text and is flagged outdated where it is read; the next "
            "incremental run tries again, a model call each (a run with "
            "indexing.reset: true writes every report)",
            settings=[IndexArtifact.COMMUNITY_REPORTS.value],
        )
    ]


def _community_depth_reasons(
    record: Mapping[str, Any], config: Config
) -> list[RebuildReason]:
    if IndexArtifact.COMMUNITY_REPORTS not in held_artifacts(record, config):
        return []
    full = record if record.get("mode") == "full" else record.get("full_build")
    if not isinstance(full, Mapping):
        return []
    version = full.get("package_version")
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version or "")
    if match is None or tuple(map(int, match.groups())) >= FIRST_PASS_DEPTH_RELEASE:
        return []
    return [
        RebuildReason(
            kind=RebuildReasonKind.COMMUNITY_DEPTHS,
            remedy=RebuildRemedy.RESET,
            detail=f"its community reports were written by {version}, which "
            "counted their depth from the coarsest root: global search's "
            "search.global_search.community_depth cut, which counts it from "
            "the first Leiden pass as MS GraphRAG does, reads them one or more "
            "levels too coarse beside the reports incremental runs write; a "
            "run with indexing.reset: true writes every report at its depth, "
            "reusing the stored extractions",
            settings=[IndexArtifact.COMMUNITY_REPORTS.value],
        )
    ]


def _answer_reasons(record: Mapping[str, Any], config: Config) -> list[RebuildReason]:
    recorded = record.get("answer_fingerprints")
    if not isinstance(recorded, Mapping):
        return []
    # Of what the namespace holds and the configuration builds: one it
    # would not write (reports of a methodology it does not build or the
    # namespace lacks, summarization off) leaves nothing stale.
    building = held_artifacts(record, config) & artifacts_to_build(config)
    current = corpus_answer_fingerprints(config, building)
    reasons = []
    for name, fingerprint in current.items():
        if fingerprint is None or name not in recorded:
            continue
        if recorded[name] == fingerprint:
            continue
        what = _ANSWERS[name]
        if recorded[name] is None:
            made = f"the last full build wrote no {what}, which the configuration makes"
            written = "writes them"
        else:
            made = (
                f"the {what} the last full build wrote were made under other "
                "settings (their model, prompt or language) than the "
                "configuration's"
            )
            written = "writes every one again"
        reasons.append(
            RebuildReason(
                kind=RebuildReasonKind.ANSWER_SETTINGS,
                remedy=RebuildRemedy.RESET,
                detail=f"{made}: incremental runs write the {what} of their "
                f"delta alone; a run with indexing.reset: true {written}, a "
                "model call each",
                settings=[name],
            )
        )
    return reasons


def _extraction_reasons(
    record: Mapping[str, Any], config: Config
) -> list[RebuildReason]:
    try:
        stored = ExtractionSettings.model_validate(record.get("extraction_settings"))
    except ValidationError:
        return []
    changed = changed_settings(stored, extraction_settings(config))
    if not changed:
        return []
    return [
        RebuildReason(
            kind=RebuildReasonKind.EXTRACTION_SETTINGS,
            remedy=RebuildRemedy.NEXT_RUN,
            detail="the extraction settings changed since the last run "
            f"({', '.join(changed)}): the next run extracts every document "
            "again (with the doc-status registry, as changed documents)",
            settings=changed,
        )
    ]


def _methodology_reasons(
    record: Mapping[str, Any], config: Config
) -> list[RebuildReason]:
    missing = missing_artifacts(record, config)
    if not missing:
        return []
    added = sorted(a.value for a in missing)
    return [
        RebuildReason(
            kind=RebuildReasonKind.METHODOLOGIES,
            remedy=RebuildRemedy.NEXT_RUN,
            detail=f"the configuration builds {', '.join(added)}, which the "
            "namespace does not hold: the next run rebuilds it from the "
            "whole corpus, reusing the stored extractions, so only the "
            "model calls that build what it adds are new",
            settings=added,
        )
    ]
