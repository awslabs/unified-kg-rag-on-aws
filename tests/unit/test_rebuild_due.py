# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``rebuild_due.rebuild_reasons``: the build record against the config,
for what the next incremental run cannot apply."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from unified_kg_rag.domain.ingestion.cache_keys import (
    COMMUNITY_REPORT_ANSWERS,
    DESCRIPTION_SUMMARY_ANSWERS,
    corpus_answer_fingerprints,
    extraction_settings,
)
from unified_kg_rag.domain.ingestion.methodologies import artifacts_to_build
from unified_kg_rag.domain.ingestion.rebuild_due import (
    RebuildReasonKind,
    RebuildRemedy,
    rebuild_reasons,
)
from unified_kg_rag.domain.models import (
    CommunityCounters,
    Config,
    IndexArtifact,
    IndexBuildRecord,
    Methodology,
)

pytestmark = pytest.mark.unit


def _record(config: Config, **update: Any) -> dict[str, Any]:
    """A full build's record under ``config``, as the stores keep it."""
    built = artifacts_to_build(config)
    settings = extraction_settings(config)
    record = IndexBuildRecord(
        pipeline_id="full",
        mode="full",
        built_at=datetime.now(UTC),
        documents=2,
        extraction_fingerprint=settings.fingerprint,
        extraction_settings=settings,
        answer_fingerprints=corpus_answer_fingerprints(config, built),
    ).with_artifacts(built)
    return record.model_copy(update=update).model_dump(mode="json")


def _kinds(record: dict[str, Any] | None, config: Config) -> list[RebuildReasonKind]:
    return [r.kind for r in rebuild_reasons(record, config)]


def test_a_namespace_built_by_the_config_is_not_due() -> None:
    config = Config()
    assert rebuild_reasons(_record(config), config) == []
    assert rebuild_reasons(None, config) == []


def test_a_record_from_before_the_fields_tells_nothing() -> None:
    config = Config()
    old = {
        k: v
        for k, v in _record(config).items()
        if k not in {"answer_fingerprints", "artifacts", "methodologies"}
    }
    changed = config.model_copy(deep=True)
    changed.models.roles.community_report = changed.tier_model_id("fast")
    assert rebuild_reasons(old, changed) == []


def test_crossed_community_counters_need_a_reset() -> None:
    config = Config()
    counters = CommunityCounters(
        clustered_at=datetime.now(UTC),
        communities_at_clustering=10,
        communities_now=20,
        runs_since=3,
    )
    (reason,) = rebuild_reasons(_record(config, communities=counters), config)
    assert reason.kind is RebuildReasonKind.COMMUNITIES
    assert reason.remedy is RebuildRemedy.RESET
    assert "max_community_growth: 1.5" in reason.detail
    assert "indexing.reset: true" in reason.detail
    config.indexing.community_compaction.max_community_growth = None
    assert rebuild_reasons(_record(config, communities=counters), config) == []


def test_reports_under_another_model_or_prompt_need_a_reset() -> None:
    config = Config()
    record = _record(config)
    changed = config.model_copy(deep=True)
    changed.models.roles.community_report = changed.tier_model_id("fast")
    (reason,) = rebuild_reasons(record, changed)
    assert reason.kind is RebuildReasonKind.ANSWER_SETTINGS
    assert reason.remedy is RebuildRemedy.RESET
    assert reason.settings == [COMMUNITY_REPORT_ANSWERS]
    assert "community reports" in reason.detail
    prompted = config.model_copy(deep=True)
    prompted.custom_prompts.description_summarization_system = (
        "Summarize the descriptions in {language}."
    )
    (reason,) = rebuild_reasons(record, prompted)
    assert reason.settings == [DESCRIPTION_SUMMARY_ANSWERS]


def test_answers_the_config_would_not_write_are_not_stale() -> None:
    config = Config()
    record = _record(config)
    # Summarization switched off: nothing the next runs write disagrees.
    off = config.model_copy(deep=True)
    off.processing.graph_extraction.description_summarization.enabled = False
    off.models.roles.description_summary = off.tier_model_id("fast")
    assert RebuildReasonKind.ANSWER_SETTINGS not in _kinds(record, off)
    # Switched on since a build without it: the stored ones are unsummarized.
    built_off = _record(off)
    assert _kinds(built_off, config) == [RebuildReasonKind.ANSWER_SETTINGS]


def test_a_record_that_fingerprinted_fewer_answers_tells_nothing_of_the_others() -> (
    None
):
    # A record whose release fingerprinted only some answers says nothing of
    # the rest, as a record from before the field says nothing of any.
    config = Config()
    record = _record(config)
    record["answer_fingerprints"] = {
        COMMUNITY_REPORT_ANSWERS: record["answer_fingerprints"][
            COMMUNITY_REPORT_ANSWERS
        ]
    }
    prompted = config.model_copy(deep=True)
    prompted.custom_prompts.description_summarization_system = (
        "Summarize the descriptions in {language}."
    )
    assert rebuild_reasons(record, prompted) == []
    # The answers it did fingerprint are still compared.
    prompted.models.roles.community_report = prompted.tier_model_id("fast")
    (reason,) = rebuild_reasons(record, prompted)
    assert reason.settings == [COMMUNITY_REPORT_ANSWERS]


def test_changed_extraction_settings_are_applied_by_the_next_run() -> None:
    config = Config()
    changed = config.model_copy(deep=True)
    changed.processing.chunking.max_chunk_size = 6000
    (reason,) = rebuild_reasons(_record(config), changed)
    assert reason.kind is RebuildReasonKind.EXTRACTION_SETTINGS
    assert reason.remedy is RebuildRemedy.NEXT_RUN
    assert reason.settings == ["processing.chunking.max_chunk_size"]


def test_a_methodology_the_namespace_lacks_is_added_by_the_next_run() -> None:
    lightrag = Config()
    lightrag.indexing.methodologies = [Methodology.LIGHTRAG]
    record = _record(lightrag)
    (reason,) = rebuild_reasons(record, Config())
    assert reason.kind is RebuildReasonKind.METHODOLOGIES
    assert reason.remedy is RebuildRemedy.NEXT_RUN
    assert {
        IndexArtifact.COMMUNITIES.value,
        IndexArtifact.COMMUNITY_REPORTS.value,
    } <= set(reason.settings)


def test_an_incremental_record_keeps_its_full_build_s_answers() -> None:
    config = Config()
    full = _record(config)
    changed = config.model_copy(deep=True)
    changed.models.roles.community_report = changed.tier_model_id("fast")
    delta = IndexBuildRecord(
        pipeline_id="delta",
        mode="incremental",
        built_at=datetime.now(UTC),
        documents=1,
        extraction_fingerprint="x",
        answer_fingerprints=None,
    )
    written = delta.over(full)
    assert written.answer_fingerprints == full["answer_fingerprints"]
    # So the delta's settings do not hide the stored reports' staleness.
    assert _kinds(written.model_dump(mode="json"), changed) == [
        RebuildReasonKind.ANSWER_SETTINGS
    ]
    # A full build replaces them.
    rebuilt = IndexBuildRecord.model_validate(_record(changed)).over(full)
    assert rebuild_reasons(rebuilt.model_dump(mode="json"), changed) == []


def test_reports_a_narrower_config_does_not_write_are_not_stale() -> None:
    # Regression (W2 review E2): the reasons read what the namespace holds,
    # not what the config builds of it, so a LightRAG-only config (whose run
    # removes the reports) reported stale community reports.
    config = Config()
    record = _record(config)
    lightrag = config.model_copy(deep=True)
    lightrag.indexing.methodologies = [Methodology.LIGHTRAG]
    lightrag.models.roles.community_report = lightrag.tier_model_id("fast")
    reasons = rebuild_reasons(record, lightrag)
    assert all(COMMUNITY_REPORT_ANSWERS not in r.settings for r in reasons), reasons
    # The same change under a config that builds them is a reason.
    graphrag = config.model_copy(deep=True)
    graphrag.models.roles.community_report = graphrag.tier_model_id("fast")
    assert [r.settings for r in rebuild_reasons(record, graphrag)] == [
        [COMMUNITY_REPORT_ANSWERS]
    ]


def test_summaries_switched_on_since_the_build_say_so() -> None:
    # W2 review E5: "were made under other settings" was wrong for answers
    # the full build never wrote.
    off = Config()
    off.processing.graph_extraction.description_summarization.enabled = False
    (reason,) = rebuild_reasons(_record(off), Config())
    assert "wrote no description summaries" in reason.detail
    # None was written: a reset writes them, not "every one again".
    assert "again" not in reason.detail and "writes them" in reason.detail
    changed = Config()
    changed.models.roles.community_report = changed.tier_model_id("fast")
    (reason,) = rebuild_reasons(_record(Config()), changed)
    assert "under other settings (their model, prompt or language)" in reason.detail
