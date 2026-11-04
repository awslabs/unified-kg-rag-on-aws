# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Community counters of the build record and the reset recommendation
(``IndexBuildRecord.over``, ``community_drift.reset_reasons``)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pytest

from unified_kg_rag.domain.ingestion.community_drift import (
    CommunityDrift,
    commit_identity,
    reset_reasons,
)
from unified_kg_rag.domain.models import (
    CommunityChange,
    CommunityCounters,
    Config,
    DocumentDelta,
    IndexBuildRecord,
)
from unified_kg_rag.domain.models.config import CommunityCompactionConfig

pytestmark = pytest.mark.unit

_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _full(communities: int = 10, commit: str | None = None) -> IndexBuildRecord:
    return IndexBuildRecord(
        pipeline_id="full",
        mode="full",
        built_at=_AT,
        documents=5,
        extraction_fingerprint="fp",
        communities=CommunityCounters(
            clustered_at=_AT,
            communities_at_clustering=communities,
            communities_now=communities,
            runs_since=0,
            last_commit=commit,
        ),
    )


def _delta(commit: str, added: int, deleted: int = 0) -> IndexBuildRecord:
    return IndexBuildRecord(
        pipeline_id=f"run-{commit}",
        mode="incremental",
        built_at=_AT,
        documents=1,
        extraction_fingerprint="fp",
        community_change=CommunityChange(commit=commit, added=added, deleted=deleted),
    )


def _stored(record: IndexBuildRecord) -> dict[str, Any]:
    return record.model_dump(mode="json")


def test_incremental_commits_add_to_the_full_builds_counters() -> None:
    stored = _full(10)
    for commit, added, deleted in (("a", 4, 0), ("b", 3, 2), ("c", 0, 1)):
        stored = _delta(commit, added, deleted).over(_stored(stored))
    counters = stored.communities
    assert counters is not None
    assert counters.communities_at_clustering == 10
    assert counters.communities_now == 14
    assert counters.runs_since == 3
    assert counters.clustered_at == _AT
    assert counters.growth_ratio == 1.4
    # The kept full build is not stored with counters of its own.
    assert stored.full_build is not None and stored.full_build.communities is None


def test_the_same_commit_is_counted_once() -> None:
    once = _delta("a", 4).over(_stored(_full(10)))
    again = _delta("a", 4).over(_stored(once))
    assert again.communities == once.communities
    # Another commit after it is counted.
    later = _delta("b", 2).over(_stored(again))
    assert later.communities is not None
    assert (later.communities.communities_now, later.communities.runs_since) == (16, 2)


def test_a_full_build_starts_the_counters_again() -> None:
    drifted = _delta("a", 30).over(_stored(_full(10)))
    rebuilt = _full(12).over(_stored(drifted))
    assert rebuilt.communities is not None
    assert rebuilt.communities.runs_since == 0
    assert rebuilt.communities.communities_now == 12


def test_a_record_without_counters_starts_none() -> None:
    # Written before the counters existed: they start at the next full build.
    old = _full(10).model_copy(update={"communities": None})
    assert _delta("a", 4).over(_stored(old)).communities is None
    assert _delta("a", 4).over(None).communities is None
    assert _full(10).over(_stored(old)).communities == _full(10).communities


def test_a_delta_written_full_record_adds_to_stored_counters() -> None:
    # The registry knew none of the scope's documents (mode "full"), but
    # the suffix holds what another scope wrote: counted as a commit.
    first = _full(4, commit="x").model_copy(
        update={"community_change": CommunityChange(commit="x", added=4, deleted=0)}
    )
    assert first.over(None).communities == first.communities
    resumed = first.over(_stored(first))
    assert resumed.communities == first.communities
    stored = _delta("a", 6).over(_stored(_full(10)))
    other_scope = _full(4, commit="y").model_copy(
        update={"community_change": CommunityChange(commit="y", added=4, deleted=0)}
    )
    counters = other_scope.over(_stored(stored)).communities
    assert counters is not None
    assert (counters.communities_now, counters.runs_since) == (20, 2)
    assert counters.communities_at_clustering == 10


def test_the_change_is_not_stored() -> None:
    assert "community_change" not in _stored(_delta("a", 1))


def test_communities_now_never_goes_negative() -> None:
    counters = _delta("a", 0, deleted=50).over(_stored(_full(10))).communities
    assert counters is not None and counters.communities_now == 0


def _counters(at: int, now: int, runs: int) -> CommunityCounters:
    return CommunityCounters(
        clustered_at=_AT,
        communities_at_clustering=at,
        communities_now=now,
        runs_since=runs,
    )


@pytest.mark.parametrize(
    ("counters", "expected"),
    [
        (_counters(10, 14, 19), []),
        (_counters(10, 14, 20), ["20 incremental runs"]),
        (_counters(10, 15, 3), ["15 communities against the 10 it clustered, 1.5x"]),
        (_counters(10, 30, 25), ["25 incremental runs", "3x"]),
        # Nothing clustered and nothing appended: no communities to compare.
        (_counters(0, 0, 30), []),
        # Appended to a build that clustered none: only the run count applies.
        (_counters(0, 5, 20), ["20 incremental runs"]),
    ],
)
def test_reset_reasons_at_the_default_thresholds(
    counters: CommunityCounters, expected: list[str]
) -> None:
    reasons = reset_reasons(counters, CommunityCompactionConfig())
    assert len(reasons) == len(expected)
    for reason, phrase in zip(reasons, expected, strict=True):
        assert phrase in reason


def test_a_null_threshold_turns_its_check_off() -> None:
    compaction = CommunityCompactionConfig(
        max_runs_since_clustering=None, max_community_growth=None
    )
    assert reset_reasons(_counters(10, 100, 100), compaction) == []


def test_the_drift_recommendation_counts_only_when_asked() -> None:
    counters = _counters(10, 11, 1)
    assert reset_reasons(counters, CommunityCompactionConfig(), True) == []
    asked = CommunityCompactionConfig(on_drift_recommendation=True)
    assert reset_reasons(counters, asked, False) == []
    assert reset_reasons(counters, asked, True) == [
        "this run's community drift recommends a rebuild"
    ]


def test_growth_must_exceed_one() -> None:
    with pytest.raises(ValueError):
        CommunityCompactionConfig(max_community_growth=1.0)
    with pytest.raises(ValueError):
        CommunityCompactionConfig(max_runs_since_clustering=0)


def test_commit_identity_follows_the_pipeline_and_the_delta() -> None:
    delta = DocumentDelta(new=["a"], changed=["b"], deleted=["c"], unchanged=["d"])
    fingerprints = {"a": "1", "b": "2", "d": "4"}
    same = commit_identity("p", delta, fingerprints)
    assert commit_identity("p", delta, dict(fingerprints)) == same
    # An unchanged document's content is no part of the commit.
    assert commit_identity("p", delta, {**fingerprints, "d": "5"}) == same
    assert commit_identity("q", delta, fingerprints) != same
    assert commit_identity("p", delta, {**fingerprints, "b": "3"}) != same
    other = delta.model_copy(update={"deleted": []})
    assert commit_identity("p", other, fingerprints) != same


def _drift(recommended: bool) -> CommunityDrift:
    return CommunityDrift(
        delta_communities=1,
        clustered_entities=1,
        overlapping_entities=1,
        stale_communities=1,
        added_roots=0,
        overlap_share=1.0,
        rebuild_recommended=recommended,
    )


def _stage(config: Config) -> Any:
    from unified_kg_rag.application.ingestion import pipeline_stages as ps

    stage = ps.IndexingStage.__new__(ps.IndexingStage)
    stage.config = config
    return stage


def test_the_stage_warns_once_per_suffix_with_the_numbers(caplog) -> None:
    stage = _stage(Config())
    counters = {"default": _counters(10, 16, 4), "tenant": _counters(8, 8, 20)}
    with caplog.at_level(logging.WARNING):
        metrics = stage._check_community_counters(counters, None)
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 2
    assert "'default'" in warnings[0] and "16 communities against the 10" in warnings[0]
    assert "'tenant'" in warnings[1] and "20 incremental runs" in warnings[1]
    assert all("indexing.reset: true" in w for w in warnings)
    assert metrics["community_growth_ratio"] == 1.6
    assert metrics["runs_since_clustering"] == 20
    assert metrics["community_counters"]["default"]["communities_now"] == 16


def test_the_stage_stays_quiet_below_the_thresholds(caplog) -> None:
    config = Config()
    config.indexing.community_compaction.on_drift_recommendation = True
    stage = _stage(config)
    with caplog.at_level(logging.WARNING):
        metrics = stage._check_community_counters(
            {"default": _counters(10, 11, 1)}, _drift(False)
        )
        assert caplog.records == []
        stage._check_community_counters({"default": _counters(10, 11, 1)}, _drift(True))
    assert len(caplog.records) == 1
    assert metrics["community_growth_ratio"] == 1.1
    assert metrics["runs_since_clustering"] == 1
