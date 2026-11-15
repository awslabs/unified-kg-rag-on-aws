# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild-due across whole-pipeline runs: a full build records the
settings its reports and summaries were written under, an incremental run
keeps them (it writes its delta's alone) and warns where they are not the
configuration's, and a reset clears the reason. AWS-free: the harness of
``test_community_counters``."""

from __future__ import annotations

import logging

import pytest

from tests.integration.test_community_counters import _Harness, _indexing_metrics
from unified_kg_rag.domain.ingestion.cache_keys import (
    COMMUNITY_REPORT_ANSWERS,
    corpus_answer_fingerprints,
)
from unified_kg_rag.domain.ingestion.methodologies import artifacts_to_build
from unified_kg_rag.domain.ingestion.rebuild_due import (
    RebuildReasonKind,
    rebuild_reasons,
)

pytestmark = pytest.mark.integration

_DUE = "is due for a rebuild"


def test_reports_written_under_other_settings_stay_due_until_a_reset(
    tmp_path, caplog
) -> None:
    harness = _Harness(tmp_path)
    # A two-document corpus doubles its communities in one commit.
    harness.config.indexing.community_compaction.max_community_growth = None
    harness.run("full")
    built = harness.vectors.builds["default"]
    assert built.answer_fingerprints == corpus_answer_fingerprints(
        harness.config, artifacts_to_build(harness.config)
    )
    assert built.answer_fingerprints[COMMUNITY_REPORT_ANSWERS]
    assert (
        rebuild_reasons(harness.vectors.read_build_record("default"), harness.config)
        == []
    )

    # Another report model: the next incremental run reports its delta's
    # communities with it, and keeps the stored ones.
    changed = harness.config.model_copy(deep=True)
    changed.models.roles.community_report = changed.tier_model_id("fast")
    caplog.set_level(logging.WARNING)
    harness.add("carrier.txt", "Carrier ships widgets for Depot.")
    context = harness.run("add-1", changed)
    kept = harness.vectors.builds["default"]
    assert kept.mode == "incremental"
    assert kept.answer_fingerprints == built.answer_fingerprints
    (reason,) = rebuild_reasons(harness.vectors.read_build_record("default"), changed)
    assert reason.kind is RebuildReasonKind.ANSWER_SETTINGS
    warnings = [r.getMessage() for r in caplog.records if _DUE in r.getMessage()]
    assert len(warnings) == 1 and "community reports" in warnings[0]
    # The run's result says so too (the console's job card offers --reset).
    ((due,),) = _indexing_metrics(context)["rebuild_due"].values()
    assert (due["kind"], due["remedy"]) == ("answer_settings", "reset")

    reset = changed.model_copy(deep=True)
    reset.indexing.reset = True
    harness.run("reset", reset)
    assert rebuild_reasons(harness.vectors.read_build_record("default"), changed) == []
