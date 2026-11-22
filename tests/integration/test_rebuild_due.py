# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild-due across whole-pipeline runs: a full build records the
settings its reports and summaries were written under, an incremental run
keeps them (it writes its delta's alone) and warns where they are not the
configuration's, and a reset clears the reason. AWS-free: the harness of
``test_community_counters``."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from tests.integration.test_community_counters import _Harness, _indexing_metrics
from unified_kg_rag.adapters.providers import Providers
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


def test_reports_an_earlier_release_wrote_are_due_whatever_the_record_says(
    tmp_path, caplog
) -> None:
    # R3A-1 (s17 lostrec): a 0.3.0 store whose record was lost. A run that
    # indexed nothing rewrote it dated by this release, so the reason read
    # off the record's version was gone, and global search stepped below
    # depth 0 onto the next delta alone. The reports tell it instead.
    from unified_kg_rag.application.storage.inspection import describe_namespace

    harness = _Harness(tmp_path)
    harness.config.indexing.community_compaction.max_community_growth = None
    harness.run("full")
    reports = harness.vectors.data["community_reports"]
    assert reports and not harness.vectors.holds_root_depth_reports("default")
    # As 0.3.0 wrote them: no min_cut_depth.
    for report in reports.values():
        report.min_cut_depth = None
    assert harness.vectors.holds_root_depth_reports("default")

    harness.vectors.builds.pop("default")
    caplog.set_level(logging.WARNING)
    rewritten = harness.run("no-op")
    record = harness.vectors.read_build_record("default")
    assert record is not None and record["mode"] == "full"
    # The release that built the stores is unknown: none is claimed.
    assert record["package_version"] is None
    ((due,),) = _indexing_metrics(rewritten)["rebuild_due"].values()
    assert (due["kind"], due["remedy"]) == ("community_depths", "reset")

    harness.add("carrier.txt", "Carrier ships widgets for Depot.")
    delta = harness.run("add-1")
    kinds = [d["kind"] for d in _indexing_metrics(delta)["rebuild_due"]["default"]]
    assert "community_depths" in kinds
    providers = Providers(
        harness.config,
        boto_session=MagicMock(),
        doc_status=harness.store,
        vector_indexer_factory=lambda: harness.vectors,
        graph_indexer_factory=lambda: harness.graph,
    )
    described = describe_namespace(harness.config, providers=providers)
    assert "community_depths" in [r.kind.value for r in described.rebuild]

    reset = harness.config.model_copy(deep=True)
    reset.indexing.reset = True
    harness.run("reset", reset)
    assert not harness.vectors.holds_root_depth_reports("default")
    assert describe_namespace(harness.config, providers=providers).rebuild == []
