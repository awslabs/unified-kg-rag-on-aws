# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Only resolved claims reach the graph, the claim index and the lineage.

An extracted claim carries empty ``subject_id`` / ``object_id`` until claim
resolution ties it to an entity. When no claim resolves (or resolution did
not run), the stages must index no claim rather than fall back to the raw
extraction output.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from tests.fixtures.fakes.stores import FakeVectorStore
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.domain.ingestion.claim_resolver import ClaimResolutionStats
from unified_kg_rag.domain.models import (
    Claim,
    Config,
    Entity,
    PipelineContext,
    PipelineStageStatus,
    TextUnit,
)
from unified_kg_rag.ports.indexer import IndexingStats

pytestmark = pytest.mark.unit


def _raw_claim() -> Claim:
    # As ClaimExtractor emits it: names only, no entity ids yet.
    return Claim(
        id="c-raw",
        subject_id="",
        subject_name="Vendor",
        object_id="",
        object_name="Buyer",
        type="SUPPLY",
        description="Vendor supplies Buyer",
        text_unit_ids=["t1"],
    )


def _resolved_claim() -> Claim:
    return _raw_claim().model_copy(
        update={"id": "c-ok", "subject_id": "e-vendor", "object_id": "e-buyer"}
    )


def _context(**fields: Any) -> PipelineContext:
    ctx = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory="/corpus",
    )
    ctx.text_units = [TextUnit(id="t1", text="Vendor supplies Buyer.")]
    ctx.resolved_entities = [
        Entity(id="e-vendor", name="Vendor", type="ORG", text_unit_ids=["t1"]),
        Entity(id="e-buyer", name="Buyer", type="ORG", text_unit_ids=["t1"]),
    ]
    for name, value in fields.items():
        setattr(ctx, name, value)
    return ctx


class _RecordingManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.opensearch_indexer = FakeVectorStore()

    def initialize(self) -> None:
        return None

    def index_all_data(self, **kwargs: Any) -> dict[str, IndexingStats]:
        self.calls.append(kwargs)
        return {"entities": IndexingStats(total_items=2, successful_items=2)}


def _indexed_claims(mocker, ctx: PipelineContext) -> list[Claim]:
    manager = _RecordingManager()
    mocker.patch.object(ps, "IndexingManager", return_value=manager)
    ps.IndexingStage(config=Config())._execute_core(ctx)
    assert len(manager.calls) == 1
    return manager.calls[0]["claims"]


def test_no_claim_resolves_so_none_is_indexed(mocker) -> None:
    ctx = _context(claims=[_raw_claim()])
    stage = ps.ClaimResolutionStage(Config())
    mocker.patch.object(
        stage.resolver, "resolve", return_value=([], ClaimResolutionStats())
    )

    stage._execute_core(ctx)

    assert ctx.resolved_claims == []
    assert _indexed_claims(mocker, ctx) == []
    graph = ps._build_knowledge_graph(ctx)
    assert "c-raw" not in graph.nodes
    assert "" not in graph.nodes


def test_skipped_resolution_clears_a_stale_result(mocker) -> None:
    # No resolved entities: resolution is skipped, and a result restored from
    # an earlier run must not survive it.
    ctx = _context(claims=[_raw_claim()], resolved_claims=[_resolved_claim()])
    ctx.resolved_entities = []

    ps.ClaimResolutionStage(Config())._execute_core(ctx)

    assert ctx.resolved_claims == []


def test_resolution_disabled_indexes_no_raw_claim(mocker) -> None:
    # claim_resolution switched off through stages_enabled: extraction ran,
    # resolution did not, so nothing ties the claims to the graph.
    ctx = _context(claims=[_raw_claim()])

    assert _indexed_claims(mocker, ctx) == []


def test_resolved_claims_are_indexed(mocker) -> None:
    ctx = _context(claims=[_raw_claim()], resolved_claims=[_resolved_claim()])

    assert [c.id for c in _indexed_claims(mocker, ctx)] == ["c-ok"]


def test_a_claim_without_subject_id_is_never_indexed(mocker) -> None:
    ctx = _context(resolved_claims=[_resolved_claim(), _raw_claim()])

    assert [c.id for c in _indexed_claims(mocker, ctx)] == ["c-ok"]
