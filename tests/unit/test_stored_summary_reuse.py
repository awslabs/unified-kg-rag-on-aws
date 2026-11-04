# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph resolution's reuse of stored summaries (``_reuse_stored_summaries``).

A full build gives the items over the summarization threshold the summaries
their stored copies were made from the same description with. Only a graph
of the whole corpus may, a stored copy the graph capped is never taken, and
a failed read reuses nothing. A fake graph indexer and summarizer stand in.
"""

from __future__ import annotations

import logging
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.domain.models import (
    Config,
    DocumentDelta,
    Entity,
    PipelineContext,
    PipelineStageStatus,
)

pytestmark = pytest.mark.unit

_LONG = " ".join(f"word{i}" for i in range(40))


class _Graph:
    def __init__(self, stored: list[Entity], error: Exception | None = None) -> None:
        self.stored = stored
        self.error = error
        self.reads = 0

    def read_entities(self, ids: list[str], suffix: Any = None) -> list[Entity]:
        self.reads += 1
        if self.error is not None:
            raise self.error
        return [e for e in self.stored if e.id in ids]

    def read_relationships(self, ids: list[str], suffix: Any = None) -> list[Any]:
        return []

    def close(self) -> None:
        return None


class _Summarizer:
    def __init__(self) -> None:
        self.offered: list[list[str]] = []

    def reuse_stored(self, items: list[Any], stored: list[Any]) -> set[str]:
        self.offered.append([s.id for s in stored])
        return {s.id for s in stored}


def _stage(graph: _Graph) -> tuple[ps.GraphResolutionStage, _Summarizer]:
    config = Config()
    summarization = config.processing.graph_extraction.description_summarization
    summarization.enabled = True
    summarization.force_summary_threshold_tokens = 5
    config.indexing.neptune.property_max_length = 400
    stage = ps.GraphResolutionStage.__new__(ps.GraphResolutionStage)
    summarizer = _Summarizer()
    stage.config = config
    stage.reuse_stored = True
    stage.providers = SimpleNamespace(build_graph_indexer=lambda: graph)  # type: ignore[assignment]
    stage.description_summarizer = summarizer  # type: ignore[assignment]
    return stage, summarizer


def _context(delta: DocumentDelta | None) -> PipelineContext:
    return PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory="/",
        incremental_delta=delta,
    )


def _entities(*ids: str) -> list[Entity]:
    return [Entity(id=i, name=i, description=_LONG) for i in ids]


@pytest.mark.parametrize(
    ("delta", "reads"),
    [
        (None, 1),
        (DocumentDelta(new=["d1", "d2"]), 1),
        (DocumentDelta(new=["d2"], unchanged=["d1"]), 0),
    ],
    ids=["no-registry", "first-run", "incremental"],
)
def test_only_a_graph_of_the_whole_corpus_reuses_stored_summaries(
    delta: DocumentDelta | None, reads: int
) -> None:
    # An incremental run's items hold its delta's descriptions only: a stored
    # summary of the whole description is not theirs.
    graph = _Graph(_entities("a"))
    stage, _ = _stage(graph)

    reused, _ = stage._reuse_stored_summaries(_context(delta), _entities("a"), [])

    assert graph.reads == reads
    assert reused == ({"a"} if reads else set())


def test_a_stored_copy_the_graph_capped_is_not_reused() -> None:
    # property_max_length cut the stored description: its summary was not
    # made from the description the item now holds.
    capped = Entity(id="b", name="b", description="x" * 400)
    graph = _Graph([*_entities("a"), capped])
    stage, summarizer = _stage(graph)

    reused, _ = stage._reuse_stored_summaries(_context(None), _entities("a", "b"), [])

    assert summarizer.offered[0] == ["a"]
    assert reused == {"a"}


def test_a_failed_read_reuses_nothing(caplog: pytest.LogCaptureFixture) -> None:
    graph = _Graph([], error=RuntimeError("graph unavailable"))
    stage, summarizer = _stage(graph)

    with caplog.at_level(logging.WARNING):
        reused = stage._reuse_stored_summaries(_context(None), _entities("a"), [])

    assert reused == (set(), set())
    assert summarizer.offered == []
    assert "Reading the stored summaries failed" in caplog.text
