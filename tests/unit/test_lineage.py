# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One lineage merge (domain/retrieval/lineage.py) for every strategy that
names the units its graph items rest on."""

from __future__ import annotations

import threading

import pytest

from unified_kg_rag.adapters.retrieval import base
from unified_kg_rag.adapters.retrieval.base import evidence_lineage, units_by_support
from unified_kg_rag.adapters.search_strategies.local_search import (
    LocalSearchStrategy,
)
from unified_kg_rag.domain.models import RetrievalResult
from unified_kg_rag.domain.retrieval.lineage import merge

pytestmark = pytest.mark.unit


def _entity(i: int, score: float, units: object) -> RetrievalResult:
    return RetrievalResult(
        content=f"E{i}",
        score=score,
        source=f"e{i}",
        retriever_type="entity",
        metadata={"id": f"e{i}", "text_unit_ids": units},
    )


ENTITIES = [
    _entity(1, 0.9, ["u9", "u3"]),
    _entity(2, 0.5, ["u3", "u1"]),
    _entity(3, 0.4, ["u1", "u2"]),
]


def test_first_seen_keeps_the_items_order() -> None:
    assert merge([["u9", "u3"], ["u3", "u1"], ["u1", "u2"]]) == ["u9", "u3", "u1", "u2"]
    assert evidence_lineage(ENTITIES)["text_unit_ids"] == ["u9", "u3", "u1", "u2"]


def test_support_ranks_by_citing_items_then_the_best_score() -> None:
    lineages = [["u9", "u3"], ["u3", "u1"], ["u1", "u2"]]
    # u3 and u1 have two citing items each; u3's best (0.9) beats u1's (0.5).
    assert merge(lineages, order="support", scores=[0.9, 0.5, 0.4]) == [
        "u3",
        "u1",
        "u9",
        "u2",
    ]
    # A tie goes to the best citing item, not to the first seen.
    assert merge(
        [["a"], ["b"], ["b", "a"]], order="support", scores=[0.1, 0.9, 0.5]
    ) == [
        "b",
        "a",
    ]
    # Without scores, ties keep the first-seen order.
    assert merge([["b", "a"], ["a", "b"]], order="support") == ["b", "a"]
    # An item citing a unit twice supports it once.
    assert merge([["x", "x"], ["y"], ["y"]], order="support") == ["y", "x"]


async def test_local_and_structured_rank_their_units_alike() -> None:
    """The unify probe: local ranked (count, best score), structured
    (count, id), so one result's citation and chunk fetch disagreed. Both
    now rank through ``units_by_support``."""
    expected = ["u3", "u1", "u9", "u2"]
    assert await units_by_support(ENTITIES) == expected
    assert await LocalSearchStrategy._rank_text_unit_ids(ENTITIES) == expected


async def test_a_json_text_lineage_is_read_as_its_list() -> None:
    """A graph store may return the list as its JSON text: local dropped it,
    structured split it into characters."""
    nodes = [_entity(4, 0.9, '["u7", "u8"]')]
    assert evidence_lineage(nodes)["text_unit_ids"] == ["u7", "u8"]
    assert await units_by_support(nodes) == ["u7", "u8"]
    assert await LocalSearchStrategy._rank_text_unit_ids(nodes) == ["u7", "u8"]


async def test_the_support_merge_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression (rv25): local search merged its hub lineages (thousands of
    # units per hub entity) on the event loop, 160-615 ms per query that
    # stalled every concurrent one; LightRAG's pick already ran on a thread.
    loop_thread = threading.get_ident()
    threads: list[int] = []
    merge_units = base._units_by_support

    def recording(results: list[RetrievalResult]) -> list[str]:
        threads.append(threading.get_ident())
        return merge_units(results)

    monkeypatch.setattr(base, "_units_by_support", recording)
    assert await LocalSearchStrategy._rank_text_unit_ids(ENTITIES)
    assert threads and loop_thread not in threads
