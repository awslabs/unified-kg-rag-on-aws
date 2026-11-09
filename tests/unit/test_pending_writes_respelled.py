# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which stored entity a noted edge's re-keyed endpoint now is.

A removal can re-key an endpoint to another spelling between a rejected
write and the commit that retries it. A full build keeps spellings of one
name key apart when their types differ, so the graph may hold several
entities of the endpoint's key; the endpoint is the one a full build over
the edge's text units gives it, which cites those units (every endpoint
cites its edges' units).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.fixtures.fakes.stores import FakeGraphStore
from unified_kg_rag.application.storage.pending_writes import PendingWrites
from unified_kg_rag.domain.models import Entity, Relationship

pytestmark = pytest.mark.unit


def _respelled(
    stored: list[Entity], units: list[str]
) -> tuple[list[Entity], dict[str, tuple[str, str | None]]]:
    graph = FakeGraphStore()
    graph.upsert_entities(stored)
    pending = PendingWrites(SimpleNamespace(graph_indexer=graph))  # type: ignore[arg-type]
    edge = Relationship(
        id="r",
        source_id="gone",
        target_id="kept",
        source_name="Apple",
        target_name="Orchard",
        text_unit_ids=units,
    )
    return pending._respelled("default", {"gone"}, [edge], set())


# Two spellings of one key the types kept apart; the lower id first.
_COMPANY = Entity(
    id="a-company", name="Apple", type="ORGANIZATION", text_unit_ids=["u1"]
)
_FRUIT = Entity(id="b-fruit", name="APPLE", type="PRODUCT", text_unit_ids=["u2"])


@pytest.mark.parametrize("units", [["u2"], ["u2", "u3"]])
def test_the_endpoint_is_the_entity_its_edges_units_state(units: list[str]) -> None:
    respelled, moved = _respelled([_COMPANY, _FRUIT], units)
    assert moved == {"gone": ("b-fruit", "APPLE")}
    assert [e.id for e in respelled] == ["b-fruit"]


def test_the_most_shared_units_win_then_the_first_id() -> None:
    # Both cite one of the edge's units; the later id cites both.
    more = _FRUIT.model_copy(update={"text_unit_ids": ["u1", "u2"]})
    assert _respelled([_COMPANY, more], ["u1", "u2"])[1]["gone"][0] == "b-fruit"
    tied = _FRUIT.model_copy(update={"text_unit_ids": ["u1"]})
    assert _respelled([_COMPANY, tied], ["u1"])[1]["gone"][0] == "a-company"


def test_an_entity_sharing_no_unit_is_not_the_endpoint() -> None:
    # Neither spelling cites the edge's unit: a full build over it gives the
    # endpoint neither, so the edge is not drawn to one of them.
    assert _respelled([_COMPANY, _FRUIT], ["u9"]) == ([], {})
