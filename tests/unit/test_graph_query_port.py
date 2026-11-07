# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GraphQueryPort`` and its in-memory fake (docs/design.md §6.6)."""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.graph_query import FakeGraphQueryRetriever, graph_of
from tests.fixtures.fakes.stores import FakeGraphStore
from tests.fixtures.graph_query import (
    CONFORMANCE_PLANS,
    procurement_entities,
    procurement_relationships,
    validated,
)
from unified_kg_rag.domain.models.graph_query import GraphQueryRequest
from unified_kg_rag.domain.retrieval.graph_query import hydrate_keys
from unified_kg_rag.ports import GraphQueryPort

pytestmark = pytest.mark.unit


def _store() -> FakeGraphStore:
    store = FakeGraphStore()
    for suffix in ("alpha", "beta"):
        store.index_entities(procurement_entities(suffix), suffixes=[suffix])
        store.index_relationships(procurement_relationships(suffix))
    # beta loses a supply edge: the suffixes differ.
    store.delete_by_id(["r1"], suffix="beta")
    return store


def test_the_fake_is_a_graph_query_port() -> None:
    assert isinstance(FakeGraphQueryRetriever({}), GraphQueryPort)


async def test_each_suffix_is_its_own_graph() -> None:
    store = _store()
    fake = FakeGraphQueryRetriever(
        {suffix: graph_of(store, suffix) for suffix in ("alpha", "beta")}
    )
    plan = validated(CONFORMANCE_PLANS["count_supply_pairs"])
    counts = {}
    for suffix in ("alpha", "beta", "gamma"):
        request = GraphQueryRequest(
            plan=plan,
            suffix=suffix,
            operation="count",
            vars=plan.vars,
            hydrate=hydrate_keys(plan),
        )
        counts[suffix] = (await fake.aexecute(request)).count
    assert counts == {"alpha": 6, "beta": 5, "gamma": 0}


async def test_the_profile_is_sampled_and_lists_what_the_graph_holds() -> None:
    fake = FakeGraphQueryRetriever(graph_of(_store(), "alpha"))
    whole = await fake.adescribe(
        None, sample_vertices=100, edges_per_vertex=10, max_examples=10
    )
    assert fake.describes == [None]
    # The default suffix holds nothing here: "alpha" is keyed None.
    assert not whole.sampled and whole.sample_size == 10
    assert [t.label for t in whole.entity_types] == ["VENDOR", "BUYER", "PRODUCT"]
    assert {t.label for t in whole.edge_types} == {"SUPPLIES", "MAKES", "BUYS"}
    assert ("vendor", "supplies", "buyer") in {
        (p.source, p.edge, p.target) for p in whole.patterns
    }
    part = await fake.adescribe(
        None, sample_vertices=3, edges_per_vertex=1, max_examples=10
    )
    assert part.sampled and part.sample_size == 3


async def test_a_failure_is_raised_as_given() -> None:
    from unified_kg_rag.shared.exceptions import GraphQueryExecutionError

    fake = FakeGraphQueryRetriever({}, fail=GraphQueryExecutionError("refused"))
    plan = validated(CONFORMANCE_PLANS["count_supply_pairs"])
    with pytest.raises(GraphQueryExecutionError):
        await fake.aexecute(
            GraphQueryRequest(plan=plan, operation="count", vars=plan.vars)
        )
