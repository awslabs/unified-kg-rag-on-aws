# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One ``GraphQueryPort`` contract for every graph backend (docs/design.md
§6.6).

The same plans run on the in-memory reference (``fake``), the embedded
SQLite graph, Gremlin Server (``LOCAL_STORES=1``) and Neptune (``-m aws``),
the last two also with custom vertex ids, over two suffixes holding
different graphs; every answer is compared with the reference executor's
over that suffix's graph. A difference is a bug in one of them, whichever is
right. Seeded random plans add cases no one wrote by hand.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.graph_query import FakeGraphQueryRetriever, graph_of
from tests.fixtures.graph_query import (
    CONFORMANCE_PLANS,
    procurement_entities,
    procurement_relationships,
    profile_of,
    property_graph,
    reference_operation,
)
from tests.fixtures.store_backends import (
    GRAPH_BACKENDS,
    StoreHarness,
    open_harness,
)
from unified_kg_rag.adapters.retrievers.graph_expansion import (
    GraphExpansionRetriever,
    edge_predicate,
)
from unified_kg_rag.adapters.retrievers.sqlite_graph_retriever import vertex_filter
from unified_kg_rag.domain.models import Entity, Relationship
from unified_kg_rag.domain.models.graph_query import (
    GraphQueryPlan,
    GraphQueryRequest,
    PropertyGraph,
    QueryAnswer,
    ValidatedPlan,
)
from unified_kg_rag.domain.retrieval.graph_query import (
    aggregate_bindings,
    answer_from_result,
    hydrate_keys,
    plan_operation,
    reference_execute,
    validate_plan,
)
from unified_kg_rag.ports import GraphQueryPort
from unified_kg_rag.shared.exceptions import (
    GraphQueryExecutionError,
    InvalidGraphQueryError,
)

pytestmark = pytest.mark.integration


@pytest.fixture(params=GRAPH_BACKENDS)
def harness(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StoreHarness]:
    yield from open_harness(request.param, tmp_path)


def _beta_relationships(suffix: str) -> list[Relationship]:
    # beta: one supply edge fewer and one more product.
    rels = [r for r in procurement_relationships(suffix) if r.id != "r1"]
    rels.append(
        Relationship(
            id="r13",
            source_id="v4",
            target_id="p4",
            type="MAKES",
            attributes={"index": suffix},
        )
    )
    return rels


def _beta_entities(suffix: str) -> list[Entity]:
    return [
        *procurement_entities(suffix),
        Entity(id="p4", name="Seal", type="PRODUCT", attributes={"index": suffix}),
    ]


class _Graphs:
    """The two suffixes written to a harness, with their reference graphs."""

    def __init__(self, harness: StoreHarness) -> None:
        self.harness = harness
        self.alpha = harness.suffix("alpha")
        self.beta = harness.suffix("beta")
        written = {
            self.alpha: (
                procurement_entities(self.alpha),
                procurement_relationships(self.alpha),
            ),
            self.beta: (_beta_entities(self.beta), _beta_relationships(self.beta)),
        }
        self.reference: dict[str, PropertyGraph] = {}
        for suffix, (entities, rels) in written.items():
            harness.graph_indexer.index_entities(entities, suffixes=[suffix])
            stats = harness.graph_indexer.index_relationships(rels)
            assert stats.failed_items == 0, stats.errors
            self.reference[suffix] = property_graph(entities, rels)

    def port(self) -> Any:
        if self.harness.kind == "fake":
            return FakeGraphQueryRetriever(
                {s: graph_of(self.harness.graph_indexer, s) for s in self.reference}
            )
        assert self.harness.graph_retriever is not None
        return self.harness.graph_retriever()


def _request(
    plan: ValidatedPlan,
    suffix: str,
    *,
    scan: bool = False,
    cap: int = 1000,
    **fields: Any,
) -> GraphQueryRequest:
    op = plan_operation(plan, scan=scan, cap=cap)
    return GraphQueryRequest(
        plan=plan,
        suffix=suffix,
        operation=op.operation,
        vars=op.vars,
        group_var=op.group_var,
        group_key=op.group_key,
        order=op.order,
        limit=op.limit,
        hydrate=hydrate_keys(plan),
        **fields,
    )


def _answer(
    plan: ValidatedPlan, request: GraphQueryRequest, result: Any, scan: bool
) -> QueryAnswer:
    if not scan:
        return answer_from_result(plan, result)
    rows = result.rows or tuple(
        dict(zip(plan.vars, (p.vertex_ids[0], p.vertex_ids[-1]), strict=True))
        for p in result.paths
    )
    return aggregate_bindings(
        plan, rows, result.vertices, paths=result.paths, truncated=result.truncated
    )


def _expected(
    plan: ValidatedPlan, request: GraphQueryRequest, graph: PropertyGraph, scan: bool
) -> QueryAnswer:
    filters = GraphExpansionRetriever._scope_filters_to_label("x", request.filters)
    result = reference_operation(
        request,
        graph,
        vertex_ok=vertex_filter(filters),
        edge_ok=edge_predicate(filters),
    )
    return _answer(plan, request, result, scan)


async def _close(port: Any) -> None:
    close = getattr(port, "aclose", None)
    if close is not None:
        await close()


def _plan(raw: dict[str, Any], graph: PropertyGraph) -> ValidatedPlan:
    return validate_plan(
        GraphQueryPlan.model_validate(raw), profile_of(graph), max_hops=4
    )


async def _run_all(
    graphs: _Graphs, cases: list[tuple[str, dict[str, Any], dict[str, Any]]]
) -> list[str]:
    port = graphs.port()
    assert isinstance(port, GraphQueryPort)
    failures: list[str] = []
    try:
        for name, raw, fields in cases:
            for suffix, graph in graphs.reference.items():
                try:
                    plan = _plan(raw, graph)
                except InvalidGraphQueryError:
                    continue
                scan = bool(fields.get("scan")) or plan.reads_seed_properties
                request = _request(plan, suffix, **fields)
                got = _answer(plan, request, await port.aexecute(request), scan)
                want = _expected(plan, request, graph, scan)
                if got != want:
                    failures.append(f"{name} @ {suffix}: {got} != {want}")
    finally:
        await _close(port)
    return failures


def test_every_backend_answers_as_the_reference(harness: StoreHarness) -> None:
    graphs = _Graphs(harness)
    cases = [(name, raw, {}) for name, raw in sorted(CONFORMANCE_PLANS.items())]
    # The same plans scanned, as under a document filter.
    cases += [
        (f"{name}+scan", raw, {"scan": True})
        for name, raw in sorted(CONFORMANCE_PLANS.items())
    ]
    assert asyncio.run(_run_all(graphs, cases)) == []


def test_caller_filters_apply_alike(harness: StoreHarness) -> None:
    graphs = _Graphs(harness)
    filter_sets: list[dict[str, Any]] = [
        {"filters": {"source_kinds": "document"}},
        {"filters": {"source_kinds": ["seed_graph"]}},
        {"filters": {"type": ["VENDOR", "BUYER"]}},
        {"filters": {"attr_role": "supplier"}},
        {"filters": {"rank": {"gte": 3}}},
        {"allowed_ids": ("v1", "v3", "b1", "b2", "p1")},
    ]
    cases = [
        (f"{name}/{index}", CONFORMANCE_PLANS[name], fields)
        for index, fields in enumerate(filter_sets)
        for name in ("count_supply_pairs", "buyers_per_vendor", "two_hop_any")
    ]
    assert asyncio.run(_run_all(graphs, cases)) == []


def _random_plans(seed: int, count: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    types, edges = ["VENDOR", "BUYER", "PRODUCT"], ["SUPPLIES", "MAKES", "BUYS"]
    found = []
    while len(found) < count:
        size = rng.randint(1, 3)
        nodes = []
        for i in range(size):
            node: dict[str, Any] = {"var": f"n{i}"}
            if rng.random() < 0.6:
                node["types"] = [rng.choice(types)]
            if rng.random() < 0.3:
                node["where"] = [
                    {
                        "prop": "rank",
                        "op": rng.choice(["gte", "lt"]),
                        "value": rng.randint(1, 5),
                    }
                ]
            nodes.append(node)
        hops = [
            {
                "types": rng.sample(edges, rng.randint(0, 2)),
                "direction": rng.choice(["out", "in", "both"]),
                "hops": [1, rng.randint(1, 2)],
            }
            for _ in range(size - 1)
        ]
        var = rng.choice(nodes)["var"]
        kind = rng.choice(
            ["rows", "count", "count_distinct", "group_count", "aggregate", "exists"]
            + (["path"] if size == 2 else [])
        )
        returns: dict[str, Any] = {"kind": kind, "limit": 50}
        if kind == "count_distinct":
            returns["vars"] = [var]
        elif kind == "group_count":
            returns["group_by"] = {"var": var, "prop": rng.choice([None, "type"])}
        elif kind == "aggregate":
            returns["aggregate"] = {
                "fn": rng.choice(["min", "max", "sum", "avg"]),
                "var": var,
                "prop": "rank",
            }
        found.append({"nodes": nodes, "edges": hops, "returns": returns})
    return found


def test_random_plans_answer_as_the_reference(harness: StoreHarness) -> None:
    graphs = _Graphs(harness)
    cases = [(f"random-{i}", raw, {}) for i, raw in enumerate(_random_plans(7, 40))]
    assert asyncio.run(_run_all(graphs, cases)) == []


def test_a_capped_result_is_marked_truncated(harness: StoreHarness) -> None:
    graphs = _Graphs(harness)
    graph = graphs.reference[graphs.alpha]
    plan = _plan(CONFORMANCE_PLANS["count_supply_pairs"], graph)
    request = _request(plan, graphs.alpha, scan=True, cap=4)

    async def run() -> Any:
        port = graphs.port()
        try:
            return await port.aexecute(request)
        finally:
            await _close(port)

    result = asyncio.run(run())
    assert result.truncated and len(result.rows) == 4
    answer = _answer(plan, request, result, True)
    assert answer.truncated and answer.count == 4


def test_the_profile_lists_what_the_graph_holds(harness: StoreHarness) -> None:
    graphs = _Graphs(harness)

    async def run() -> Any:
        port = graphs.port()
        try:
            return await port.adescribe(
                graphs.beta, sample_vertices=1000, edges_per_vertex=50, max_examples=10
            )
        finally:
            await _close(port)

    profile = asyncio.run(run())
    expected = profile_of(graphs.reference[graphs.beta])
    assert [(t.label, t.count) for t in profile.entity_types] == [
        (t.label, t.count) for t in expected.entity_types
    ]
    assert {(t.label, t.count) for t in profile.edge_types} == {
        (t.label, t.count) for t in expected.edge_types
    }
    assert {p.name: p.kinds for p in profile.properties} == {
        p.name: p.kinds for p in expected.properties
    }
    assert not profile.sampled


def test_a_query_past_its_time_limit_falls_back(harness: StoreHarness) -> None:
    if harness.kind == "fake":
        pytest.skip("the reference executor has no time limit")
    suffix = harness.suffix("dense")
    n = 40
    entities = [
        Entity(id=f"d{i}", name=f"D{i}", type="NODE", attributes={"index": suffix})
        for i in range(n)
    ]
    rels = [
        Relationship(
            id=f"d{i}-{j}",
            source_id=f"d{i}",
            target_id=f"d{j}",
            type="LINK",
            attributes={"index": suffix},
        )
        for i in range(n)
        for j in range(n)
        if i != j
    ]
    harness.graph_indexer.index_entities(entities, suffixes=[suffix])
    harness.graph_indexer.index_relationships(rels)
    graph = property_graph(entities, rels)
    plan = _plan(
        {
            "nodes": [{"var": "a"}, {"var": "b"}],
            "edges": [{"direction": "both", "hops": [4, 4]}],
            "returns": {"kind": "count"},
        },
        graph,
    )
    assert harness.graph_retriever is not None
    build = harness.graph_retriever

    async def run() -> None:
        port = build()
        try:
            await port.aexecute(_request(plan, suffix, timeout_seconds=0.3))
        finally:
            await port.aclose()

    with pytest.raises(GraphQueryExecutionError) as error:
        asyncio.run(run())
    assert error.value.timed_out


def _ranked_graph(suffix: str) -> tuple[list[Entity], list[Relationship]]:
    """Vendor k (rank k) supplies k buyers, k = 1..8: the top vendor is v8."""
    entities, rels = [], []
    for k in range(1, 9):
        entities.append(
            Entity(
                id=f"v{k}",
                name=f"V{k}",
                type="VENDOR",
                rank=k,
                attributes={"index": suffix},
            )
        )
        for j in range(k):
            buyer = f"b{k}-{j}"
            entities.append(
                Entity(
                    id=buyer,
                    name=buyer.upper(),
                    type="BUYER",
                    attributes={"index": suffix},
                )
            )
            rels.append(
                Relationship(
                    id=f"r{k}-{j}",
                    source_id=f"v{k}",
                    target_id=buyer,
                    type="SUPPLIES",
                    attributes={"index": suffix},
                )
            )
    return entities, rels


CAPPED_PLANS = {
    "top_vendor": {
        "nodes": [{"var": "v", "types": ["VENDOR"]}, {"var": "b", "types": ["BUYER"]}],
        "edges": [{"types": ["SUPPLIES"]}],
        "returns": {
            "kind": "group_count",
            "group_by": {"var": "v"},
            "vars": ["b"],
            "limit": 1,
        },
    },
    "fewest": {
        "nodes": [{"var": "v", "types": ["VENDOR"]}, {"var": "b", "types": ["BUYER"]}],
        "edges": [{"types": ["SUPPLIES"]}],
        "returns": {
            "kind": "group_count",
            "group_by": {"var": "v"},
            "vars": ["b"],
            "order": "asc",
            "limit": 2,
        },
    },
    "top_ranked_vendors": {
        "nodes": [{"var": "v", "types": ["VENDOR"]}],
        "returns": {"kind": "rows", "limit": 2},
    },
}


@pytest.mark.parametrize("name", sorted(CAPPED_PLANS))
def test_a_capped_store_page_keeps_the_first_by_the_plan_order(
    harness: StoreHarness, name: str
) -> None:
    """With a page (cap) shorter than the matches, every store cuts in the
    plan's order, so the answer is the exact top, not an arbitrary subset."""
    suffix = harness.suffix("capped")
    entities, rels = _ranked_graph(suffix)
    harness.graph_indexer.index_entities(entities, suffixes=[suffix])
    harness.graph_indexer.index_relationships(rels)
    graph = property_graph(entities, rels)
    plan = _plan(CAPPED_PLANS[name], graph)
    request = _request(plan, suffix, cap=3)

    async def run() -> Any:
        port = (
            FakeGraphQueryRetriever({suffix: graph_of(harness.graph_indexer, suffix)})
            if harness.kind == "fake"
            else harness.graph_retriever()  # type: ignore[misc]
        )
        try:
            return await port.aexecute(request)
        finally:
            await _close(port)

    result = asyncio.run(run())
    assert result.truncated
    answer = answer_from_result(plan, result)
    full = reference_execute(plan, graph)
    assert not answer.truncated
    assert (answer.groups, answer.rows) == (full.groups, full.rows)
    if name == "top_vendor":
        assert [(g.key, g.count) for g in answer.groups] == [("v8", 8)]


MIXED_NUMBER_PLANS = {
    "per_size": {
        "nodes": [{"var": "v", "types": ["VENDOR"]}],
        "returns": {
            "kind": "group_count",
            "group_by": {"var": "v", "prop": "attr.size"},
        },
    },
    "size_five": {
        "nodes": [
            {
                "var": "v",
                "types": ["VENDOR"],
                "where": [{"prop": "attr.size", "op": "eq", "value": 5}],
            }
        ],
        "returns": {"kind": "count"},
    },
    "size_in": {
        "nodes": [
            {
                "var": "v",
                "where": [{"prop": "attr.size", "op": "in", "value": [5.0, 7.5]}],
            }
        ],
        "returns": {"kind": "rows"},
    },
    "largest": {
        "nodes": [{"var": "v", "types": ["VENDOR"]}],
        "returns": {
            "kind": "aggregate",
            "aggregate": {"fn": "max", "var": "v", "prop": "attr.size"},
        },
    },
}


@pytest.mark.parametrize("name", sorted(MIXED_NUMBER_PLANS))
def test_an_integral_float_is_the_int_it_equals(
    harness: StoreHarness, name: str
) -> None:
    """A property stored as 5 on some vertices and 5.0 on others (two
    extraction runs, a seed file) is one value: one group, one match."""
    suffix = harness.suffix("mixed")
    entities = [
        Entity(
            id=f"m{i}",
            name=f"M{i}",
            type="VENDOR",
            rank=1,
            attributes={"index": suffix, "size": size},
        )
        for i, size in enumerate([5.0, 5, 5, 7.5, 5.0])
    ]
    harness.graph_indexer.index_entities(entities, suffixes=[suffix])
    graph = property_graph(entities, [])
    plan = _plan(MIXED_NUMBER_PLANS[name], graph)
    request = _request(plan, suffix)

    async def run() -> Any:
        port = (
            FakeGraphQueryRetriever({suffix: graph_of(harness.graph_indexer, suffix)})
            if harness.kind == "fake"
            else harness.graph_retriever()  # type: ignore[misc]
        )
        try:
            return await port.aexecute(request)
        finally:
            await _close(port)

    answer = answer_from_result(plan, asyncio.run(run()))
    assert answer == _expected(plan, request, graph, False)
    if name == "per_size":
        assert [(g.key, g.count) for g in answer.groups] == [(5, 4), (7.5, 1)]
        assert type(answer.groups[0].key) is int
    elif name == "size_five":
        assert answer.count == 4
