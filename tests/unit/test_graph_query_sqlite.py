# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SQLite compiler, the SQL checker and the read-only execution of the
embedded graph's structured queries (docs/design.md §6.6)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings

from tests.fixtures.graph_query import (
    CONFORMANCE_PLANS,
    procurement_entities,
    procurement_graph,
    procurement_relationships,
    profile_of,
    validated,
)
from tests.property.test_graph_query_properties import plans
from unified_kg_rag.adapters.graph_query.checks import check_sql
from unified_kg_rag.adapters.graph_query.sqlite import SQLiteQuery
from unified_kg_rag.adapters.local.sqlite_graph import SQLiteGraphStore
from unified_kg_rag.adapters.retrievers.sqlite_graph_retriever import (
    SQLiteGraphRetriever,
)
from unified_kg_rag.adapters.storage.sqlite_graph_indexer import SQLiteGraphIndexer
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.graph_query import GraphQueryPlan, GraphQueryRequest
from unified_kg_rag.domain.retrieval.graph_query import (
    hydrate_keys,
    plan_operation,
    validate_plan,
)
from unified_kg_rag.shared.exceptions import (
    GraphQueryCompileError,
    GraphQueryExecutionError,
    InvalidGraphQueryError,
    StoreConfigurationError,
)

pytestmark = pytest.mark.unit

LABEL = "Entity-default"


def _query(raw: str | dict, **fields: Any) -> SQLiteQuery:
    plan = validated(CONFORMANCE_PLANS[raw] if isinstance(raw, str) else raw)
    op = plan_operation(plan, scan=False, cap=50)
    request = GraphQueryRequest(
        plan=plan,
        operation=op.operation,
        vars=op.vars,
        group_var=op.group_var,
        group_key=op.group_key,
        limit=op.limit,
        hydrate=hydrate_keys(plan),
        **fields,
    )
    return SQLiteQuery(request, LABEL, vertex_ok=None, edge_ok=None)


@pytest.mark.parametrize("name", sorted(CONFORMANCE_PLANS))
def test_every_compiled_plan_passes_the_checker(name: str) -> None:
    query = _query(name)
    check_sql(query.sql, query.params, LABEL)
    assert "'" not in query.sql and '"' not in query.sql
    assert query.params.count(LABEL) == query.sql.count(".label = ?")


def test_a_both_hop_is_an_out_and_an_in_branch() -> None:
    query = _query("two_hop_any")
    # 1..3 hops each way: 2 + 4 + 8 branches.
    assert query.sql.count("SELECT b0.id") == 14
    assert "e2.target_id" in query.sql and "e2.source_id" in query.sql


def test_values_are_bound_parameters() -> None:
    injected = "x'; DROP TABLE vertices; --"
    query = _query(
        {
            "nodes": [
                {"var": "v", "mention": injected},
                {"var": "b", "types": ["BUYER"]},
            ],
            "edges": [{"types": ["SUPPLIES"]}],
            "returns": {"kind": "count"},
        },
        linked={"v": (injected,)},
    )
    assert injected not in query.sql
    assert injected in query.params
    assert "DROP" not in query.sql


@pytest.mark.parametrize(
    "sql,params",
    [
        ("DELETE FROM vertices", []),
        ("SELECT 1; DELETE FROM vertices", []),
        ("SELECT id FROM vertices AS v WHERE v.label = ?", ["Entity-other"]),
        ("SELECT id FROM vertices AS v", []),
        ("SELECT id FROM vertices", []),
        ("SELECT id FROM sqlite_master", []),
        ("SELECT id FROM vertices AS v WHERE v.label = 'Entity-default'", []),
        ("SELECT id FROM vertices AS v WHERE v.label = ? -- x", [LABEL]),
        ("SELECT load_extension(?) FROM vertices AS v WHERE v.label = ?", ["x", LABEL]),
        (
            "SELECT json_extract(v.props, ?) FROM vertices AS v WHERE v.label = ?",
            ["$", LABEL],
        ),
        ("ATTACH DATABASE ? AS x", ["/tmp/x"]),
        ("PRAGMA query_only = OFF", []),
        ("SELECT id FROM vertices AS v WHERE v.label = :label", []),
        (
            "SELECT a.id FROM vertices AS a JOIN edges AS e ON e.source_id = a.id "
            "WHERE a.label = ?",
            [LABEL],
        ),
        ("SELECT id FROM vertices AS v WHERE v.label = ?", [LABEL, "extra"]),
    ],
)
def test_the_checker_refuses_writes_and_escapes(sql: str, params: list) -> None:
    with pytest.raises(GraphQueryCompileError):
        check_sql(sql, params, LABEL)


@settings(max_examples=200)
@given(plans())
def test_every_valid_plan_compiles_to_a_checked_statement(raw: dict) -> None:
    try:
        plan = validate_plan(
            GraphQueryPlan.model_validate(raw),
            profile_of(procurement_graph()),
            max_hops=4,
        )
    except InvalidGraphQueryError:
        return
    for scan in (False, True):
        op = plan_operation(plan, scan=scan, cap=10)
        request = GraphQueryRequest(
            plan=plan,
            operation=op.operation,
            vars=op.vars,
            group_var=op.group_var,
            group_key=op.group_key,
            limit=op.limit,
        )
        query = SQLiteQuery(request, LABEL, vertex_ok=None, edge_ok=lambda p: True)
        check_sql(query.sql, query.params, LABEL)
        assert query.params.count(LABEL) == query.sql.count(".label = ?")


# --- read-only execution -----------------------------------------------------------


@pytest.fixture(params=["file", "memory"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> SQLiteGraphStore:
    path = tmp_path / "graph.sqlite3" if request.param == "file" else ":memory:"
    config = Config()
    config.providers.graph.kind = "sqlite"
    config.providers.graph.sqlite_path = str(path)
    graph = SQLiteGraphStore(path)
    indexer = SQLiteGraphIndexer(config, store=graph)
    indexer.index_entities(procurement_entities(), suffixes=["default"])
    indexer.index_relationships(procurement_relationships())
    return graph


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM vertices",
        "INSERT INTO vertices (label, id, props) VALUES ('x', 'y', '{}')",
        "CREATE TABLE stolen (id TEXT)",
    ],
)
def test_the_read_only_connection_refuses_a_write(
    store: SQLiteGraphStore, statement: str
) -> None:
    """Even a statement the checker never lets through cannot write."""
    with pytest.raises((StoreConfigurationError, sqlite3.Error)):
        store.query_read_only(statement, [], functions={}, timeout_seconds=5)
    with store.read("count") as connection:
        assert connection.execute("SELECT count(*) FROM vertices").fetchone()[0] == 10
    # The store's own connection still writes (query_only was for the query).
    with store.transaction("write") as connection:
        connection.execute("DELETE FROM edges WHERE id = ?", ("r1",))


def test_the_progress_handler_stops_a_long_query(store: SQLiteGraphStore) -> None:
    long = (
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) "
        "SELECT count(*) FROM n"
    )
    with pytest.raises(GraphQueryExecutionError) as error:
        store.query_read_only(long, [], functions={}, timeout_seconds=0.2)
    assert error.value.timed_out


def test_functions_are_registered_for_one_query_only(store: SQLiteGraphStore) -> None:
    rows = store.query_read_only(
        "SELECT kg_node(?, v.id, v.props) FROM vertices AS v WHERE v.label = ?",
        [0, LABEL],
        functions={"kg_node": (3, lambda *_: 1)},
        timeout_seconds=5,
    )
    assert rows and all(row == (1,) for row in rows)
    # Afterwards the name no longer runs the function (sqlite3 cannot
    # unregister one; it is replaced by one that fails).
    with pytest.raises((GraphQueryExecutionError, StoreConfigurationError)):
        store.query_read_only(
            "SELECT kg_node(?, v.id, v.props) FROM vertices AS v WHERE v.label = ?",
            [0, LABEL],
            functions={},
            timeout_seconds=5,
        )


async def test_the_retriever_answers_a_plan(store: SQLiteGraphStore) -> None:
    config = Config()
    retriever = SQLiteGraphRetriever(config, store=store)
    plan = validated(CONFORMANCE_PLANS["buyers_per_vendor"])
    op = plan_operation(plan, scan=False, cap=50)
    result = await retriever.aexecute(
        GraphQueryRequest(
            plan=plan,
            operation=op.operation,
            vars=op.vars,
            group_var=op.group_var,
            group_key=op.group_key,
            limit=op.limit,
            hydrate=hydrate_keys(plan),
        )
    )
    assert {(g.key, g.count) for g in result.groups} == {
        ("v1", 2),
        ("v2", 1),
        ("v3", 3),
    }
    # The group's vertices come back with their properties.
    assert result.vertices["v3"]["name"] == ["가나다 상사"]
    assert result.backend == "sqlite" and result.compiled_digest
