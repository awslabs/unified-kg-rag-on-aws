# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GraphBuilder pure logic (AWS-free).

Covers building a networkx graph from entities/relationships/claims: node and
edge typing, orphan relationship skipping, self-loops, duplicate ids, claim
subject/object connection (full/partial/none), literal-object claims, and the
empty-input edge cases. The graph is the observable surface, so assertions are
made against node/edge contents rather than internal stats dictionaries.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.ingestion.graph_builder import GraphBuilder
from unified_kg_rag.domain.models import Claim, Entity, Relationship

pytestmark = pytest.mark.unit


def _entity(id_: str, name: str | None = None, **kw) -> Entity:
    return Entity(id=id_, name=name or id_, **kw)


def _rel(id_: str, src: str, tgt: str, **kw) -> Relationship:
    return Relationship(id=id_, source_id=src, target_id=tgt, **kw)


def _claim(
    id_: str,
    subject_id: str,
    object_id: str | None,
    object_name: str = "obj",
    **kw,
) -> Claim:
    return Claim(
        id=id_,
        subject_id=subject_id,
        subject_name="subj",
        object_id=object_id,
        object_name=object_name,
        type="HAS",
        **kw,
    )


# --------------------------------------------------------------------------- #
# Entity nodes
# --------------------------------------------------------------------------- #
class TestEntityNodes:
    def test_entities_become_nodes_typed_entity(self) -> None:
        g = GraphBuilder([_entity("e1", "Alice"), _entity("e2", "Bob")], []).build()
        assert set(g.nodes) == {"e1", "e2"}
        assert g.nodes["e1"]["node_type"] == "entity"
        assert g.nodes["e1"]["name"] == "Alice"

    def test_empty_input_yields_empty_graph(self) -> None:
        g = GraphBuilder([], []).build()
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0

    def test_single_entity_no_edges(self) -> None:
        g = GraphBuilder([_entity("e1")], []).build()
        assert g.number_of_nodes() == 1
        assert g.number_of_edges() == 0

    def test_duplicate_entity_ids_collapse_to_one_node(self) -> None:
        # networkx keys nodes by id; the later entity's attributes win.
        g = GraphBuilder([_entity("e1", "First"), _entity("e1", "Second")], []).build()
        assert g.number_of_nodes() == 1
        assert g.nodes["e1"]["name"] == "Second"


# --------------------------------------------------------------------------- #
# Relationship edges
# --------------------------------------------------------------------------- #
class TestRelationshipEdges:
    def test_edge_added_between_existing_entities(self) -> None:
        ents = [_entity("e1"), _entity("e2")]
        g = GraphBuilder(ents, [_rel("r1", "e1", "e2", type="KNOWS")]).build()
        assert g.has_edge("e1", "e2")
        assert g.edges["e1", "e2"]["edge_type"] == "relationship"
        assert g.edges["e1", "e2"]["type"] == "KNOWS"

    def test_orphan_relationship_missing_source_skipped(self) -> None:
        # Source not an entity node -> edge dropped, no phantom node created.
        g = GraphBuilder([_entity("e2")], [_rel("r1", "GONE", "e2")]).build()
        assert g.number_of_edges() == 0
        assert "GONE" not in g.nodes

    def test_orphan_relationship_missing_target_skipped(self) -> None:
        g = GraphBuilder([_entity("e1")], [_rel("r1", "e1", "GONE")]).build()
        assert g.number_of_edges() == 0
        assert "GONE" not in g.nodes

    def test_self_loop_added_when_entity_exists(self) -> None:
        # _both_nodes_exist is satisfied by a single node, so self-loops survive.
        g = GraphBuilder([_entity("e1")], [_rel("r1", "e1", "e1")]).build()
        assert g.has_edge("e1", "e1")
        assert g.number_of_edges() == 1

    def test_relationships_with_no_entities_all_skipped(self) -> None:
        rels = [_rel("r1", "e1", "e2"), _rel("r2", "e2", "e3")]
        g = GraphBuilder([], rels).build()
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0

    def test_parallel_edges_collapse_in_simple_graph(self) -> None:
        # nx.Graph is simple: a second edge between the same pair overwrites.
        ents = [_entity("e1"), _entity("e2")]
        rels = [
            _rel("r1", "e1", "e2", weight=1.0),
            _rel("r2", "e1", "e2", weight=5.0),
        ]
        g = GraphBuilder(ents, rels).build()
        assert g.number_of_edges() == 1
        assert g.edges["e1", "e2"]["weight"] == 5.0


# --------------------------------------------------------------------------- #
# Claims
# --------------------------------------------------------------------------- #
class TestClaims:
    def test_claim_node_typed_claim_and_fully_connected(self) -> None:
        ents = [_entity("e1"), _entity("e2")]
        claims = [_claim("c1", "e1", "e2")]
        g = GraphBuilder(ents, [], claims).build()
        assert g.nodes["c1"]["node_type"] == "claim"
        assert g.edges["e1", "c1"]["edge_type"] == "is_subject_of"
        assert g.edges["e2", "c1"]["edge_type"] == "is_object_of"

    def test_claim_partial_connection_when_object_missing(self) -> None:
        # Subject exists, object_id not an entity -> only subject edge made.
        ents = [_entity("e1")]
        claims = [_claim("c1", "e1", "MISSING")]
        g = GraphBuilder(ents, [], claims).build()
        assert g.has_edge("e1", "c1")
        assert g.degree("c1") == 1

    def test_claim_literal_object_none_id_connects_subject_only(self) -> None:
        # object_id=None (literal value) -> object edge skipped, subject linked.
        ents = [_entity("e1")]
        claims = [_claim("c1", "e1", None, object_name="2024-01-01")]
        g = GraphBuilder(ents, [], claims).build()
        assert g.has_edge("e1", "c1")
        assert g.degree("c1") == 1

    def test_disconnected_claim_still_added_as_node(self) -> None:
        # Neither subject nor object resolves -> claim node exists, no edges.
        claims = [_claim("c1", "MISSING_S", "MISSING_O")]
        g = GraphBuilder([], [], claims).build()
        assert "c1" in g.nodes
        assert g.degree("c1") == 0

    def test_no_claims_yields_no_claim_nodes(self) -> None:
        ents = [_entity("e1"), _entity("e2")]
        g = GraphBuilder(ents, [_rel("r1", "e1", "e2")]).build()
        assert all(d.get("node_type") != "claim" for _, d in g.nodes(data=True))

    def test_claims_default_to_empty_list(self) -> None:
        # claims=None constructor arg normalizes to [] and build runs cleanly.
        gb = GraphBuilder([_entity("e1")], [], claims=None)
        assert gb.claims == []
        g = gb.build()
        assert set(g.nodes) == {"e1"}


# --------------------------------------------------------------------------- #
# build() return + idempotency
# --------------------------------------------------------------------------- #
class TestBuild:
    def test_build_returns_same_graph_instance(self) -> None:
        gb = GraphBuilder([_entity("e1")], [])
        g = gb.build()
        assert g is gb.graph

    def test_build_is_idempotent_on_repeat(self) -> None:
        # Re-running build over the same simple graph reproduces the same shape.
        ents = [_entity("e1"), _entity("e2")]
        rels = [_rel("r1", "e1", "e2")]
        gb = GraphBuilder(ents, rels)
        first = gb.build()
        n, e = first.number_of_nodes(), first.number_of_edges()
        second = gb.build()
        assert (second.number_of_nodes(), second.number_of_edges()) == (n, e)

    def test_full_graph_node_and_edge_counts(self) -> None:
        ents = [_entity("e1"), _entity("e2"), _entity("e3")]
        rels = [_rel("r1", "e1", "e2"), _rel("r2", "e2", "e3")]
        claims = [_claim("c1", "e1", "e3")]
        g = GraphBuilder(ents, rels, claims).build()
        # 3 entity nodes + 1 claim node; 2 rel edges + 2 claim edges.
        assert g.number_of_nodes() == 4
        assert g.number_of_edges() == 4
