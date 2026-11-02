# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invariants of converting a seed graph into shard documents.

Over random seeds (declared, respelled and undeclared types, colliding
names, duplicate edges and self-loops): the documents do not depend on the
order of the nodes and edges, a node added without edges changes only its
own shard, and strict keeps a subset of the statements align keeps.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from unified_kg_rag.domain.ingestion.seed_graph import (
    SeedConversion,
    SeedConversionSettings,
    convert_seed,
    seed_statements,
    shard_of,
)
from unified_kg_rag.domain.models import ExtractionSchema
from unified_kg_rag.domain.models.seed_graph import SeedEdge, SeedGraph, SeedNode

pytestmark = pytest.mark.property

_NAMES = ["Vendor", "vendor", "Buyer", "Depot", "Widget", "Ghost", "Acme Co"]
_ENTITY_TYPES = ["ORGANIZATION", "organization", "PRODUCT", "ALIEN", None]
_RELATIONSHIP_TYPES = ["SUPPLIES", "supplies", "BUYS", "KNOWS"]


@cache
def _schema(mode: str) -> ExtractionSchema:
    return ExtractionSchema.from_items(
        ["ORGANIZATION", "PRODUCT"],
        ["SUPPLIES", "BUYS"],
        ["ORGANIZATION -SUPPLIES-> ORGANIZATION", "ORGANIZATION -BUYS-> PRODUCT"],
        mode=mode,  # type: ignore[arg-type]
    )


def _settings(**overrides: Any) -> SeedConversionSettings:
    values: dict[str, Any] = {
        "name": "procure",
        "shards": 4,
        "max_items_per_unit": 3,
        "on_name_collision": "merge",
    }
    values.update(overrides)
    return SeedConversionSettings(**values)


@st.composite
def _seeds(draw: st.DrawFn) -> SeedGraph:
    count = draw(st.integers(min_value=2, max_value=6))
    nodes = [
        SeedNode(
            id=f"n{i}",
            name=draw(st.sampled_from(_NAMES))
            + ("" if draw(st.booleans()) else f" {i}"),
            type=draw(st.sampled_from(_ENTITY_TYPES)),
            description=draw(st.sampled_from([None, "first", "second"])),
            properties=draw(
                st.dictionaries(
                    st.sampled_from(["a", "b"]),
                    st.one_of(st.integers(0, 3), st.sampled_from(["x", "y"])),
                    max_size=2,
                )
            ),
        )
        for i in range(count)
    ]
    ids = [node.id for node in nodes]
    edges = draw(
        st.lists(
            st.builds(
                SeedEdge,
                source=st.sampled_from(ids),
                target=st.sampled_from(ids),
                type=st.sampled_from(_RELATIONSHIP_TYPES),
                description=st.sampled_from([None, "one", "two"]),
                weight=st.sampled_from([0.5, 1.0, 2.0]),
            ),
            max_size=12,
        )
    )
    # Restatements of drawn edges, which the conversion merges.
    if edges:
        edges += [
            edges[position].model_copy(
                update={"description": description, "properties": {"a": value}}
            )
            for position, description, value in draw(
                st.lists(
                    st.tuples(
                        st.integers(0, len(edges) - 1),
                        st.sampled_from(["three", "four", None]),
                        st.integers(0, 3),
                    ),
                    max_size=4,
                )
            )
        ]
    return SeedGraph(nodes=tuple(nodes), edges=tuple(edges))


def _texts(conversion: SeedConversion) -> dict[str, str]:
    return {s.relative_path: s.page_content for s in conversion.documents_content}


@given(
    seed=_seeds(),
    mode=st.sampled_from(["off", "align", "strict"]),
    data=st.data(),
)
def test_conversion_does_not_depend_on_statement_order(
    seed: SeedGraph, mode: str, data: st.DataObject
) -> None:
    shuffled = SeedGraph(
        nodes=tuple(data.draw(st.permutations(seed.nodes))),
        edges=tuple(data.draw(st.permutations(seed.edges))),
    )

    first = convert_seed(seed, _settings(), _schema(mode))
    second = convert_seed(shuffled, _settings(), _schema(mode))

    assert _texts(first) == _texts(second)
    assert first.digest == second.digest
    assert first.stats == second.stats


@given(seed=_seeds(), mode=st.sampled_from(["align", "strict"]))
def test_a_node_without_edges_changes_only_its_own_shard(
    seed: SeedGraph, mode: str
) -> None:
    added = SeedNode(id="added", name="Brand New Node", type="PRODUCT")
    grown = SeedGraph(nodes=(*seed.nodes, added), edges=seed.edges)

    before = _texts(convert_seed(seed, _settings(), _schema(mode)))
    after = _texts(convert_seed(grown, _settings(), _schema(mode)))

    home = f"seed_graphs/procure/{shard_of('added', 4):04d}-of-0004.jsonl"
    assert {path for path in after if before.get(path) != after[path]} == {home}
    assert set(before) <= set(after)


@given(seed=_seeds())
def test_strict_keeps_a_subset_of_what_align_keeps(seed: SeedGraph) -> None:
    align = convert_seed(seed, _settings(), _schema("align"))
    strict = convert_seed(seed, _settings(), _schema("strict"))

    assert set(seed_statements(strict)) <= set(seed_statements(align))
    assert strict.stats.nodes <= align.stats.nodes
    assert strict.stats.edges <= align.stats.edges
    assert align.stats.entities_dropped_off_schema == 0
