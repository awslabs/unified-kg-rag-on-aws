# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property tests of the summary tree's assembly
(``domain.ingestion.summary_tree.build_tree``).

For any corpus of leaves and either algorithm: the tree does not depend on
the order the leaves come in, every node's lineage is the union of its
children's, every leaf is covered by the top of the tree or reaches it
promoted, no cluster a summary covers holds more tokens than the bound, and
a node keeps a document filter only when all its leaves share its value.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from unified_kg_rag.domain.ingestion.summary_tree import (
    Cluster,
    Leaf,
    Realized,
    TreeSettings,
    build_tree,
    summary_input_key,
)

pytestmark = pytest.mark.property


@st.composite
def _corpora(draw: st.DrawFn) -> tuple[list[Leaf], TreeSettings]:
    n = draw(st.integers(min_value=0, max_value=70))
    seed = draw(st.integers(min_value=0, max_value=10_000))
    centers = draw(st.integers(min_value=1, max_value=6))
    rng = np.random.default_rng(seed)
    middles = rng.normal(size=(centers, 12)) * 3
    categories = draw(st.lists(st.sampled_from(["a", "b"]), min_size=n, max_size=n))
    tokens = draw(
        st.lists(st.integers(min_value=1, max_value=400), min_size=n, max_size=n)
    )
    leaves = [
        Leaf(
            id=f"u{seed}-{i:03d}",
            text=f"passage {i}",
            vector=(middles[i % centers] + rng.normal(size=12) * 0.3).tolist(),
            n_tokens=tokens[i],
            filters={"category": categories[i], "corpus": "one"},
        )
        for i in range(n)
    ]
    tree = TreeSettings(
        algorithm=draw(st.sampled_from(["kmeans", "gmm"])),
        target_children=draw(st.integers(min_value=2, max_value=8)),
        min_nodes_to_cluster=draw(st.integers(min_value=2, max_value=11)),
        max_cluster_tokens=draw(st.integers(min_value=400, max_value=3000)),
        random_state=seed,
    )
    return leaves, tree


class _Mean:
    def __init__(self) -> None:
        self.clusters: list[Cluster] = []

    def __call__(self, clusters: list[Cluster]) -> list[Realized | None]:
        self.clusters.extend(clusters)
        return [
            Realized(
                text="summary",
                vector=np.mean([m.vector for m in c.members], axis=0).tolist(),
                n_tokens=30,
                input_key=summary_input_key("fp", "English", c.child_texts),
            )
            for c in clusters
        ]


@given(_corpora(), st.randoms(use_true_random=False))
@settings(max_examples=40)
def test_tree_is_a_function_of_the_leaf_set(corpus, shuffler) -> None:
    leaves, tree = corpus
    shuffled = list(leaves)
    shuffler.shuffle(shuffled)
    first = build_tree(leaves, settings=tree, realize=_Mean(), language="English")
    second = build_tree(shuffled, settings=tree, realize=_Mean(), language="English")
    assert [n.model_dump() for n in first.nodes] == [
        n.model_dump() for n in second.nodes
    ]


@given(_corpora())
@settings(max_examples=40)
def test_lineage_cover_tokens_and_filters(corpus) -> None:
    leaves, tree = corpus
    realize = _Mean()
    build = build_tree(leaves, settings=tree, realize=realize, language="English")
    by_id = {node.id: node for node in build.nodes}
    leaf_by_id = {leaf.id: leaf for leaf in leaves}
    for node in build.nodes:
        lineage: set[str] = set()
        for child in node.child_ids:
            lineage |= set(by_id[child].text_unit_ids) if child in by_id else {child}
        assert set(node.text_unit_ids) == lineage
        assert len(node.child_ids) >= 2
        categories = {leaf_by_id[u].filters["category"] for u in lineage}
        node_filters = (node.attributes or {}).get("filters") or {}
        assert node_filters.get("corpus") == "one"
        assert ("category" in node_filters) == (len(categories) == 1)
    for cluster in realize.clusters:
        if len(cluster.members) > 1:
            assert sum(m.n_tokens for m in cluster.members) <= tree.max_cluster_tokens
    # No leaf is lost: each is a child of some node, or no node was built
    # over it at level 1 (a cluster of one moves up as it is).
    covered = {u for node in build.nodes for u in node.text_unit_ids}
    first_level = [c for c in realize.clusters if c.level == 1]
    in_first = {m.id for c in first_level for m in c.members}
    assert in_first <= covered
