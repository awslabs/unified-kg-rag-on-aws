# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The summary tree's pure clustering and assembly
(``domain.ingestion.summary_tree``)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests.fixtures.collapsed_tree import collapsed_tree_reference
from unified_kg_rag.domain.ingestion.summary_tree import (
    Cluster,
    Leaf,
    Realized,
    TreeSettings,
    bic_candidates,
    build_tree,
    choose_k,
    fit_mixture,
    gmm_clusters,
    kmeans,
    kmeans_clusters,
    leaf_digest,
    node_id,
    reduce_dimensions,
    soft_labels,
    summary_input_key,
    uniform_filters,
)
from unified_kg_rag.domain.models.config import SummaryTreeConfig

pytestmark = pytest.mark.unit


def _blobs(n: int, *, dims: int = 32, centers: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    middles = rng.normal(size=(centers, dims)) * 4
    return np.stack(
        [middles[i % centers] + rng.normal(size=dims) * 0.2 for i in range(n)]
    )


def _leaves(
    n: int, *, tokens: int = 100, filters: list[dict[str, str]] | None = None
) -> list[Leaf]:
    vectors = _blobs(n)
    return [
        Leaf(
            id=f"unit-{i:04d}",
            text=f"passage {i}",
            vector=vectors[i].tolist(),
            n_tokens=tokens,
            filters=(filters[i] if filters else {}),
        )
        for i in range(n)
    ]


class _Summaries:
    """A realize callback: the mean of the children's vectors, and a record
    of every cluster it was given."""

    def __init__(self, fail: set[str] | None = None, fail_level: int = 1) -> None:
        self.calls: list[list[Cluster]] = []
        self.fail = fail or set()
        self.fail_level = fail_level

    def __call__(self, clusters: list[Cluster]) -> list[Realized | None]:
        self.calls.append(clusters)
        made: list[Realized | None] = []
        for cluster in clusters:
            if cluster.level == self.fail_level and self.fail & set(cluster.child_ids):
                made.append(None)
                continue
            vector = np.mean([m.vector for m in cluster.members], axis=0)
            made.append(
                Realized(
                    text="summary of " + ",".join(cluster.child_ids),
                    vector=vector.tolist(),
                    n_tokens=40,
                    input_key=summary_input_key("fp", "English", cluster.child_texts),
                )
            )
        return made


def test_settings_follow_the_config() -> None:
    config = SummaryTreeConfig(algorithm="gmm", target_children=4, random_state=7)
    settings = TreeSettings.from_config(config)
    assert (settings.algorithm, settings.target_children, settings.random_state) == (
        "gmm",
        4,
        7,
    )
    assert not settings.summarize_single
    # One source: the config's fields and defaults.
    assert TreeSettings() == TreeSettings.from_config(SummaryTreeConfig())
    names = {f.name for f in dataclasses.fields(TreeSettings)} - {"summarize_single"}
    assert names <= set(SummaryTreeConfig.model_fields)


def test_reduction_is_sign_fixed_rounded_and_scale_free() -> None:
    vectors = _blobs(40, dims=64)
    reduced = reduce_dimensions(vectors, 10)
    assert reduced.shape == (40, 10)
    # Rounded to 1e-6, and the same for vectors of another length (they
    # are L2-normalized first) or a flipped sign (each axis's largest
    # loading is made positive).
    assert np.array_equal(reduced, np.round(reduced, 6))
    assert np.allclose(reduce_dimensions(vectors * 3.0, 10), reduced, atol=1e-5)
    # Fewer points than dimensions: n - 1 axes.
    assert reduce_dimensions(vectors[:5], 10).shape == (5, 4)


def test_kmeans_is_seeded_and_finds_separated_blobs() -> None:
    points = reduce_dimensions(_blobs(80, centers=4), 10)
    labels = kmeans(points, 4, seed=1)
    assert np.array_equal(labels, kmeans(points, 4, seed=1))
    # Every blob in one cluster.
    for blob in range(4):
        assert len(set(labels[blob::4])) == 1
    assert len(set(labels)) == 4


def test_kmeans_splits_a_large_problem_in_two_steps() -> None:
    # n * k over the direct limit: coarse then fine, every point labelled
    # and about k clusters.
    points = reduce_dimensions(_blobs(9000, dims=16, centers=30), 10)
    labels = kmeans(points, 1500, seed=3)
    assert len(labels) == 9000
    # Each coarse cluster takes its share of k, rounded.
    assert 1350 <= len(set(labels)) <= 1550


def test_mixture_bic_prefers_the_true_count() -> None:
    points = reduce_dimensions(_blobs(120, centers=3), 4)
    settings = TreeSettings(bic_grid="upstream")
    assert choose_k(points, settings, 3) == 3
    mixture = fit_mixture(points, 3, seed=0)
    assert mixture.posteriors.shape == (120, 3)
    assert np.allclose(mixture.posteriors.sum(axis=1), 1.0)


def test_bic_grids() -> None:
    upstream = TreeSettings(bic_grid="upstream")
    assert bic_candidates(100, upstream, 5) == list(range(1, 50))
    assert bic_candidates(7, upstream, 5) == list(range(1, 7))
    bounded = bic_candidates(1000, TreeSettings(), 10)
    assert len(bounded) <= 8 and 10 in bounded
    assert min(bounded) == 5 and max(bounded) == 20
    assert all(k >= 2 for k in bic_candidates(1000, TreeSettings(), 1, split=True))


def test_soft_labels_keep_every_point() -> None:
    posteriors = np.array([[0.5, 0.45, 0.05], [0.05, 0.05, 0.9], [0.34, 0.33, 0.33]])
    assert soft_labels(posteriors, 0.1) == [[0, 1], [2], [0, 1, 2]]
    # None above the threshold: the likeliest one (RAPTOR drops the point).
    assert soft_labels(np.array([[0.3, 0.3, 0.4]]), 0.5) == [[2]]


def test_gmm_two_step_clusters_cover_every_point() -> None:
    vectors = _blobs(90, centers=3)
    clusters = gmm_clusters(vectors, TreeSettings(algorithm="gmm"))
    assert set().union(*map(set, clusters)) == set(range(90))


def test_kmeans_clusters_aim_at_the_target_size() -> None:
    clusters = kmeans_clusters(_blobs(60), TreeSettings(target_children=6))
    assert len(clusters) == 10
    assert sorted(i for c in clusters for i in c) == list(range(60))


def test_identity_and_digests_are_order_free() -> None:
    assert node_id(1, None, ["b", "a"]) == node_id(1, None, ["a", "b"])
    assert node_id(1, None, ["a"]) != node_id(2, None, ["a"])
    assert node_id(1, '["x"]', ["a"]) != node_id(1, None, ["a"])
    a, b = ("a", "text a", {}), ("b", "text b", {"category": "x"})
    assert leaf_digest([b, a]) == leaf_digest([a, b])
    # A leaf's text or filters changing under its id is another leaf set.
    assert leaf_digest([a, ("b", "text b2", {"category": "x"})]) != leaf_digest([a, b])
    assert leaf_digest([a, ("b", "text b", {"category": "y"})]) != leaf_digest([a, b])
    assert leaf_digest([a, ("b", "text b", {})]) != leaf_digest([a, b])
    assert summary_input_key("fp", "Korean", ["가", "나"]) != summary_input_key(
        "fp", "English", ["가", "나"]
    )


def test_uniform_filters_keep_only_shared_values() -> None:
    maps = [
        {"category": "logistics", "region": "eu", "tags": ["a", "b"]},
        {"category": "logistics", "region": "us", "tags": ["a", "b"]},
    ]
    assert uniform_filters(maps) == {"category": "logistics", "tags": ["a", "b"]}
    assert uniform_filters([{"category": "x"}, {}]) == {}
    assert uniform_filters([]) == {}


def test_build_tree_levels_lineage_and_stop() -> None:
    summaries = _Summaries()
    build = build_tree(
        _leaves(120),
        settings=TreeSettings(),
        realize=summaries,
        language="English",
        attributes={"index": "corpus"},
    )
    assert build.levels >= 2 and build.failed == 0
    by_id = {node.id: node for node in build.nodes}
    leaf_ids = {f"unit-{i:04d}" for i in range(120)}
    for node in build.nodes:
        # Lineage is the union of the children's.
        expected: set[str] = set()
        for child in node.child_ids:
            expected |= set(by_id[child].text_unit_ids) if child in by_id else {child}
        assert set(node.text_unit_ids) == expected <= leaf_ids
        assert node.n_leaves == len(expected) and node.lineage_complete
        assert node.attributes == {"index": "corpus"}
        assert node.id == node_id(node.level, None, node.child_ids)
        assert len(node.child_ids) >= 2
    # The top level is small enough not to be clustered again.
    top = [n for n in build.nodes if n.level == build.levels]
    assert len(top) <= 11 or build.stop_reason in {"max_levels", "no_shrink"}
    # Every vector came from the callback.
    assert set(build.vectors) == set(by_id)


def test_tree_is_independent_of_input_order() -> None:
    leaves = _leaves(150)
    first = build_tree(
        leaves, settings=TreeSettings(), realize=_Summaries(), language="English"
    )
    second = build_tree(
        leaves[::-1], settings=TreeSettings(), realize=_Summaries(), language="English"
    )
    assert [n.model_dump() for n in first.nodes] == [
        n.model_dump() for n in second.nodes
    ]


def test_small_corpus_builds_no_node() -> None:
    build = build_tree(
        _leaves(8), settings=TreeSettings(), realize=_Summaries(), language="English"
    )
    assert build.nodes == [] and build.stop_reason == "few_nodes"


def test_failed_summary_promotes_its_children() -> None:
    leaves = _leaves(60)
    failing = _Summaries(fail={"unit-0000"})
    build = build_tree(leaves, settings=TreeSettings(), realize=failing, language="en")
    assert build.failed >= 1
    # unit-0000 is still reachable: some node above it cites it.
    first_level = failing.calls[0]
    assert any("unit-0000" in c.child_ids for c in first_level)
    # Promoted: clustered again at level 2 with the summaries, and covered
    # by a level-2 node.
    second_level_ids = {m.id for c in failing.calls[1] for m in c.members}
    assert "unit-0000" in second_level_ids
    assert any("unit-0000" in n.child_ids and n.level == 2 for n in build.nodes)


def test_clusters_respect_the_token_bound() -> None:
    summaries = _Summaries()
    build_tree(
        _leaves(60, tokens=900),
        settings=TreeSettings(max_cluster_tokens=2000, target_children=10),
        realize=summaries,
        language="en",
    )
    for cluster in summaries.calls[0]:
        assert sum(m.n_tokens for m in cluster.members) <= 2000


@pytest.mark.parametrize(
    "unsplit",
    [
        lambda n: [list(range(n))],
        # Soft clusters, one of which holds every member.
        lambda n: [list(range(n)), [0, 1]],
    ],
    ids=["one-cluster", "a-cluster-of-all"],
)
def test_a_cluster_too_large_to_split_is_packed_by_tokens_in_id_order(
    unsplit: object,
) -> None:
    def partition(vectors: np.ndarray, split: bool, keys: object) -> list[list[int]]:
        # The level is one cluster; asked to split it, clustering cannot.
        if not split:
            return [list(range(len(vectors)))]
        return unsplit(len(vectors))  # type: ignore[operator]

    summaries = _Summaries()
    leaves = _leaves(20, tokens=100)
    build_tree(
        list(reversed(leaves)),
        settings=TreeSettings(max_cluster_tokens=350, max_levels=1),
        realize=summaries,
        language="en",
        partition=partition,
    )
    ids = [leaf.id for leaf in leaves]
    # Three leaves fit in 350 tokens, a fourth does not.
    assert [c.child_ids for c in summaries.calls[0]] == [
        ids[i : i + 3] for i in range(0, 20, 3)
    ]


def test_single_member_clusters_are_promoted_or_summarized() -> None:
    def single(vectors: np.ndarray, split: bool, keys: object) -> list[list[int]]:
        return [[i] for i in range(len(vectors))]

    promoted = build_tree(
        _leaves(20),
        settings=TreeSettings(min_level_shrink=0.0),
        realize=_Summaries(),
        language="en",
        partition=single,
    )
    assert promoted.nodes == [] and promoted.stop_reason == "no_summary"
    upstream = build_tree(
        _leaves(20),
        settings=TreeSettings(
            min_level_shrink=0.0, summarize_single=True, max_levels=1
        ),
        realize=_Summaries(),
        language="en",
        partition=single,
    )
    assert len(upstream.nodes) == 20


def test_no_shrink_stops_before_any_call() -> None:
    def pairs_of_one(vectors: np.ndarray, split: bool, keys: object) -> list[list[int]]:
        n = len(vectors)
        return [[0, 1], *([i] for i in range(2, n))]

    summaries = _Summaries()
    build = build_tree(
        _leaves(30),
        settings=TreeSettings(),
        realize=summaries,
        language="en",
        partition=pairs_of_one,
    )
    assert build.stop_reason == "no_shrink" and summaries.calls == []


def test_lineage_cap_marks_incomplete() -> None:
    build = build_tree(
        _leaves(400),
        settings=TreeSettings(max_lineage_text_units=20, target_children=50),
        realize=_Summaries(),
        language="en",
    )
    capped = [n for n in build.nodes if not n.lineage_complete]
    assert capped
    for node in capped:
        assert len(node.text_unit_ids) == 20 < node.n_leaves
        assert node.text_unit_ids == sorted(node.text_unit_ids)


def test_nodes_keep_only_uniform_filters() -> None:
    filters = [{"category": "a" if i < 60 else "b", "shared": "x"} for i in range(120)]
    build = build_tree(
        _leaves(120, filters=filters),
        settings=TreeSettings(),
        realize=_Summaries(),
        language="en",
    )
    for node in build.nodes:
        node_filters = (node.attributes or {}).get("filters") or {}
        categories = {
            "a" if int(unit.split("-")[1]) < 60 else "b" for unit in node.text_unit_ids
        }
        assert node_filters.get("shared") == "x"
        if len(categories) == 1:
            assert node_filters["category"] == categories.pop()
        else:
            assert "category" not in node_filters


def test_collapsed_tree_reference_takes_top_k_until_the_budget_breaks() -> None:
    candidates = [
        ("a", [1.0, 0.0], 1000),
        ("b", [0.9, 0.1], 2000),
        ("c", [0.8, 0.2], 1000),
        ("d", [0.0, 1.0], 10),
    ]
    assert collapsed_tree_reference([1.0, 0.0], candidates, top_k=10) == ["a", "b"]
    assert collapsed_tree_reference([1.0, 0.0], candidates, top_k=1) == ["a"]
    assert collapsed_tree_reference([0.0, 1.0], candidates, max_tokens=5) == []
