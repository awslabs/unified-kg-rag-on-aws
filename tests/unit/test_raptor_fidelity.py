# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Differential tests against the original RAPTOR code
(github.com/parthsarthi03/raptor, MIT; docs/design.md#65-raptor-summary-tree-raptor-tree-opt-in).

The original's outputs on synthetic points (``tests/fixtures/raptor_fidelity``
generates them on both sides) were recorded once in its own environment
(``tests/fixtures/raptor/README.md``); these tests compare ours with them,
without the original installed:

- tree assembly, exact: the original's clusterings replayed into our
  assembly, with our deviations set to its behaviour, build its tree;
- collapsed-tree selection, exact: ``collapsed_tree_reference`` selects the
  original's nodes in its order;
- clustering, by agreement: BIC picks the original's cluster count on at
  least 90% of the cases (within one on all), the soft memberships reach an
  adjusted Rand index of at least 0.9, and the global/local two-step finds
  the original's global count. Our EM is not scikit-learn's initialization,
  so agreement, not equality, is the bar.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.fixtures import raptor_fidelity as shared
from tests.fixtures.collapsed_tree import collapsed_tree_reference
from unified_kg_rag.domain.ingestion import summary_tree as st
from unified_kg_rag.domain.ingestion.summary_tree import (
    Cluster,
    Leaf,
    Realized,
    TreeSettings,
    build_tree,
    node_id,
)

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).parents[1] / "fixtures" / "raptor" / "upstream.json"
UPSTREAM = json.loads(FIXTURE.read_text(encoding="utf-8"))
# The original's settings our comparisons take.
SETTINGS = TreeSettings(
    algorithm="gmm",
    bic_grid="upstream",
    soft_threshold=0.1,
    reduction_dims=10,
    max_levels=5,
    min_nodes_to_cluster=11,
    min_level_shrink=0.0,
    max_cluster_tokens=3500,
    summarize_single=True,
)


def _points(case: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        shared.blobs(
            case["case"],
            case["n"],
            case["centers"],
            case["spread"],
            case.get("dims", 10),
        )
    )


def _adjusted_rand(a: list[int], b: list[int]) -> float:
    """The adjusted Rand index of two hard labelings (Hubert and Arabie)."""
    pairs: dict[tuple[int, int], int] = defaultdict(int)
    for x, y in zip(a, b, strict=True):
        pairs[(x, y)] += 1
    rows: dict[int, int] = defaultdict(int)
    cols: dict[int, int] = defaultdict(int)
    for (x, y), count in pairs.items():
        rows[x] += count
        cols[y] += count

    def choose2(v: int) -> float:
        return v * (v - 1) / 2

    index = sum(choose2(v) for v in pairs.values())
    row_sum = sum(choose2(v) for v in rows.values())
    col_sum = sum(choose2(v) for v in cols.values())
    expected = row_sum * col_sum / choose2(len(a))
    maximum = (row_sum + col_sum) / 2
    if maximum == expected:
        return 1.0
    return (index - expected) / (maximum - expected)


def _hard(labels: list[list[int]]) -> list[int]:
    """Each point's first (lowest) cluster: the hard view of soft labels."""
    return [min(row) if row else -1 for row in labels]


# Pull requests run a sample of the clustering cases (minutes of EM in all);
# the weekly fidelity workflow (.github/workflows/fidelity.yml) runs them all.
weekly = pytest.mark.weekly
# One case for each cluster count the original chose (1 to 5).
_SAMPLED_CLUSTERINGS = (
    "cluster-36",
    "cluster-13",
    "cluster-2",
    "cluster-12",
    "cluster-22",
)
_SAMPLED_TWO_STEPS = ("two-step-1", "two-step-5")


def _sampled(cases: list[dict[str, Any]], sample: tuple[str, ...]) -> list[Any]:
    return [
        pytest.param(
            case, id=case["case"], marks=() if case["case"] in sample else weekly
        )
        for case in cases
    ]


# --- clustering, by agreement -------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [c for c in UPSTREAM["clustering"] if c["case"] in _SAMPLED_CLUSTERINGS],
    ids=lambda c: c["case"],
)
def test_bic_picks_the_originals_cluster_count_on_a_sample(
    case: dict[str, Any],
) -> None:
    assert st.choose_k(_points(case), SETTINGS, expected=1.0) == case["k"]


@weekly
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_bic_picks_the_originals_cluster_count() -> None:
    cases = UPSTREAM["clustering"]
    equal = within_one = 0
    for case in cases:
        k = st.choose_k(_points(case), SETTINGS, expected=1.0)
        equal += k == case["k"]
        within_one += abs(k - case["k"]) <= 1
    assert equal / len(cases) >= 0.9, (equal, len(cases))
    assert within_one == len(cases)


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_soft_memberships_agree_with_the_original() -> None:
    scores = []
    for case in UPSTREAM["clustering"]:
        points = _points(case)
        mixture = st.fit_mixture(points, case["k"], SETTINGS.random_state)
        ours = st.soft_labels(mixture.posteriors, SETTINGS.soft_threshold)
        scores.append(_adjusted_rand(_hard(ours), _hard(case["labels"])))
    assert float(np.mean(scores)) >= 0.9, np.mean(scores)
    assert float(np.median(scores)) >= 0.95


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.parametrize("case", _sampled(UPSTREAM["two_step"], _SAMPLED_TWO_STEPS))
def test_the_two_step_finds_the_originals_global_clusters(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Both sides cluster the same points: the original's UMAP was an
    # identity truncation, so is our reduction here.
    monkeypatch.setattr(
        st, "reduce_dimensions", lambda vectors, dims: np.asarray(vectors)[:, :dims]
    )
    points = _points(case)
    truth = [i % case["centers"] for i in range(case["n"])]
    global_points = points[:, : min(10, case["n"] - 2)]
    k = st.choose_k(global_points, SETTINGS, expected=1.0)
    assert k == case["global_k"] == case["centers"]
    mixture = st.fit_mixture(global_points, k, SETTINGS.random_state)
    hard = _hard(st.soft_labels(mixture.posteriors, SETTINGS.soft_threshold))
    assert _adjusted_rand(hard, truth) >= 0.9
    # The local step refines the same global clusters on both sides: every
    # local cluster lies inside one blob. How finely it splits a blob is a
    # BIC overfit of full covariances on one Gaussian, which no two EM
    # implementations split alike (the original 6-25 clusters a blob, ours
    # 1-16 on these cases), so the counts are not compared.
    for members in st.gmm_clusters(points, SETTINGS):
        assert len({truth[i] for i in members}) == 1
    upstream: dict[int, set[int]] = defaultdict(set)
    for i, row in enumerate(case["labels"]):
        for label in row:
            upstream[label].add(truth[i])
    assert all(len(blobs) == 1 for blobs in upstream.values())


# --- tree assembly, exact -------------------------------------------------------


def _leaf_id(index: int) -> str:
    return f"leaf-{index:05d}"


class _Replay:
    """Our assembly driven by the original's recorded clusterings.

    The partition function looks up the call the original made on the same
    node set and returns its clusters (as RAPTOR_Clustering turns labels
    into clusters); ``realize`` makes each summary as the recording script
    did ("S" and the children's indices, the mean of their vectors) and
    maps our node ids to the original's indices."""

    def __init__(self, tree: dict[str, Any], vectors: np.ndarray) -> None:
        self.calls = {frozenset(c["nodes"]): c for c in tree["calls"]}
        self.index_of = {_leaf_id(i): i for i in range(len(vectors))}
        self.vectors = {i: vectors[i] for i in range(len(vectors))}
        self.next_index = len(vectors)

    def partition(
        self, vectors: np.ndarray, split: bool, keys: list[str]
    ) -> list[list[int]]:
        indices = [self.index_of[key] for key in keys]
        call = self.calls[frozenset(indices)]
        position = {node: i for i, node in enumerate(call["nodes"])}
        labels = call["labels"]
        clusters = []
        for label in sorted({v for row in labels for v in row}):
            members = [
                node for node in call["nodes"] if label in labels[position[node]]
            ]
            clusters.append(
                sorted(keys.index(_key) for _key in self._keys(members, keys))
            )
        return clusters

    def _keys(self, members: list[int], keys: list[str]) -> list[str]:
        wanted = set(members)
        return [key for key in keys if self.index_of[key] in wanted]

    def realize(self, clusters: list[Cluster]) -> list[Realized | None]:
        made = []
        for cluster in clusters:
            children = sorted(self.index_of[c] for c in cluster.child_ids)
            index = self.next_index
            self.next_index += 1
            vector = np.mean([self.vectors[c] for c in children], axis=0)
            self.vectors[index] = vector
            self.index_of[node_id(cluster.level, None, cluster.child_ids)] = index
            text = "S " + " ".join(str(c) for c in children)
            made.append(
                Realized(
                    text=text,
                    vector=vector.tolist(),
                    n_tokens=len(text.split()),
                    input_key=text,
                )
            )
        return made


@pytest.mark.parametrize(
    "tree", UPSTREAM["trees"], ids=[t["case"] for t in UPSTREAM["trees"]]
)
def test_tree_assembly_matches_the_original(tree: dict[str, Any]) -> None:
    vectors = _points(tree)
    words = [
        shared.integer(150, 700, tree["case"], "words", i) for i in range(tree["n"])
    ]
    leaves = [
        Leaf(
            id=_leaf_id(i),
            text=" ".join(["w"] * words[i]),
            vector=vectors[i].tolist(),
            n_tokens=words[i],
        )
        for i in range(tree["n"])
    ]
    replay = _Replay(tree, vectors)
    build = build_tree(
        leaves,
        settings=SETTINGS,
        realize=replay.realize,
        language="English",
        partition=replay.partition,
    )
    ours: dict[int, list[list[int]]] = defaultdict(list)
    for node in build.nodes:
        ours[node.level].append(sorted(replay.index_of[c] for c in node.child_ids))
    expected = {int(level): children for level, children in tree["layers"].items()}
    assert {level: sorted(c) for level, c in ours.items()} == expected
    assert build.levels == tree["num_layers"]


# --- collapsed-tree selection, exact ---------------------------------------------


@pytest.mark.parametrize(
    "case", UPSTREAM["selection"], ids=[c["case"] for c in UPSTREAM["selection"]]
)
def test_collapsed_tree_selection_matches_the_original(case: dict[str, Any]) -> None:
    vectors = shared.blobs(case["case"], case["n"], 1, 1.0, dims=16)
    words = [
        shared.integer(50, 900, case["case"], "words", i) for i in range(case["n"])
    ]
    query = shared.blobs(case["case"] + "-query", 1, 1, 1.0, dims=16)[0]
    selected = collapsed_tree_reference(
        query,
        [(str(i), vectors[i], words[i]) for i in range(case["n"])],
        top_k=10,
        max_tokens=3500,
    )
    assert [int(s) for s in selected] == case["selected"]


# --- the original itself (optional) ---------------------------------------------


def _upstream_cluster_utils() -> Any:
    """The original's cluster_utils with its model clients stubbed, as the
    recording script loads it; skipped unless ``RAPTOR_UPSTREAM`` names a
    checkout and scikit-learn is installed (neither is a dependency)."""
    import importlib
    import os
    import sys
    import types

    root = os.environ.get("RAPTOR_UPSTREAM")
    if not root or not (Path(root) / "raptor" / "cluster_utils.py").is_file():
        pytest.skip("set RAPTOR_UPSTREAM to a raptor checkout to compare live")
    pytest.importorskip("sklearn")

    class _UMAP:
        def __init__(self, n_components: int = 2, **_: Any) -> None:
            self.n_components = n_components

        def fit_transform(self, x: Any) -> Any:
            return np.asarray(x)[:, : self.n_components]

    class _Words:
        def encode(self, text: str) -> list[str]:
            return text.split()

    stubs = {
        "umap": {"UMAP": _UMAP},
        "tiktoken": {"get_encoding": lambda name: _Words()},
        "openai": {"OpenAI": object},
        "sentence_transformers": {"SentenceTransformer": object},
    }
    saved = {name: sys.modules.get(name) for name in [*stubs, "raptor"]}
    try:
        for name, attrs in stubs.items():
            module = types.ModuleType(name)
            for key, value in attrs.items():
                setattr(module, key, value)
            sys.modules[name] = module
        package = types.ModuleType("raptor")
        package.__path__ = [str(Path(root) / "raptor")]
        sys.modules["raptor"] = package
        return importlib.import_module("raptor.cluster_utils")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@weekly
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_the_recorded_answers_are_the_originals() -> None:
    upstream = _upstream_cluster_utils()
    for case in UPSTREAM["clustering"][:5]:
        labels, k = upstream.GMM_cluster(_points(case), threshold=0.1)
        assert int(k) == case["k"]
        assert [[int(v) for v in row] for row in labels] == case["labels"]
