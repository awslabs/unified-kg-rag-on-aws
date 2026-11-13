# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""RAPTOR's summary tree: clustering and tree assembly, pure numpy.

Reimplements the clustering and the tree construction of RAPTOR (Sarthi et
al., ICLR 2024; reference code github.com/parthsarthi03/raptor, MIT), with
the deviations docs/design.md#65-raptor-summary-tree-raptor-tree-opt-in lists: PCA in place of UMAP, k-means by
default (RAPTOR's Gaussian mixture with BIC as ``algorithm: gmm``), a
cluster of one node promoted instead of summarized, and a stop when a level
no longer shrinks.

Everything is deterministic: inputs are sorted by id, every random choice
draws from a generator seeded by ``random_state``, and reduced coordinates
are rounded so a different BLAS thread count does not flip a tie. The model
calls are not here: :func:`build_tree` hands each level's clusters to a
``realize`` callback (the summarizer and the embedding model), so the
assembly is a pure function of the leaves, the settings and what the
callback returns.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from unified_kg_rag.domain.models import Constants, SummaryNode, TextUnit
from unified_kg_rag.domain.models.config import SummaryTreeConfig
from unified_kg_rag.domain.retrieval.filters import TRANSLATED_TEXT_PREFIX
from unified_kg_rag.shared.utils import compute_hash

# Covariance regularization of the Gaussian mixtures (scikit-learn's
# reg_covar, which RAPTOR's GaussianMixture uses).
_REG_COVAR = 1e-6
# EM and Lloyd iterations at most, and EM's convergence tolerance on the
# mean log-likelihood (scikit-learn's max_iter and tol).
_MAX_ITER = 100
_EM_TOL = 1e-3
# Reduced coordinates are rounded to this many decimals.
_ROUND = 6
# Rows reduced at a time (reduce_dimensions), and the largest n * k one
# k-means solves directly (kmeans): its distance matrix is n x k.
_BLOCK_ROWS = 8192
_DIRECT_KMEANS = 4_000_000
# Values the bounded BIC grid compares at most.
_BOUNDED_GRID = 8
# RAPTOR's max_clusters: BIC compares 1 .. min(50, n) - 1 clusters.
_UPSTREAM_MAX_CLUSTERS = 50


def _default(name: str) -> Any:
    return SummaryTreeConfig.model_fields[name].default


@dataclass(frozen=True)
class TreeSettings:
    """The clustering and assembly settings of one tree build: the fields of
    ``indexing.summary_tree`` (``SummaryTreeConfig``) the build reads, with
    its defaults. A plain record, not validated, so the fidelity tests can
    set what the config refuses."""

    algorithm: str = _default("algorithm")
    target_children: int = _default("target_children")
    soft_threshold: float = _default("soft_threshold")
    reduction_dims: int = _default("reduction_dims")
    max_levels: int = _default("max_levels")
    min_nodes_to_cluster: int = _default("min_nodes_to_cluster")
    min_level_shrink: float = _default("min_level_shrink")
    max_cluster_tokens: int = _default("max_cluster_tokens")
    max_lineage_text_units: int = _default("max_lineage_text_units")
    random_state: int = _default("random_state")
    # RAPTOR summarizes a cluster of one node; this promotes it instead.
    # True only to compare with the original (tests/fidelity).
    summarize_single: bool = False
    # gmm only: the cluster counts BIC compares. "bounded": at most
    # _BOUNDED_GRID values around n / target_children; "upstream": 1 to 49,
    # as RAPTOR's code (slow on large levels), only to compare with it.
    bic_grid: str = "bounded"

    @classmethod
    def from_config(cls, config: SummaryTreeConfig) -> TreeSettings:
        return cls(
            **{
                f.name: getattr(config, f.name)
                for f in fields(cls)
                if f.name in SummaryTreeConfig.model_fields
            }
        )


@dataclass(frozen=True)
class Leaf:
    """A stored text unit as the tree reads it."""

    id: str
    text: str
    # A float32 array as the summary_tree stage reads it (a list reads too).
    vector: Sequence[float] | np.ndarray
    n_tokens: int
    # Its document's filters (``attributes.filters``).
    filters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Entry:
    """A node of the level being clustered: a leaf (level 0), a summary, or
    a node promoted from a lower level."""

    id: str
    level: int
    text: str
    vector: np.ndarray
    n_tokens: int
    leaves: frozenset[str]
    filters: Mapping[str, Any]


@dataclass(frozen=True)
class Cluster:
    """Entries one summary will cover, sorted by id."""

    level: int
    members: tuple[Entry, ...]

    @property
    def child_ids(self) -> list[str]:
        return [member.id for member in self.members]

    @property
    def child_texts(self) -> list[str]:
        return [member.text for member in self.members]


@dataclass(frozen=True)
class Realized:
    """What the ``realize`` callback made of a cluster: its summary, the
    summary's vector and token count, and its ``input_key``."""

    text: str
    vector: Sequence[float]
    n_tokens: int
    input_key: str


# Clusters of the rows of the points (n x d), as row-index lists (a row may
# be in several: soft membership). ``split``: the caller needs at least two
# clusters (a cluster over the token bound is clustered again).
Partition = Callable[[np.ndarray, bool, Sequence[str]], list[list[int]]]
Realize = Callable[[list[Cluster]], list[Realized | None]]


@dataclass
class TreeBuild:
    """The nodes of a built tree and how the build went."""

    nodes: list[SummaryNode]
    # Each node's vector (float32, the array its entry clusters by).
    vectors: dict[str, np.ndarray]
    levels: int
    failed: int
    stop_reason: str


# --- identity and digests ---------------------------------------------------


def node_id(level: int, partition: str | None, child_ids: Sequence[str]) -> str:
    """The content id of a summary node: its level, partition and children."""
    payload = json.dumps(
        ["summary_node", level, partition or "", sorted(child_ids)],
        ensure_ascii=False,
    )
    return compute_hash(payload, length=32)


def leaf_entry(
    leaf_id: str, text: str, filters: Mapping[str, Any]
) -> tuple[str, str, str]:
    """What :func:`leaf_entries_digest` keeps of one leaf: its id, its
    text's hash and its filters, so a caller streaming the text units holds
    no text."""
    return (
        leaf_id,
        compute_hash(text),
        json.dumps(filters, sort_keys=True, default=str),
    )


def stored_leaf_text(row: Mapping[str, Any], translated_field: str) -> str:
    """The text a stored text unit (``row``: its stored fields) is a leaf of
    the tree with: its translation into the index language
    (``translated_field``), when it holds one, else its text."""
    return str(row.get(translated_field) or row.get("text") or "")


def stored_leaf_filters(row: Mapping[str, Any]) -> dict[str, Any]:
    """A stored text unit's document filters."""
    filters = (row.get("attributes") or {}).get(Constants.FILTERS.value) or {}
    return filters if isinstance(filters, dict) else {}


def stored_leaf_entry(
    row: Mapping[str, Any], translated_field: str
) -> tuple[str, str, str]:
    """:func:`leaf_entry` of a stored text unit (``row``: its ``id`` and
    stored fields)."""
    return leaf_entry(
        str(row["id"]),
        stored_leaf_text(row, translated_field),
        stored_leaf_filters(row),
    )


def translated_leaf_field(language: str) -> str:
    """The stored text-unit field a leaf's translation into ``language``
    (the index language) is read from."""
    return f"{TRANSLATED_TEXT_PREFIX}{language}"


def unit_leaf_entry(
    unit: TextUnit, language: str, translated_languages: Collection[str]
) -> tuple[str, str, str]:
    """:func:`stored_leaf_entry` of ``unit`` once written: a vector store
    keeps its translations into the ``translated_languages`` only, and the
    tree reads the one into ``language`` (the index language)."""
    translated = (
        (unit.translated_texts or {}).get(language)
        if language in translated_languages
        else None
    )
    field = translated_leaf_field(language)
    return stored_leaf_entry(
        {
            "id": unit.id,
            "text": unit.text or "",
            **({field: translated} if translated else {}),
            "attributes": unit.attributes or {},
        },
        field,
    )


def leaf_entries_digest(entries: Iterable[tuple[str, str, str]]) -> str:
    """Digest of a tree's leaves (:func:`leaf_entry` of each, in any order):
    each text unit's id, the text the tree reads (its translation, when it
    holds one) and its document filters.

    A leaf whose text or filters changed under the same id (a reset after
    a re-translation, or after a document's filters changed) changes the
    digest, so the tree is built again: its summaries were made from the
    old text and its nodes carry the old uniform filters. The leaf vectors
    are a function of the text and the embedding model, which the tree's
    fingerprint holds.
    """
    payload = json.dumps(sorted(list(entry) for entry in entries), ensure_ascii=False)
    return compute_hash(payload, length=32)


def summary_input_key(
    fingerprint: str, language: str, child_texts: Sequence[str]
) -> str:
    """The digest a summary is reused by: its children's texts in order,
    the language and the summary settings' ``fingerprint``."""
    payload = json.dumps([fingerprint, language, list(child_texts)], ensure_ascii=False)
    return compute_hash(payload, length=32)


def uniform_filters(maps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The filter keys every one of ``maps`` holds with the same value
    (uniform-only: a node passes a filter only when all its documents do)."""
    if not maps:
        return {}
    first, *rest = maps
    kept: dict[str, Any] = {}
    for key, value in first.items():
        shown = json.dumps(value, sort_keys=True, default=str)
        if all(
            key in other
            and json.dumps(other[key], sort_keys=True, default=str) == shown
            for other in rest
        ):
            kept[key] = value
    return kept


def partition_key(values: Sequence[Any]) -> str:
    """A partition's name: its ``partition_by`` value tuple as JSON."""
    return json.dumps(list(values), ensure_ascii=False, sort_keys=True, default=str)


# --- reduction ----------------------------------------------------------------


def _normalized(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized: np.ndarray = vectors / norms
    return normalized


def reduce_dimensions(vectors: np.ndarray, dims: int) -> np.ndarray:
    """``vectors`` L2-normalized, centered and projected onto their top
    ``min(dims, n - 1)`` principal axes (each axis's largest loading made
    positive), rounded (``_ROUND``). Replaces RAPTOR's UMAP.

    The axes are the eigenvectors of the covariance, accumulated a block of
    rows at a time, so a large level is never copied whole in float64."""
    points = _normalized(np.asarray(vectors, dtype=np.float32))
    n, width = points.shape
    d = max(1, min(dims, n - 1, width))
    if n < 2:
        return np.zeros((n, d))
    mean = points.mean(axis=0, dtype=np.float64)
    covariance = np.zeros((width, width))
    for start in range(0, n, _BLOCK_ROWS):
        block = points[start : start + _BLOCK_ROWS].astype(np.float64) - mean
        covariance += block.T @ block
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # Largest first; a tie keeps the lower index.
    order = sorted(range(width), key=lambda i: (-round(float(eigenvalues[i]), 12), i))
    axes = eigenvectors[:, order[:d]].T
    # Signs are arbitrary in an eigendecomposition: fix each by its largest
    # loading.
    largest = np.argmax(np.abs(axes), axis=1)
    signs = np.sign(axes[np.arange(len(axes)), largest])
    signs[signs == 0] = 1.0
    axes = axes * signs[:, None]
    projected = np.empty((n, d))
    for start in range(0, n, _BLOCK_ROWS):
        block = points[start : start + _BLOCK_ROWS].astype(np.float64) - mean
        projected[start : start + _BLOCK_ROWS] = block @ axes.T
    rounded: np.ndarray = np.round(projected, _ROUND)
    return rounded


# --- k-means ------------------------------------------------------------------


def _squared_distances(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, points by centers (never negative)."""
    distances = (
        np.einsum("ij,ij->i", points, points)[:, None]
        - 2.0 * points @ centers.T
        + np.einsum("ij,ij->i", centers, centers)[None, :]
    )
    clipped: np.ndarray = np.maximum(distances, 0.0)
    return clipped


def _kmeans_pp(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ initial centers (seeded: ``rng``)."""
    n = len(points)
    chosen = [int(rng.integers(n))]
    closest = _squared_distances(points, points[chosen])[:, 0]
    for _ in range(1, k):
        total = float(closest.sum())
        if total <= 0.0:
            # Every point sits on a center: take the first unchosen ones.
            pick = next(i for i in range(n) if i not in chosen)
        else:
            pick = int(np.searchsorted(np.cumsum(closest) / total, rng.random()))
            pick = min(pick, n - 1)
        chosen.append(pick)
        closest = np.minimum(closest, _squared_distances(points, points[[pick]])[:, 0])
    centers: np.ndarray = points[chosen].copy()
    return centers


def _stable_init(
    points: np.ndarray, k: int, keys: Sequence[str], seed: int
) -> np.ndarray:
    """Initial centers a small change of the points barely moves: from the
    ``2k`` points of lowest seeded key hash (a sample a new or removed point
    changes by one at most), the farthest-first ``k`` (each the candidate
    farthest from those chosen; ties to the lower hash). k-means++ samples
    by cumulative distances instead, so one inserted point changed every
    later pick and a delta re-clustered, and re-summarized, the whole tree."""
    ranked = sorted(
        range(len(points)), key=lambda i: compute_hash(f"{seed}\x1f{keys[i]}")
    )
    pool = ranked[: min(len(points), 2 * k)]
    chosen = [pool[0]]
    closest = _squared_distances(points[pool], points[[pool[0]]])[:, 0]
    while len(chosen) < k:
        pick = int(np.argmax(closest))
        if closest[pick] <= 0.0:
            # The rest sit on a chosen center: take them in hash order.
            pick = next(i for i, p in enumerate(pool) if p not in chosen)
        chosen.append(pool[pick])
        closest = np.minimum(
            closest, _squared_distances(points[pool], points[[pool[pick]]])[:, 0]
        )
        closest[pick] = 0.0
    centers: np.ndarray = points[chosen].copy()
    return centers


def _lloyd(
    points: np.ndarray, k: int, seed: int, keys: Sequence[str] | None = None
) -> np.ndarray:
    """Hard k-means labels: initial centers (``_stable_init`` by the points'
    ``keys``, k-means++ without them), then Lloyd's iterations (at most
    ``_MAX_ITER``); a tie goes to the lowest cluster, an empty cluster is
    re-seeded at the point farthest from its own center."""
    n = len(points)
    centers = (
        _kmeans_pp(points, k, np.random.default_rng(seed))
        if keys is None
        else _stable_init(points, k, keys, seed)
    )
    labels = np.full(n, -1)
    for _ in range(_MAX_ITER):
        distances = _squared_distances(points, centers)
        new = np.argmin(distances, axis=1)
        counts = np.bincount(new, minlength=k)
        for cluster in np.flatnonzero(counts == 0):
            # The point farthest from its own center, from a cluster it
            # does not empty.
            own = distances[np.arange(n), new]
            own[counts[new] <= 1] = -1.0
            farthest = int(np.argmax(own))
            counts[new[farthest]] -= 1
            new[farthest] = cluster
            counts[cluster] += 1
        if np.array_equal(new, labels):
            break
        labels = new
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, points)
        filled = counts > 0
        centers[filled] = sums[filled] / counts[filled][:, None]
    return labels


def kmeans(
    points: np.ndarray, k: int, seed: int, keys: Sequence[str] | None = None
) -> np.ndarray:
    """Hard k-means labels of ``points`` (``_lloyd``; ``keys``, the points'
    ids, make the initialization stable under small changes).

    A large problem (``n * k`` over ``_DIRECT_KMEANS``) is solved in two
    steps, as RAPTOR clusters globally then locally: about ``sqrt(k)``
    coarse clusters, then each coarse cluster's points into its share of
    the ``k`` (by size), recursively; so a 200,000-leaf level costs about
    ``2 n sqrt(k)`` distances instead of ``n k``."""
    n = len(points)
    k = max(1, min(k, n))
    if k == 1:
        return np.zeros(n, dtype=int)
    if n * k <= _DIRECT_KMEANS:
        return _lloyd(points, k, seed, keys)
    coarse = max(2, math.isqrt(k))
    coarse_labels = _lloyd(points, coarse, seed, keys)
    labels = np.zeros(n, dtype=int)
    offset = 0
    for cluster in range(coarse):
        members = np.flatnonzero(coarse_labels == cluster)
        if not len(members):
            continue
        share = max(1, min(len(members), round(k * len(members) / n)))
        fine = kmeans(
            points[members],
            share,
            seed,
            None if keys is None else [keys[int(i)] for i in members],
        )
        labels[members] = fine + offset
        offset += int(fine.max()) + 1
    return labels


# --- Gaussian mixtures --------------------------------------------------------


@dataclass(frozen=True)
class Mixture:
    """A fitted Gaussian mixture: its posteriors and BIC."""

    posteriors: np.ndarray
    bic: float


def _log_gaussians(
    points: np.ndarray, means: np.ndarray, covariances: np.ndarray
) -> np.ndarray:
    """``log N(x | mean_k, cov_k)`` per point and component, every component
    at once (a batched Cholesky factor and triangular solve)."""
    n, d = points.shape
    try:
        chol = np.linalg.cholesky(covariances)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(covariances + np.eye(d) * 1e-3)
    diff = points[None, :, :] - means[:, None, :]  # (k, n, d)
    solved = np.linalg.solve(chol, np.swapaxes(diff, 1, 2))  # (k, d, n)
    maha = np.einsum("kdn,kdn->kn", solved, solved)
    log_det = 2.0 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
    out: np.ndarray = (-0.5 * (d * math.log(2.0 * math.pi) + log_det[:, None] + maha)).T
    return out


def _m_step(
    points: np.ndarray, resp: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = points.shape
    counts = resp.sum(axis=0) + 10 * np.finfo(float).eps
    weights = counts / n
    means = (resp.T @ points) / counts[:, None]
    diff = points[None, :, :] - means[:, None, :]  # (k, n, d)
    covariances = np.einsum("nk,kni,knj->kij", resp, diff, diff) / counts[:, None, None]
    covariances[:, np.arange(d), np.arange(d)] += _REG_COVAR
    return weights, means, covariances


def _e_step(
    points: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Log posteriors and each point's log likelihood."""
    weighted = _log_gaussians(points, means, covariances) + np.log(weights)
    top = weighted.max(axis=1, keepdims=True)
    log_norm = top[:, 0] + np.log(np.exp(weighted - top).sum(axis=1))
    return weighted - log_norm[:, None], log_norm


def fit_mixture(
    points: np.ndarray, k: int, seed: int, keys: Sequence[str] | None = None
) -> Mixture:
    """A full-covariance Gaussian mixture of ``k`` components fitted by EM,
    started from the k-means labels (as scikit-learn's ``init_params``
    ``kmeans``), and its BIC."""
    n, d = points.shape
    k = max(1, min(k, n))
    labels = kmeans(points, k, seed, keys)
    resp = np.zeros((n, k))
    resp[np.arange(n), labels] = 1.0
    weights, means, covariances = _m_step(points, resp)
    previous = -np.inf
    for _ in range(_MAX_ITER):
        log_resp, log_norm = _e_step(points, weights, means, covariances)
        weights, means, covariances = _m_step(points, np.exp(log_resp))
        current = float(log_norm.mean())
        if abs(current - previous) < _EM_TOL:
            break
        previous = current
    log_resp, log_norm = _e_step(points, weights, means, covariances)
    parameters = k * d + k * d * (d + 1) / 2 + k - 1
    bic = -2.0 * float(log_norm.sum()) + parameters * math.log(n)
    return Mixture(posteriors=np.exp(log_resp), bic=bic)


def bic_candidates(
    n: int, settings: TreeSettings, expected: float, split: bool = False
) -> list[int]:
    """The cluster counts BIC compares for ``n`` points: ``upstream``, 1 to
    ``min(50, n) - 1`` (RAPTOR's ``max_clusters``); ``bounded``, at most
    ``_BOUNDED_GRID`` values from half to twice ``expected`` (and never
    above upstream's 49)."""
    top = max(1, min(_UPSTREAM_MAX_CLUSTERS, n) - 1)
    if settings.bic_grid == "upstream":
        values = list(range(1, top + 1))
    else:
        center = min(max(1, round(expected)), top)
        low, high = max(1, center // 2), min(top, 2 * center)
        values = sorted(
            {int(v) for v in np.round(np.linspace(low, high, _BOUNDED_GRID - 1))}
            | {center}
        )
    if split:
        values = [v for v in values if v >= 2] or [min(2, n)]
    return values or [1]


def choose_k(
    points: np.ndarray,
    settings: TreeSettings,
    expected: float,
    split: bool = False,
    keys: Sequence[str] | None = None,
) -> int:
    """The cluster count of lowest BIC (the first of a tie)."""
    best_k, best_bic = 1, math.inf
    for k in bic_candidates(len(points), settings, expected, split):
        bic = fit_mixture(points, k, settings.random_state, keys).bic
        if bic < best_bic:
            best_k, best_bic = k, bic
    return best_k


def soft_labels(posteriors: np.ndarray, threshold: float) -> list[list[int]]:
    """Per point, the components whose posterior exceeds ``threshold``, its
    likeliest one when none does (RAPTOR drops such a point from the tree)."""
    labels = []
    for row in posteriors:
        above = [int(i) for i in np.flatnonzero(row > threshold)]
        labels.append(above or [int(np.argmax(row))])
    return labels


def _members(labels: list[list[int]], k: int) -> list[list[int]]:
    return [[i for i, own in enumerate(labels) if c in own] for c in range(k)]


def gmm_clusters(
    vectors: np.ndarray,
    settings: TreeSettings,
    split: bool = False,
    keys: Sequence[str] | None = None,
) -> list[list[int]]:
    """RAPTOR's two-step clustering: a global mixture over the reduced
    vectors, then a local one inside every global cluster of more than
    ``reduction_dims + 1`` members (over those members' own reduction)."""
    n = len(vectors)
    dims = settings.reduction_dims
    global_points = reduce_dimensions(vectors, min(dims, max(1, n - 2)))
    # The global and local counts multiply to about n / target_children.
    k = choose_k(
        global_points, settings, math.sqrt(n / settings.target_children), split, keys
    )
    mixture = fit_mixture(global_points, k, settings.random_state, keys)
    clusters: list[list[int]] = []
    for members in _members(
        soft_labels(mixture.posteriors, settings.soft_threshold), k
    ):
        if not members:
            continue
        if len(members) <= dims + 1:
            clusters.append(members)
            continue
        local_points = reduce_dimensions(vectors[members], dims)
        local_keys = None if keys is None else [keys[i] for i in members]
        local_k = choose_k(
            local_points,
            settings,
            len(members) / settings.target_children,
            keys=local_keys,
        )
        local = fit_mixture(local_points, local_k, settings.random_state, local_keys)
        for local_members in _members(
            soft_labels(local.posteriors, settings.soft_threshold), local_k
        ):
            if local_members:
                clusters.append([members[i] for i in local_members])
    return clusters


def kmeans_clusters(
    vectors: np.ndarray,
    settings: TreeSettings,
    split: bool = False,
    keys: Sequence[str] | None = None,
) -> list[list[int]]:
    """Hard k-means over the reduced vectors, ``k = ceil(n /
    target_children)`` (at least 2 when the caller needs a split)."""
    n = len(vectors)
    k = math.ceil(n / settings.target_children)
    if split:
        k = max(k, 2)
    labels = kmeans(
        reduce_dimensions(vectors, settings.reduction_dims),
        k,
        settings.random_state,
        keys,
    )
    return _groups(labels)


def _groups(labels: np.ndarray) -> list[list[int]]:
    """Row indices by label, labels in ascending order, rows in order."""
    order = np.argsort(labels, kind="stable")
    bounds = np.flatnonzero(np.diff(labels[order])) + 1
    return [[int(i) for i in group] for group in np.split(order, bounds) if len(group)]


def default_partition(settings: TreeSettings) -> Partition:
    """The partition function ``settings.algorithm`` names."""
    cluster = gmm_clusters if settings.algorithm == "gmm" else kmeans_clusters

    def partition(
        vectors: np.ndarray, split: bool, keys: Sequence[str]
    ) -> list[list[int]]:
        return cluster(vectors, settings, split, keys)

    return partition


# --- assembly -----------------------------------------------------------------


def _pack_by_tokens(entries: Sequence[Entry], budget: int) -> list[list[Entry]]:
    """``entries`` in id order, packed into groups within ``budget`` tokens
    (a single entry over it is a group of its own)."""
    groups: list[list[Entry]] = []
    current: list[Entry] = []
    used = 0
    for entry in entries:
        if current and used + entry.n_tokens > budget:
            groups.append(current)
            current, used = [], 0
        current.append(entry)
        used += entry.n_tokens
    if current:
        groups.append(current)
    return groups


def _bounded_clusters(
    entries: Sequence[Entry], partition: Partition, settings: TreeSettings
) -> list[list[Entry]]:
    """One level's clusters, each within ``max_cluster_tokens``: a cluster
    over it is clustered again (at least in two) until it fits, and packed
    in id order when clustering cannot split it."""
    vectors = np.stack([entry.vector for entry in entries])
    pending: list[list[Entry]] = [
        [entries[i] for i in members]
        for members in partition(vectors, False, [entry.id for entry in entries])
    ]
    done: list[list[Entry]] = []
    seen: set[tuple[str, ...]] = set()
    while pending:
        members = sorted(pending.pop(0), key=lambda entry: entry.id)
        key = tuple(entry.id for entry in members)
        if key in seen:
            continue
        tokens = sum(entry.n_tokens for entry in members)
        if len(members) == 1 or tokens <= settings.max_cluster_tokens:
            seen.add(key)
            done.append(members)
            continue
        sub = partition(
            np.stack([entry.vector for entry in members]),
            True,
            [entry.id for entry in members],
        )
        parts = [[members[i] for i in group] for group in sub if group]
        if len(parts) < 2 or any(len(part) == len(members) for part in parts):
            pending.extend(_pack_by_tokens(members, settings.max_cluster_tokens))
            continue
        pending.extend(parts)
    return done


def _entry_for(cluster: Cluster, realized: Realized, node: SummaryNode) -> Entry:
    return Entry(
        id=node.id,
        level=cluster.level,
        text=realized.text,
        vector=np.asarray(realized.vector, dtype=np.float32),
        n_tokens=realized.n_tokens,
        leaves=frozenset().union(*(member.leaves for member in cluster.members)),
        filters=(node.attributes or {}).get(Constants.FILTERS.value) or {},
    )


def build_tree(
    leaves: Sequence[Leaf],
    *,
    settings: TreeSettings,
    realize: Realize,
    language: str,
    partition: Partition | None = None,
    partition_name: str | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> TreeBuild:
    """Build one tree over ``leaves`` (bottom-up, level by level).

    Each level's entries (sorted by id) are clustered (``partition``, by
    default ``settings.algorithm``'s), a cluster over ``max_cluster_tokens``
    clustered again, and every cluster of two or more entries (one too with
    ``summarize_single``) handed to ``realize``, which returns its summary
    or ``None`` when it failed. A failed cluster's entries and a cluster of
    one are promoted to the next level as they are. The build stops at
    ``max_levels``, at a level of at most ``min_nodes_to_cluster`` entries,
    or when the next level would keep more than ``1 - min_level_shrink`` of
    the entries. ``attributes`` (the suffix's ``index``) are given to every
    node, with the filters all its leaves share.
    """
    partition = partition or default_partition(settings)
    current = sorted(
        (
            Entry(
                id=leaf.id,
                level=0,
                text=leaf.text,
                vector=np.asarray(leaf.vector, dtype=np.float32),
                n_tokens=leaf.n_tokens,
                leaves=frozenset({leaf.id}),
                filters=dict(leaf.filters),
            )
            for leaf in leaves
        ),
        key=lambda entry: entry.id,
    )
    nodes: list[SummaryNode] = []
    vectors: dict[str, np.ndarray] = {}
    failed = 0
    top_level = 0
    stop_reason = "max_levels"
    for level in range(1, settings.max_levels + 1):
        if len(current) <= settings.min_nodes_to_cluster:
            stop_reason = "few_nodes"
            break
        groups = _bounded_clusters(current, partition, settings)
        if settings.min_level_shrink > 0 and len(groups) > (
            1.0 - settings.min_level_shrink
        ) * len(current):
            stop_reason = "no_shrink"
            break
        clusters = [
            Cluster(level=level, members=tuple(group))
            for group in groups
            if len(group) > 1 or settings.summarize_single
        ]
        made = realize(clusters) if clusters else []
        if len(made) != len(clusters):
            raise ValueError(
                f"realize returned {len(made)} summaries for {len(clusters)} clusters"
            )
        summarized: dict[tuple[str, ...], Entry] = {}
        for cluster, realized in zip(clusters, made, strict=True):
            if realized is None:
                failed += 1
                continue
            node = _summary_node(
                cluster, realized, settings, language, partition_name, attributes
            )
            nodes.append(node)
            entry = _entry_for(cluster, realized, node)
            # One array for the entry and the node: a vector as a list of
            # float objects took eight times its array, per node.
            vectors[node.id] = entry.vector
            summarized[tuple(cluster.child_ids)] = entry
        # A summary takes its members' place; an entry no summary covers (a
        # cluster of one, a failed summary) moves up as it is. Under soft
        # membership an entry in both stays only under its summary.
        covered = {child for key in summarized for child in key}
        following: dict[str, Entry] = {entry.id: entry for entry in summarized.values()}
        for group in groups:
            for member in group:
                if member.id not in covered:
                    following.setdefault(member.id, member)
        if not summarized:
            stop_reason = "no_summary"
            break
        top_level = level
        current = sorted(following.values(), key=lambda entry: entry.id)
    else:
        stop_reason = "max_levels"
    return TreeBuild(
        nodes=nodes,
        vectors=vectors,
        levels=top_level,
        failed=failed,
        stop_reason=stop_reason,
    )


def _summary_node(
    cluster: Cluster,
    realized: Realized,
    settings: TreeSettings,
    language: str,
    partition_name: str | None,
    attributes: Mapping[str, Any] | None,
) -> SummaryNode:
    leaves = sorted(frozenset().union(*(member.leaves for member in cluster.members)))
    filters = uniform_filters([member.filters for member in cluster.members])
    node_attributes = dict(attributes or {})
    if filters:
        node_attributes[Constants.FILTERS.value] = filters
    cap = settings.max_lineage_text_units
    return SummaryNode(
        id=node_id(cluster.level, partition_name, cluster.child_ids),
        level=cluster.level,
        text=realized.text,
        child_ids=cluster.child_ids,
        text_unit_ids=leaves[:cap],
        lineage_complete=len(leaves) <= cap,
        n_leaves=len(leaves),
        n_tokens=realized.n_tokens,
        partition=partition_name,
        language=language,
        input_key=realized.input_key,
        attributes=node_attributes or None,
    )
