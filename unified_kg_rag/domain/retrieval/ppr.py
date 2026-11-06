# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Personalized PageRank over entities and passages (HippoRAG 2's ranking).

HippoRAG 2 (Gutiérrez et al., ICML 2025; OSU-NLP-Group/HippoRAG, MIT) seeds
a Personalized PageRank (PPR) with the entities of the facts most similar to
the question and with a passage search, over a graph of entities and the
passages that cite them, and ranks the passages by their PPR score. This
module is that algorithm, reimplemented on in-memory data: the search
strategies read the graph and the seeds from the stores
(docs/design.md §6.4) and hand them here.

- :func:`personalized_pagerank`: power iteration on an undirected weighted
  graph, the mass of a node without edges returned to the reset vector (as
  igraph's PRPACK personalized PageRank, which HippoRAG 2 calls).
- :func:`seed_weights` / :func:`passage_weights`: HippoRAG 2's reset vector
  (:func:`summed_seed_weights`: the variant ``local``'s ranking uses).
- :func:`cap_edges_per_vertex`: the per-vertex edge bound every backend's
  ``aneighbourhood_edges`` applies.
- :class:`PassageGraph`: the entity + passage graph, ranked.

Everything is deterministic: nodes are ordered by id and ties rank by id.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, Field

from unified_kg_rag.domain.models import GraphEdge

# The defaults HippoRAG 2 runs with (its ``BaseConfig``).
DEFAULT_DAMPING = 0.5
DEFAULT_LINK_TOP_K = 5


class PPRResult(BaseModel):
    """A PPR run: each node's score (summing to 1) and how it converged."""

    scores: dict[str, float] = Field(description="PPR score per node id")
    iterations: int = Field(description="Power-iteration steps taken")
    converged: bool = Field(
        description="Whether the L1 change fell below the tolerance"
    )


def personalized_pagerank(
    nodes: Iterable[str],
    edges: Iterable[tuple[str, str, float]],
    reset: Mapping[str, float],
    *,
    damping: float = DEFAULT_DAMPING,
    max_iterations: int = 30,
    tolerance: float = 1e-6,
) -> PPRResult:
    """The PPR of ``reset`` over the undirected graph ``nodes`` / ``edges``.

    ``edges`` are ``(u, v, weight)``: parallel edges add up, self-loops and
    edges of a non-positive weight are dropped, and an endpoint not in
    ``nodes`` is added. ``reset`` maps nodes to non-negative weights,
    normalized to sum 1 (a node it names that is not in the graph is
    added). ``damping`` is the probability a walk follows an edge (HippoRAG
    2: 0.5); a walk at a node without edges restarts on the reset vector.
    Iterates ``p = (1 - d) r + d (p P + (p . dangling) r)`` from ``p = r``
    until the L1 change is below ``tolerance`` or ``max_iterations`` steps.

    Raises:
        ValueError: ``damping`` outside (0, 1), or a reset vector that is
            negative, not finite or without mass.
    """
    if not 0.0 < damping < 1.0:
        raise ValueError(f"damping must be in (0, 1), got {damping}")
    weights: dict[tuple[str, str], float] = {}
    names = {str(node) for node in nodes}
    for head, tail, weight in edges:
        u, v = str(head), str(tail)
        names.update((u, v))
        if u == v or not weight > 0:
            continue
        key = (u, v) if u < v else (v, u)
        weights[key] = weights.get(key, 0.0) + float(weight)
    for node, value in reset.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"reset weight of '{node}' is {value}")
        names.add(str(node))
    order = sorted(names)
    index = {node: i for i, node in enumerate(order)}
    n = len(order)
    r = np.zeros(n)
    for node, value in reset.items():
        r[index[str(node)]] += float(value)
    total = r.sum()
    if not total > 0:
        raise ValueError("the reset vector has no mass")
    r /= total
    pairs = sorted(weights)
    if pairs:
        src = np.array([index[u] for u, _ in pairs] + [index[v] for _, v in pairs])
        dst = np.array([index[v] for _, v in pairs] + [index[u] for u, _ in pairs])
        w = np.array([weights[p] for p in pairs] * 2)
    else:
        src = dst = np.zeros(0, dtype=int)
        w = np.zeros(0)
    strength = np.bincount(src, weights=w, minlength=n)
    dangling = strength == 0
    share = w / strength[src] if len(w) else w
    p = r.copy()
    converged = False
    steps = 0
    for step in range(1, max_iterations + 1):
        steps = step
        spread = np.bincount(dst, weights=p[src] * share, minlength=n)
        following = (1.0 - damping) * r + damping * (spread + p[dangling].sum() * r)
        change = float(np.abs(following - p).sum())
        p = following
        if change < tolerance:
            converged = True
            break
    p /= p.sum()
    return PPRResult(
        scores={node: float(p[i]) for i, node in enumerate(order)},
        iterations=steps,
        converged=converged,
    )


def min_max(scores: Mapping[str, float]) -> dict[str, float]:
    """``scores`` min-max normalized to [0, 1]; all 1 when they tie (as
    HippoRAG 2's ``min_max_normalize``)."""
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high == low:
        return dict.fromkeys(scores, 1.0)
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def seed_weights(
    facts: Sequence[tuple[float, Sequence[str]]],
    unit_counts: Mapping[str, int],
    link_top_k: int | None = DEFAULT_LINK_TOP_K,
) -> dict[str, float]:
    """HippoRAG 2's entity reset weights from scored ``facts``.

    Each fact is ``(score, entity ids)``: its endpoints (or one entity a
    hit names). An entity weighs the mean over the facts naming it of the
    fact's score divided by the passages it cites (``unit_counts``, at
    least 1), so an entity in many passages seeds less
    (``graph_search_with_fact_entities``). With ``link_top_k`` only that
    many entities, best first (then by id), keep their weight
    (``get_top_k_weights``). Non-positive scores seed nothing.
    """
    total: dict[str, float] = {}
    occurrences: dict[str, int] = {}
    for score, entities in facts:
        # Each end counts, as HippoRAG 2 counts a self-loop's subject and
        # object apart.
        for entity in (str(e) for e in entities if e):
            weighted = max(float(score), 0.0) / max(1, unit_counts.get(entity, 0))
            total[entity] = total.get(entity, 0.0) + weighted
            occurrences[entity] = occurrences.get(entity, 0) + 1
    weights = {
        entity: value / occurrences[entity]
        for entity, value in total.items()
        if value > 0
    }
    if link_top_k is not None and link_top_k >= 0:
        kept = sorted(weights, key=lambda e: (-weights[e], e))[:link_top_k]
        weights = {entity: weights[entity] for entity in kept}
    return weights


def summed_seed_weights(
    entity_scores: Mapping[str, float],
    facts: Sequence[tuple[float, Sequence[str]]],
    unit_counts: Mapping[str, int],
) -> dict[str, float]:
    """Entity reset weights for ``local``'s ranking: an entity's own hit
    score (``entity_scores``) plus the mean score of the ``facts`` naming
    it, divided by the passages it cites (at least 1). The variant measured
    on this stack (docs/design.md §6.4); ``seed_weights`` is HippoRAG 2's
    for the ``ppr`` strategy. Non-positive totals seed nothing."""
    fact_scores: dict[str, list[float]] = {}
    for score, entities in facts:
        for entity in (str(e) for e in entities if e):
            fact_scores.setdefault(entity, []).append(float(score))
    weights: dict[str, float] = {}
    for entity in dict.fromkeys([*entity_scores, *fact_scores]):
        total = float(entity_scores.get(entity, 0.0))
        if scores := fact_scores.get(entity):
            total += sum(scores) / len(scores)
        if total > 0:
            weights[entity] = total / max(1, unit_counts.get(entity, 0))
    return weights


def passage_weights(
    scores: Mapping[str, float], passage_node_weight: float
) -> dict[str, float]:
    """HippoRAG 2's passage reset weights: ``passage_node_weight`` times each
    passage's score min-max normalized over ``scores``."""
    return {
        unit: passage_node_weight * value for unit, value in min_max(scores).items()
    }


def cap_edges_per_vertex(
    edges: Iterable[GraphEdge], vertices: Iterable[str], per_vertex: int
) -> list[GraphEdge]:
    """The ``edges`` between two of ``vertices``, each vertex contributing
    its first ``per_vertex`` by relationship id (an edge one endpoint keeps
    is kept), each once, sorted by ``(source_id, target_id, id)``.

    A self-loop takes two of its vertex's slots, as it is listed twice
    among a vertex's incident edges (Gremlin's ``bothE()``).
    """
    members = {str(v) for v in vertices}
    unique: dict[str, GraphEdge] = {}
    for edge in edges:
        if edge.source_id in members and edge.target_id in members:
            unique.setdefault(edge.id, edge)
    incident: dict[str, list[GraphEdge]] = {}
    for edge in unique.values():
        incident.setdefault(edge.source_id, []).append(edge)
        incident.setdefault(edge.target_id, []).append(edge)
    kept: dict[str, GraphEdge] = {}
    for listed in incident.values():
        for edge in sorted(listed, key=lambda e: e.id)[: max(per_vertex, 0)]:
            kept.setdefault(edge.id, edge)
    return sorted(kept.values(), key=lambda e: (e.source_id, e.target_id, e.id))


class PassageGraph:
    """HippoRAG 2's graph of entities and passages, for one query.

    Entities and passages live in separate namespaces (an entity and a text
    unit may share an id). An entity-entity edge weighs 1 per relationship,
    an entity-passage edge 1. Build it, then :meth:`rank_passages`.
    """

    _ENTITY = "e:"
    _PASSAGE = "p:"

    def __init__(self) -> None:
        self._nodes: set[str] = set()
        self._edges: list[tuple[str, str, float]] = []
        self._passages: set[str] = set()

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def passages(self) -> frozenset[str]:
        return frozenset(self._passages)

    def add_entity(self, entity_id: str) -> None:
        self._nodes.add(self._ENTITY + entity_id)

    def add_passage(self, unit_id: str) -> None:
        self._nodes.add(self._PASSAGE + unit_id)
        self._passages.add(unit_id)

    def add_relationship(self, source_id: str, target_id: str) -> None:
        """One relationship between two entities (each added if new)."""
        self.add_entity(source_id)
        self.add_entity(target_id)
        self._edges.append((self._ENTITY + source_id, self._ENTITY + target_id, 1.0))

    def link_passages(self, entity_id: str, unit_ids: Iterable[str]) -> None:
        """An edge from the entity to each passage it cites."""
        self.add_entity(entity_id)
        for unit_id in dict.fromkeys(unit_ids):
            self.add_passage(unit_id)
            self._edges.append((self._ENTITY + entity_id, self._PASSAGE + unit_id, 1.0))

    def rank_passages(
        self,
        entity_reset: Mapping[str, float],
        passage_reset: Mapping[str, float],
        *,
        damping: float = DEFAULT_DAMPING,
        max_iterations: int = 30,
        tolerance: float = 1e-6,
    ) -> tuple[list[tuple[str, float]], PPRResult | None]:
        """The passages by PPR score (best first, then by id) and the run;
        ``([], None)`` when the reset vector has no mass or the graph no
        passage. Reset entries name entities and passages by their own ids
        (one not in the graph is added)."""
        reset = {
            **{self._ENTITY + k: v for k, v in entity_reset.items() if v > 0},
            **{self._PASSAGE + k: v for k, v in passage_reset.items() if v > 0},
        }
        for unit_id in passage_reset:
            if passage_reset[unit_id] > 0:
                self.add_passage(unit_id)
        if not reset or not self._passages:
            return [], None
        result = personalized_pagerank(
            self._nodes,
            self._edges,
            reset,
            damping=damping,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        ranked = sorted(
            (
                (unit_id, result.scores[self._PASSAGE + unit_id])
                for unit_id in self._passages
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked, result
