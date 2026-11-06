# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Differential tests against the original HippoRAG 2 (optional).

HippoRAG 2 (OSU-NLP-Group/HippoRAG, MIT) is never a dependency: these tests
run only when ``python-igraph`` is importable (the PPR it calls) and, for
its seeding, when ``HIPPORAG_SRC`` names a checkout of the repository. They
execute the original functions (``HippoRAG.run_ppr``,
``graph_search_with_fact_entities``, ``get_top_k_weights`` and the
``misc_utils`` helpers they use, read from the checkout's source) on the
same graph, reset vector and damping as ``domain.retrieval.ppr`` and compare
the scores and the passage order. Run them with::

    HIPPORAG_SRC=/path/to/HippoRAG uv run --with python-igraph==0.11.8 \\
        pytest tests/unit/test_ppr_hipporag_parity.py

docs/design.md §6.3 lists the deliberate differences (a bounded
neighbourhood, hybrid seeds, no recognition-memory filter), which these
fixtures leave out: they compare the algorithm on identical inputs.
"""

from __future__ import annotations

import ast
import os
import time
import typing
from hashlib import md5
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from unified_kg_rag.domain.retrieval.ppr import (
    PassageGraph,
    passage_weights,
    personalized_pagerank,
    seed_weights,
)

pytestmark = pytest.mark.unit

igraph = pytest.importorskip("igraph", reason="python-igraph is not installed")

_TIGHT = {"max_iterations": 500, "tolerance": 1e-14}


def _igraph_ppr(nodes, edges, reset, damping):
    """``HippoRAG.run_ppr``'s call: undirected, weighted, PRPACK."""
    index = {node: i for i, node in enumerate(nodes)}
    graph = igraph.Graph(n=len(nodes))
    graph.add_edges([(index[u], index[v]) for u, v, _ in edges])
    graph.es["weight"] = [w for _, _, w in edges]
    vector = np.zeros(len(nodes))
    for node, value in reset.items():
        vector[index[node]] = value
    scores = graph.personalized_pagerank(
        vertices=range(len(nodes)),
        damping=damping,
        directed=False,
        weights="weight",
        reset=vector,
        implementation="prpack",
    )
    return dict(zip(nodes, scores, strict=True))


@pytest.mark.parametrize("seed", range(25))
def test_scores_match_igraph_personalized_pagerank(seed: int) -> None:
    # Random weighted graphs with nodes without edges (whose mass returns
    # to the reset vector) and several seeds.
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 40))
    nodes = [f"v{i:03d}" for i in range(n)]
    edges = []
    for _ in range(int(rng.integers(1, 3 * n))):
        u, v = rng.choice(n, size=2, replace=False)
        edges.append((nodes[u], nodes[v], float(rng.uniform(0.1, 5.0))))
    seeds = rng.choice(n, size=int(rng.integers(1, max(2, n // 3))), replace=False)
    reset = {nodes[i]: float(rng.uniform(0.01, 1.0)) for i in seeds}
    damping = float(rng.choice([0.5, 0.3, 0.85]))
    ours = personalized_pagerank(nodes, edges, reset, damping=damping, **_TIGHT)
    theirs = _igraph_ppr(nodes, edges, reset, damping)
    assert ours.scores == pytest.approx(theirs, abs=1e-9)


# --- the original's seeding, executed from its source ----------------------

_SOURCE = os.environ.get("HIPPORAG_SRC")
_METHODS = ("graph_search_with_fact_entities", "get_top_k_weights", "run_ppr")


def _original() -> SimpleNamespace:
    root = Path(_SOURCE or "")
    package = next(
        (p for p in (root / "src" / "hipporag", root) if (p / "HippoRAG.py").is_file()),
        None,
    )
    if package is None:
        pytest.skip("HIPPORAG_SRC does not name a HippoRAG checkout")
    namespace: dict[str, Any] = {
        "np": np,
        "time": time,
        "md5": md5,
        "logger": SimpleNamespace(info=lambda *a, **k: None),
        "StateConsistencyError": RuntimeError,
        **{
            name: getattr(typing, name)
            for name in ("Dict", "List", "Tuple", "Optional")
        },
    }
    utils = ast.parse((package / "utils" / "misc_utils.py").read_text("utf-8"))
    main = ast.parse((package / "HippoRAG.py").read_text("utf-8"))
    (cls,) = [
        n for n in main.body if isinstance(n, ast.ClassDef) and n.name == "HippoRAG"
    ]
    picked = [
        n
        for n in utils.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("min_max_normalize", "compute_mdhash_id")
    ] + [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in _METHODS]
    assert len(picked) == 2 + len(_METHODS), "the original's functions moved"
    module = ast.Module(body=picked, type_ignores=[])
    # The original's own code (MIT), run in this namespace only.
    exec(
        compile(module, str(package / "HippoRAG.py"), "exec"), namespace
    )  # noqa: S102  # nosec B102
    return SimpleNamespace(**namespace)


# A fixture with distinct scores (no ties, whose order the two break apart).
_PASSAGES = {
    "p1": ["alvar", "bexley"],
    "p2": ["bexley", "corin"],
    "p3": ["corin", "dorrit"],
    "p4": ["dorrit"],
    "p5": ["alvar", "elm", "fen"],
    "p6": ["fen"],
}
_FACTS = [  # (subject, predicate, object, query-to-fact score)
    ("alvar", "founded", "bexley", 0.91),
    ("bexley", "acquired", "corin", 0.74),
    ("corin", "based in", "dorrit", 0.52),
    ("elm", "near", "fen", 0.33),
    ("alvar", "visited", "fen", 0.21),
    ("alvar", "founded", "bexley", 0.0),  # a repeated triple: weight 2
]
_DENSE = {"p1": 0.9, "p2": 0.4, "p3": 0.65, "p4": 0.1, "p5": 0.3, "p6": 0.2}


def _their_retrieval(original: SimpleNamespace, facts, link_top_k: int):
    key = original.compute_mdhash_id
    entities = sorted({e for ents in _PASSAGES.values() for e in ents})
    names = [key(e, prefix="entity-") for e in entities] + list(_PASSAGES)
    index = {name: i for i, name in enumerate(names)}
    edges: dict[tuple[str, str], float] = {}
    for subject, _, obj, _ in _FACTS:
        pair = tuple(
            sorted((key(subject, prefix="entity-"), key(obj, prefix="entity-")))
        )
        edges[pair] = edges.get(pair, 0.0) + 1.0  # add_fact_edges
    for passage, ents in _PASSAGES.items():
        for e in ents:
            edges[(passage, key(e, prefix="entity-"))] = 1.0  # add_passage_edges
    graph = igraph.Graph(n=len(names))
    graph.vs["name"] = names
    graph.add_edges([(index[u], index[v]) for u, v in edges])
    graph.es["weight"] = list(edges.values())
    passage_keys = list(_PASSAGES)
    dense = np.array([_DENSE[p] for p in passage_keys])
    order = np.argsort(dense)[::-1]
    recorded: dict[str, Any] = {}

    def run_ppr(reset_prob, damping=0.5):
        recorded["reset"] = np.array(reset_prob, dtype=float)
        return original.run_ppr(stub, reset_prob, damping)

    stub = SimpleNamespace(
        graph=graph,
        node_name_to_vertex_idx=index,
        ent_node_to_chunk_ids={
            key(e, prefix="entity-"): {p for p, ents in _PASSAGES.items() if e in ents}
            for e in entities
        },
        passage_node_keys=passage_keys,
        passage_node_idxs=[index[p] for p in passage_keys],
        chunk_embedding_store=SimpleNamespace(get_row=lambda k: {"content": k}),
        global_config=SimpleNamespace(damping=0.5),
        ppr_time=0.0,
        dense_passage_retrieval=lambda query: (order, dense[order]),
        run_ppr=run_ppr,
    )
    stub.get_top_k_weights = lambda *a: original.get_top_k_weights(stub, *a)
    top = sorted(range(len(facts)), key=lambda i: -facts[i][3])
    doc_ids, doc_scores = original.graph_search_with_fact_entities(
        stub,
        query="q",
        link_top_k=link_top_k,
        query_fact_scores=np.array([f[3] for f in facts]),
        top_k_facts=[facts[i][:3] for i in top],
        top_k_fact_indices=top,
        passage_node_weight=0.05,
    )
    ranked = [passage_keys[i] for i in doc_ids]
    reset = {
        name: float(recorded["reset"][i])
        for name, i in index.items()
        if recorded["reset"][i] > 0
    }
    entity_of = {key(e, prefix="entity-"): e for e in entities}
    return ranked, [float(s) for s in doc_scores], reset, entity_of


@pytest.mark.skipif(not _SOURCE, reason="set HIPPORAG_SRC to a HippoRAG checkout")
@pytest.mark.parametrize("link_top_k", [5, 2])
def test_seeding_and_passage_ranking_match_hipporag_2(link_top_k: int) -> None:
    original = _original()
    facts = _FACTS[:5]
    ranked, scores, their_reset, entity_of = _their_retrieval(
        original, facts, link_top_k
    )

    # Ours: the same graph (a repeated triple is a second relationship).
    graph = PassageGraph()
    for subject, _, obj, _ in _FACTS:
        graph.add_relationship(subject, obj)
    for passage, ents in _PASSAGES.items():
        for e in ents:
            graph.link_passages(e, [passage])
    counts = {
        e: sum(e in ents for ents in _PASSAGES.values())
        for ents in _PASSAGES.values()
        for e in ents
    }
    entity_reset = seed_weights(
        [(score, [s, o]) for s, _, o, score in facts], counts, link_top_k
    )
    passage_reset = passage_weights(_DENSE, 0.05)

    # The same reset vector (before normalization) ...
    ours = {k: v for k, v in {**entity_reset, **passage_reset}.items() if v > 0}
    theirs = {entity_of.get(name, name): value for name, value in their_reset.items()}
    assert ours.keys() == theirs.keys()
    assert ours == pytest.approx(theirs, abs=1e-12)

    # ... the same passage scores and order.
    our_ranked, result = graph.rank_passages(
        entity_reset, passage_reset, damping=0.5, **_TIGHT
    )
    assert result is not None
    assert [unit for unit, _ in our_ranked] == ranked
    assert [score for _, score in our_ranked] == pytest.approx(scores, abs=1e-9)
