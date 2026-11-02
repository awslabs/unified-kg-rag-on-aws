# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invariants of converting a seed graph into shard documents.

Over random seeds (declared, respelled and undeclared types, colliding
names, duplicate edges and self-loops): the documents do not depend on the
order of the nodes and edges, a node added without edges changes only its
own shard, and strict keeps a subset of the statements align keeps.

Through the rdf source (the ontology extra): an A-Box read as N-Triples or
Turtle gives the same graph and documents whatever the order of its lines
and the labels of its blank nodes.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import pytest
from hypothesis import event, given
from hypothesis import strategies as st

from tests.fixtures.extras import requires_ontology
from unified_kg_rag.adapters.seed_sources import SeedSourceContext, get_seed_source
from unified_kg_rag.application.bootstrap import (
    load_seed_graphs,
    resolve_extraction_schema,
)
from unified_kg_rag.domain.ingestion.seed_graph import (
    SeedConversion,
    SeedConversionSettings,
    convert_seed,
    seed_statements,
    shard_of,
)
from unified_kg_rag.domain.models import Config, ExtractionSchema
from unified_kg_rag.domain.models.config import SeedGraphConfig
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


# --- through the RDF loader -------------------------------------------------------

_EX = "https://example.org/procure#"
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
_ONTOLOGY = Path(__file__).parents[1] / "fixtures" / "ontologies" / "procure.ttl"
# Declared classes (a subclass and its parent), an undeclared one; declared
# predicates and an undeclared one.
_CLASSES = ["Vendor", "Organization", "Buyer", "Person", "Distributor"]
_PREDICATES = ["supplies", "employs", "partnerOf"]
_INSTANCES = [f"i{n}" for n in range(4)]
_BLANKS = [f"b{n}" for n in range(3)]


def _iri(local: str) -> str:
    return f"<{_EX}{local}>"


def _literal(text: str, language: str) -> str:
    return f'"{text}"@{language}' if language else f'"{text}"'


_instance = st.sampled_from(_INSTANCES).map(_iri)
_blank = st.sampled_from(_BLANKS).map(lambda label: f"_:{label}")


def _typing(subjects: st.SearchStrategy[str]) -> st.SearchStrategy:
    return st.tuples(
        subjects, st.just(f"<{_RDF_TYPE}>"), st.sampled_from(_CLASSES).map(_iri)
    )


def _linking(
    subjects: st.SearchStrategy[str], objects: st.SearchStrategy[str]
) -> st.SearchStrategy:
    return st.tuples(
        subjects, st.sampled_from([*_PREDICATES, "contact"]).map(_iri), objects
    )


_describing = st.one_of(
    st.tuples(
        _instance | _blank,
        st.just(f"<{_RDFS_LABEL}>"),
        st.tuples(
            st.sampled_from(["Acme", "Globex", "공급업체"]),
            st.sampled_from(["en", "ko", ""]),
        ).map(lambda label: _literal(*label)),
    ),
    st.tuples(
        _instance | _blank, st.just(_iri("country")), st.sampled_from(['"DE"', '"FR"'])
    ),
)
# Typed subjects (instances, with several classes), links and literals.
_abox = st.tuples(
    st.lists(_typing(_instance), min_size=1, max_size=6),
    st.lists(_typing(_blank), max_size=2),
    st.lists(_linking(_instance, _instance), max_size=6),
    st.lists(_linking(_instance | _blank, _instance | _blank), max_size=2),
    st.lists(_describing, max_size=5),
).map(lambda parts: list(dict.fromkeys(t for part in parts for t in part)))


@cache
def _rdf_config(mode: str) -> Config:
    """A config whose extraction schema is the procure ontology (resolved
    once per mode)."""
    return resolve_extraction_schema(
        Config.model_validate(
            {
                "processing": {
                    "graph_extraction": {
                        "schema_mode": mode,
                        "schema_source": {"kind": "owl", "path": str(_ONTOLOGY)},
                    }
                }
            }
        )
    )


def _load_rdf(path: Path, mode: str) -> tuple[SeedGraph, SeedConversion]:
    """The rdf source's graph of ``path`` and its conversion by bootstrap."""
    config = _rdf_config(mode).model_copy(deep=True)
    seed = SeedGraphConfig(
        name="procure",
        kind="rdf",
        path=str(path),
        shards=3,
        on_name_collision="merge",
        options={"prefixes": {"ex": _EX}},
    )
    config.processing.seed_graphs = [seed]
    graph = get_seed_source("rdf")(SeedSourceContext(seed, config))
    (conversion,) = load_seed_graphs(config)
    return graph, conversion


@pytest.fixture(scope="module")
def rdf_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("rdf-seeds")


@requires_ontology
@given(
    triples=_abox,
    mode=st.sampled_from(["off", "align", "strict"]),
    suffix=st.sampled_from([".nt", ".ttl"]),
    data=st.data(),
)
def test_rdf_conversion_does_not_depend_on_triple_order_or_blank_labels(
    rdf_dir: Path,
    triples: list[tuple[str, str, str]],
    mode: str,
    suffix: str,
    data: st.DataObject,
) -> None:
    """The same A-Box, its lines permuted and its blank nodes renamed, read
    by the rdf source (N-Triples, or the same lines as Turtle), gives the
    same graph, counts and shard documents."""
    labels = data.draw(st.permutations([f"x{n}" for n in range(len(_BLANKS))]))
    renamed = dict(zip(_BLANKS, labels, strict=True))

    def rename(term: str) -> str:
        return f"_:{renamed[term[2:]]}" if term.startswith("_:") else term

    lines = [" ".join(triple) + " ." for triple in triples]
    shuffled = [
        " ".join(rename(term) for term in triple) + " ."
        for triple in data.draw(st.permutations(triples))
    ]
    first, second = rdf_dir / f"first{suffix}", rdf_dir / f"second{suffix}"
    first.write_text("\n".join(lines) + "\n", encoding="utf-8")
    second.write_text("\n".join(shuffled) + "\n", encoding="utf-8")

    graph, conversion = _load_rdf(first, mode)
    other_graph, other_conversion = _load_rdf(second, mode)

    event(f"{len(graph.nodes)} nodes, {len(graph.edges)} edges")
    assert (graph.nodes, graph.edges) == (other_graph.nodes, other_graph.edges)
    assert graph.report.skipped == other_graph.report.skipped
    assert _texts(conversion) == _texts(other_conversion)
    assert conversion.digest == other_conversion.digest
    assert conversion.stats == other_conversion.stats
