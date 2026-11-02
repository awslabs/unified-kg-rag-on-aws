# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invariants of the ontology mapping over random class and property
hierarchies.

The mapping depends on the statements only (not their order); stating what
its closure already implies changes nothing; what it returns is a valid
extraction schema with one label per type, or it raises; narrowing the
classes never adds a type; and every pattern end is a class the domain or
range in effect allows (inherited, from an inverse, or from a symmetric
property's other end), or the nearest selected ancestor of one.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from unified_kg_rag.adapters.schema_sources.ontology_mapping import (
    MappingOptions,
    map_ontology,
)
from unified_kg_rag.adapters.schema_sources.ontology_model import (
    RDFS_LABEL,
    OntologyClass,
    OntologyModel,
    OntologyProperty,
)
from unified_kg_rag.adapters.source_files.rdf_terms import Text
from unified_kg_rag.domain.models.schema import ExtractionSchema, schema_key
from unified_kg_rag.shared.exceptions import SchemaSourceError

pytestmark = pytest.mark.property

EX = "https://example.org/a#"
GEO = "https://example.org/b#"
# Few distinct labels, so collisions happen.
_LABELS = ["Vendor", "vendor", "Buyer", "공급 업체", "Part of", "!!", ""]


@st.composite
def models(
    draw: st.DrawFn, *, equivalences: bool = True, symmetric: bool = True
) -> OntologyModel:
    n_classes = draw(st.integers(1, 7))
    class_iris = [
        (GEO if draw(st.booleans()) and i % 3 == 2 else EX) + f"C{i}"
        for i in range(n_classes)
    ]
    classes = []
    for i, iri in enumerate(class_iris):
        parents = (
            draw(st.sets(st.sampled_from(class_iris[:i]), max_size=2)) if i else set()
        )
        same = (
            draw(st.sets(st.sampled_from(class_iris), max_size=1)) - {iri}
            if equivalences and draw(st.integers(0, 4)) == 0
            else set()
        )
        label = draw(st.sampled_from(_LABELS))
        classes.append(
            OntologyClass(
                iri=iri,
                parents=frozenset(parents),
                equivalents=frozenset(same),
                deprecated=draw(st.integers(0, 9)) == 0,
                annotations=(
                    {
                        RDFS_LABEL: (
                            Text(
                                value=label,
                                language=draw(st.sampled_from(["en", "", "ko"])),
                            ),
                        )
                    }
                    if label
                    else {}
                ),
            )
        )
    n_properties = draw(st.integers(0, 5))
    property_iris = [EX + f"p{i}" for i in range(n_properties)]
    union = st.frozensets(st.sampled_from(class_iris), min_size=1, max_size=2)
    properties = []
    for i, iri in enumerate(property_iris):
        properties.append(
            OntologyProperty(
                iri=iri,
                kind=draw(st.sampled_from(["object", "object", "untyped", "datatype"])),
                parents=frozenset(
                    draw(st.sets(st.sampled_from(property_iris[:i]), max_size=2))
                    if i
                    else set()
                ),
                domains=draw(st.frozensets(union, max_size=2)),
                ranges=draw(st.frozensets(union, max_size=2)),
                # Its own inverse is symmetric.
                inverses=frozenset(
                    draw(st.sets(st.sampled_from(property_iris), max_size=1))
                    - (set() if symmetric else {iri})
                ),
                symmetric=symmetric and draw(st.integers(0, 4)) == 0,
            )
        )
    return OntologyModel(
        classes=tuple(classes),
        properties=tuple(properties),
        prefixes={"ex": EX, "geo": GEO},
    )


def _outcome(model: OntologyModel, options: MappingOptions) -> object:
    try:
        mapping = map_ontology(model, options)
    except SchemaSourceError as e:
        return ("error", str(e))
    return ("ok", mapping.document, mapping.report)


def _shuffled(model: OntologyModel, seed: int) -> OntologyModel:
    rng = random.Random(seed)

    def shuffle(items: Iterable) -> list:
        items = list(items)
        rng.shuffle(items)
        return items

    return OntologyModel(
        classes=tuple(shuffle(model.classes)),
        properties=tuple(shuffle(model.properties)),
        prefixes=dict(shuffle(model.prefixes.items())),
    )


_OPTIONS = st.builds(
    MappingOptions,
    on_collision=st.sampled_from(["error", "qualify"]),
    max_depth=st.sampled_from([None, 0, 1]),
    label_case=st.sampled_from(["upper_snake", "preserve"]),
    include_deprecated=st.booleans(),
)


@given(model=models(), options=_OPTIONS, seed=st.integers(0, 2**16))
def test_order_of_statements_does_not_matter(
    model: OntologyModel, options: MappingOptions, seed: int
) -> None:
    assert _outcome(_shuffled(model, seed), options) == _outcome(model, options)


def _closed(
    parents: dict[str, set[str]], equivalents: dict[str, set[str]]
) -> dict[str, set[str]]:
    """Every term's proper ancestors (equivalents count both ways)."""
    up: dict[str, set[str]] = {}
    for node in parents:
        seen: set[str] = set()
        frontier = [node]
        while frontier:
            current = frontier.pop()
            for nxt in parents.get(current, set()) | equivalents.get(current, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        up[node] = seen - {node}
    return up


def _both_ways(
    terms: Iterable[OntologyProperty | OntologyClass],
) -> dict[str, set[str]]:
    same: dict[str, set[str]] = {term.iri: set(term.equivalents) for term in terms}
    for term in terms:
        for other in term.equivalents:
            same.setdefault(other, set()).add(term.iri)
    return same


Ends = frozenset[frozenset[str]]


def _property_closure(
    model: OntologyModel,
) -> tuple[dict[str, tuple[Ends, Ends]], set[str]]:
    """Every property's domain and range statements in effect, by a naive
    fixpoint: its own, its super-properties' (and equivalents') same end,
    its inverses' other end; then a symmetric property with statements on
    one end only takes them for the other, and the fixpoint runs again.
    And the symmetric properties (flagged, or their own inverse)."""
    by_iri = {p.iri: p for p in model.properties}
    up = _closed(
        {p.iri: set(p.parents) for p in model.properties}, _both_ways(model.properties)
    )

    def group(iri: str) -> frozenset[str]:
        return frozenset({iri} | {m for m in up[iri] if iri in up.get(m, set())})

    inverses: dict[str, set[str]] = {iri: set() for iri in by_iri}
    for p in model.properties:
        for other in p.inverses & by_iri.keys():
            inverses[p.iri].add(other)
            inverses[other].add(p.iri)
    symmetric = {
        iri
        for iri in by_iri
        if any(
            by_iri[m].symmetric or inverses[m] & group(iri)
            for m in group(iri)
            if m in by_iri
        )
    }

    def fixpoint(
        own: dict[str, tuple[set[frozenset[str]], set[frozenset[str]]]],
    ) -> dict[str, tuple[set[frozenset[str]], set[frozenset[str]]]]:
        ends = {iri: (set(d), set(r)) for iri, (d, r) in own.items()}
        changed = True
        while changed:
            changed = False
            for iri in by_iri:
                domain, range_ = set(ends[iri][0]), set(ends[iri][1])
                for above in up[iri] & by_iri.keys():
                    domain |= ends[above][0]
                    range_ |= ends[above][1]
                for inverse in inverses[iri]:
                    if inverse not in group(iri):
                        domain |= ends[inverse][1]
                        range_ |= ends[inverse][0]
                if (domain, range_) != ends[iri]:
                    ends[iri] = (domain, range_)
                    changed = True
        return ends

    own = {p.iri: (set(p.domains), set(p.ranges)) for p in model.properties}
    ends = fixpoint(own)
    for iri in symmetric:
        domain, range_ = ends[iri]
        if bool(domain) != bool(range_):
            own[iri] = (domain or range_, range_ or domain)
    ends = fixpoint(own)
    return (
        {iri: (frozenset(d), frozenset(r)) for iri, (d, r) in ends.items()},
        symmetric,
    )


def _with_inferences(model: OntologyModel) -> OntologyModel:
    """``model`` plus statements the closure implies: transitive
    subsumptions, both directions of each equivalence and inverse, and on
    every property the domains and ranges in effect (its super-properties',
    its inverses' and, symmetric, its other end's)."""
    class_same = _both_ways(model.classes)
    class_up = _closed({c.iri: set(c.parents) for c in model.classes}, class_same)
    classes = tuple(
        c.model_copy(
            update={
                "parents": frozenset(class_up[c.iri]),
                "equivalents": frozenset(class_same[c.iri] - {c.iri}),
            }
        )
        for c in model.classes
    )
    prop_same = _both_ways(model.properties)
    prop_up = _closed({p.iri: set(p.parents) for p in model.properties}, prop_same)
    inverses: dict[str, set[str]] = {p.iri: set(p.inverses) for p in model.properties}
    for p in model.properties:
        for other in p.inverses:
            inverses.setdefault(other, set()).add(p.iri)
    ends, _ = _property_closure(model)
    properties = tuple(
        p.model_copy(
            update={
                "parents": frozenset(prop_up[p.iri]),
                "equivalents": frozenset(prop_same[p.iri] - {p.iri}),
                "inverses": frozenset(inverses[p.iri]),
                "domains": ends[p.iri][0],
                "ranges": ends[p.iri][1],
            }
        )
        for p in model.properties
    )
    return OntologyModel(
        classes=classes, properties=properties, prefixes=model.prefixes
    )


# More examples than the default: a symmetric property with one end and
# something inheriting from it is rare among them.
@settings(max_examples=300)
@given(model=models(), options=_OPTIONS)
def test_stating_the_closure_changes_nothing(
    model: OntologyModel, options: MappingOptions
) -> None:
    once = _with_inferences(model)
    assert _outcome(once, options) == _outcome(model, options)
    assert _outcome(_with_inferences(once), options) == _outcome(model, options)


@given(model=models(), options=_OPTIONS)
def test_result_is_a_valid_schema_with_unique_labels(
    model: OntologyModel, options: MappingOptions
) -> None:
    outcome = _outcome(model, options)
    if outcome[0] == "error":
        return
    document = outcome[1]
    schema = ExtractionSchema.from_items(
        document.entity_types,
        document.relationship_types,
        document.relationship_patterns,
        mode="strict",
    )
    for specs in (schema.entity_types, schema.relationship_types):
        keys = [schema_key(spec.label) for spec in specs]
        assert len(keys) == len(set(keys))
        assert all(spec.iri for spec in specs)


@given(model=models(), options=_OPTIONS)
def test_colliding_labels_raise_unless_qualified(
    model: OntologyModel, options: MappingOptions
) -> None:
    unqualified = options.model_copy(update={"on_collision": "error"})
    outcome = _outcome(model, unqualified)
    if outcome[0] == "error":
        return
    # Every label was the term's own: qualifying changes nothing.
    assert (
        _outcome(model, options.model_copy(update={"on_collision": "qualify"}))
        == outcome
    )


@given(
    model=models(),
    roots=st.sets(st.integers(0, 6), min_size=1, max_size=3),
    excluded=st.sets(st.integers(0, 6), max_size=2),
    depth=st.sampled_from([None, 0, 1, 2]),
)
def test_narrowing_never_adds_a_type(
    model: OntologyModel, roots: set[int], excluded: set[int], depth: int | None
) -> None:
    iris = [c.iri for c in model.classes]

    def pick(indices: Iterable[int]) -> list[str]:
        return sorted({iris[i % len(iris)] for i in indices})

    base = MappingOptions(on_collision="qualify", classes=pick(roots))
    narrower = [
        base.model_copy(update={"exclude_classes": tuple(pick(excluded))}),
        base.model_copy(update={"max_depth": depth}),
        base.model_copy(update={"classes": tuple(pick(roots)[:1])}),
    ]
    wide = _outcome(model, base)
    if wide[0] == "error":
        return
    wide_iris = {spec.iri for spec in wide[1].entity_types}
    for options in narrower:
        narrow = _outcome(model, options)
        if narrow[0] == "ok":
            assert {spec.iri for spec in narrow[1].entity_types} <= wide_iris


@pytest.mark.parametrize("symmetric", [False, True])
@settings(max_examples=300)
@given(data=st.data(), options=_OPTIONS)
def test_pattern_ends_fit_the_domain_and_range_in_effect(
    symmetric: bool, data: st.DataObject, options: MappingOptions
) -> None:
    model = data.draw(models(equivalences=False, symmetric=symmetric))
    outcome = _outcome(model, options)
    if outcome[0] == "error":
        return
    document, report = outcome[1], outcome[2]
    up = _closed({c.iri: set(c.parents) for c in model.classes}, {})
    ends, symmetric_iris = _property_closure(model)

    def at_or_below(child: str, ancestor: str) -> bool:
        return child == ancestor or ancestor in up.get(child, set())

    iri_of = {spec.label: spec.iri for spec in document.entity_types}
    prop_of = {spec.label: spec.iri for spec in document.relationship_types}
    lifted = {(note.iri, note.detail.split()[0]) for note in report.lifted}

    def fits(prop: str, side: str, end: str) -> bool:
        unions = ends[prop][0 if side == "domain" else 1]
        if (prop, side) in lifted:
            # Lifted: an ancestor of a class each statement allows.
            return all(any(at_or_below(c, end) for c in union) for union in unions)
        return all(any(at_or_below(end, c) for c in union) for union in unions)

    for pattern in document.relationship_patterns:
        prop = prop_of[pattern.relation]
        for side, end in (
            ("domain", iri_of[pattern.source_type]),
            ("range", iri_of[pattern.target_type]),
        ):
            if prop in symmetric_iris:
                # Either end of a symmetric property may come from either.
                assert fits(prop, "domain", end) or fits(prop, "range", end)
            else:
                assert fits(prop, side, end)
