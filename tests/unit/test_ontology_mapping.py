# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ontology mapping (``adapters.schema_sources.ontology_mapping``): an
``OntologyModel`` to a ``SchemaDocument``, with no RDF library involved."""

from __future__ import annotations

import re
import subprocess
import sys

import pytest
from pydantic import ValidationError

from tests.fixtures.ontology_models import (
    EX,
    PROCURE_GOLDEN,
    ends,
    labels,
    procure_model,
)
from unified_kg_rag.adapters.schema_sources.ontology_mapping import (
    MappingOptions,
    OntologyMapping,
    map_ontology,
    truncate,
)
from unified_kg_rag.adapters.schema_sources.ontology_model import (
    RDFS_COMMENT,
    RDFS_LABEL,
    SKOS_DEFINITION,
    SKOS_PREF_LABEL,
    OntologyClass,
    OntologyModel,
    OntologyProperty,
)
from unified_kg_rag.adapters.source_files.rdf_terms import (
    OWL,
    Text,
    best_text,
    local_name,
    preserve_case,
    upper_snake,
)
from unified_kg_rag.domain.models.schema import ExtractionSchema
from unified_kg_rag.shared.exceptions import SchemaSourceError

pytestmark = pytest.mark.unit

GEO = "https://example.org/geo#"


def _map(model: OntologyModel | None = None, **options: object) -> OntologyMapping:
    return map_ontology(
        model or procure_model(), MappingOptions.model_validate(options)
    )


def _labels(mapping: OntologyMapping) -> tuple[list[str], list[str]]:
    document = mapping.document
    return (
        [spec.label for spec in document.entity_types or ()],
        [spec.label for spec in document.relationship_types],
    )


def _patterns(mapping: OntologyMapping) -> list[str]:
    return [str(pattern) for pattern in mapping.document.relationship_patterns]


def _cls(name: str, *parents: str, ns: str = EX, **fields: object) -> OntologyClass:
    return OntologyClass(
        iri=ns + name,
        parents=frozenset(ns + parent for parent in parents),
        **fields,
    )


def _prop(name: str, ns: str = EX, **fields: object) -> OntologyProperty:
    return OntologyProperty(iri=ns + name, **fields)


class TestWorkedExample:
    def test_default_mapping_is_the_golden_schema(self) -> None:
        mapping = _map()
        assert mapping.document == PROCURE_GOLDEN
        report = mapping.report
        assert (report.entity_types, report.relationship_types, report.patterns) == (
            4,
            5,
            17,
        )
        assert [note.iri for note in report.skipped] == [EX + "name"]
        assert report.dropped == report.lifted == ()

    def test_golden_schema_is_a_valid_extraction_schema(self) -> None:
        document = PROCURE_GOLDEN
        schema = ExtractionSchema.from_items(
            document.entity_types or (),
            document.relationship_types,
            document.relationship_patterns,
            mode="strict",
        )
        assert len(schema.relationship_patterns) == 17

    def test_root_class_drops_properties_with_no_selected_end(self) -> None:
        mapping = _map(classes=["ex:Organization"])
        assert _labels(mapping) == (
            ["ORGANIZATION", "BUYER", "VENDOR"],
            ["SUPPLIED_BY", "TRADES_WITH", "SUPPLIES"],
        )
        assert {(note.iri, note.detail) for note in mapping.report.dropped} == {
            (EX + "employs", "no selected class fits its range"),
            (EX + "worksFor", "no selected class fits its domain"),
        }

    def test_depth_zero_lifts_patterns_to_the_root(self) -> None:
        mapping = _map(classes=["ex:Organization"], max_depth=0)
        assert _labels(mapping)[0] == ["ORGANIZATION"]
        assert "ORGANIZATION -SUPPLIES-> ORGANIZATION" in _patterns(mapping)
        assert {note.iri for note in mapping.report.lifted} == {
            EX + "supplies",
            EX + "suppliedBy",
        }
        # The hint names only selected types.
        assert mapping.document.entity_types[0].description == ""

    def test_korean_labels(self) -> None:
        mapping = _map(label_source="label", languages=["ko"])
        entities, relations = _labels(mapping)
        assert sorted(entities) == sorted(["조직", "구매자", "공급업체", "사람"])
        assert "공급한다" in relations
        # No Korean label: the English one, then the local name.
        assert "SUPPLIED_BY" in relations
        assert "공급업체 -공급한다-> 구매자" in _patterns(mapping)
        vendor = next(
            spec
            for spec in mapping.document.entity_types or ()
            if spec.label == "공급업체"
        )
        assert vendor.description.endswith("Kind of 조직.")

    def test_patterns_none_keeps_the_types(self) -> None:
        mapping = _map(patterns="none")
        assert _labels(mapping) == _labels(_map())
        assert mapping.document.relationship_patterns == []
        assert mapping.report.patterns == 0

    def test_hints_off(self) -> None:
        mapping = _map(hierarchy_hints=False)
        descriptions = [
            spec.description
            for spec in [
                *mapping.document.entity_types,
                *mapping.document.relationship_types,
            ]
        ]
        assert (
            descriptions == ["", "", "An organization that sells goods.", ""] + [""] * 5
        )


class TestPatterns:
    def _model(self, *properties: OntologyProperty, classes=None) -> OntologyModel:
        return OntologyModel(
            classes=classes
            or (
                _cls("Org"),
                _cls("Vendor", "Org"),
                _cls("Buyer", "Org"),
                _cls("Both", "Vendor", "Buyer"),
                _cls("Person"),
            ),
            properties=properties,
            prefixes={"ex": EX},
        )

    def test_domain_statements_intersect(self) -> None:
        model = self._model(
            _prop(
                "brokers",
                domains=ends("Vendor") | ends("Buyer"),
                ranges=ends("Person"),
            )
        )
        assert _patterns(_map(model)) == ["BOTH -BROKERS-> PERSON"]

    def test_union_allows_either(self) -> None:
        model = self._model(
            _prop("pays", domains=ends("Vendor", "Buyer"), ranges=ends("Person"))
        )
        assert _patterns(_map(model)) == [
            "BUYER -PAYS-> PERSON",
            "BOTH -PAYS-> PERSON",
            "VENDOR -PAYS-> PERSON",
        ]

    def test_domain_includes_with_an_unselected_class(self) -> None:
        # schema:domainIncludes Person, Other: one union; Other is not selected.
        model = self._model(
            _prop("owns", domains=ends("Person", "Other"), ranges=ends("Org"))
        )
        assert _patterns(_map(model, exclude_classes=["ex:Vendor", "ex:Buyer"])) == [
            "PERSON -OWNS-> ORG"
        ]

    def test_lift_to_the_nearest_selected_ancestor(self) -> None:
        model = self._model(
            _prop("audits", domains=ends("Both"), ranges=ends("Person"))
        )
        mapping = _map(model, exclude_classes=["ex:Both"])
        # Both is gone: lifted to Vendor and Buyer, not to Org above them.
        assert _patterns(mapping) == [
            "BUYER -AUDITS-> PERSON",
            "VENDOR -AUDITS-> PERSON",
        ]
        assert mapping.report.lifted[0].detail.startswith("domain lifted to")

    def test_no_ancestor_drops_the_property(self) -> None:
        model = self._model(_prop("hires", domains=ends("Org"), ranges=ends("Person")))
        mapping = _map(model, classes=["ex:Person"])
        assert _labels(mapping) == (["PERSON"], [])
        assert mapping.report.dropped[0].iri == EX + "hires"

    def test_one_end_unrestricted_matches_every_class(self) -> None:
        model = self._model(_prop("knows", domains=ends("Person")))
        assert _patterns(
            _map(model, classes=["ex:Person", "ex:Buyer"], exclude_classes=["ex:Both"])
        ) == [
            "PERSON -KNOWS-> BUYER",
            "PERSON -KNOWS-> PERSON",
        ]

    def test_no_domain_nor_range_gets_no_pattern(self) -> None:
        model = self._model(
            _prop("relatedTo"),
            _prop("anything", domains=frozenset({frozenset({OWL + "Thing"})})),
        )
        mapping = _map(model)
        assert _labels(mapping)[1] == ["ANYTHING", "RELATED_TO"]
        assert _patterns(mapping) == []
        assert mapping.report.unrestricted == (EX + "anything", EX + "relatedTo")

    def test_class_equivalent_to_thing_is_unrestricted(self) -> None:
        model = self._model(
            _prop("tags", domains=ends("Top"), ranges=ends("Person")),
            _prop("undeclaredInverse", inverses=frozenset({EX + "nowhere"})),
            classes=(
                _cls("Top", equivalents=frozenset({OWL + "Thing"})),
                _cls("Person"),
            ),
        )
        mapping = _map(model)
        assert _patterns(mapping) == ["PERSON -TAGS-> PERSON", "TOP -TAGS-> PERSON"]
        assert mapping.report.unrestricted == (EX + "undeclaredInverse",)

    def test_symmetric_squares_and_fills_a_missing_end(self) -> None:
        model = self._model(_prop("meets", symmetric=True, domains=ends("Person")))
        assert _patterns(_map(model)) == ["PERSON -MEETS-> PERSON"]

    def test_symmetric_joins_both_ends(self) -> None:
        model = self._model(
            _prop(
                "partners", symmetric=True, domains=ends("Person"), ranges=ends("Buyer")
            ),
        )
        assert _patterns(
            _map(model, classes=["ex:Person", "ex:Buyer"], exclude_classes=["ex:Both"])
        ) == [
            "BUYER -PARTNERS-> BUYER",
            "BUYER -PARTNERS-> PERSON",
            "PERSON -PARTNERS-> BUYER",
            "PERSON -PARTNERS-> PERSON",
        ]

    def test_self_inverse_is_symmetric(self) -> None:
        model = self._model(
            _prop(
                "marries",
                inverses=frozenset({EX + "marries"}),
                domains=ends("Person"),
            )
        )
        assert _patterns(_map(model)) == ["PERSON -MARRIES-> PERSON"]

    @pytest.mark.parametrize("range_stated", [False, True])
    def test_a_filled_symmetric_end_reaches_sub_properties_and_inverses(
        self, range_stated: bool
    ) -> None:
        # meets ≡ Person × Person whether its range is stated or filled, and
        # so is what inherits from it.
        model = self._model(
            _prop(
                "meets",
                symmetric=True,
                domains=ends("Person"),
                ranges=ends("Person") if range_stated else frozenset(),
            ),
            _prop("greets", parents=frozenset({EX + "meets"})),
            _prop("metBy", inverses=frozenset({EX + "meets"})),
        )
        assert _patterns(_map(model)) == [
            "PERSON -MEETS-> PERSON",
            "PERSON -GREETS-> PERSON",
            "PERSON -MET_BY-> PERSON",
        ]

    def test_a_symmetric_end_is_filled_only_when_the_closure_leaves_it_empty(
        self,
    ) -> None:
        # agrees' range is its inverse's domain, which that one inherits
        # from relates: Org. Filling it with Person before that arrives
        # would leave no class both ends allow.
        model = self._model(
            _prop("agrees", symmetric=True, domains=ends("Person")),
            _prop(
                "zAgreedBy",
                parents=frozenset({EX + "zRelates"}),
                inverses=frozenset({EX + "agrees"}),
            ),
            _prop("zRelates", domains=ends("Org")),
        )
        mapping = _map(model, properties=["ex:agrees"])
        assert mapping.report.dropped == ()
        assert len(_patterns(mapping)) == 25  # (Person + 4 orgs) squared

    def test_sub_property_inherits_through_an_unselected_super(self) -> None:
        model = self._model(
            _prop("relates", domains=ends("Org"), ranges=ends("Person")),
            _prop("middle", parents=frozenset({EX + "relates"})),
            _prop("leads", parents=frozenset({EX + "middle"}), domains=ends("Vendor")),
        )
        mapping = _map(model, properties=["ex:leads"])
        assert _patterns(mapping) == ["BOTH -LEADS-> PERSON", "VENDOR -LEADS-> PERSON"]

    def test_inverse_of_a_super_property(self) -> None:
        # leads ⊑ relates, owner inverseOf relates: leads' domain is owner's range.
        model = self._model(
            _prop("owner", domains=ends("Person"), ranges=ends("Buyer")),
            _prop("relates", inverses=frozenset({EX + "owner"})),
            _prop("leads", parents=frozenset({EX + "relates"})),
        )
        mapping = _map(model, properties=["ex:leads"])
        assert _patterns(mapping) == ["BUYER -LEADS-> PERSON", "BOTH -LEADS-> PERSON"]

    def test_pattern_cap_names_the_largest_properties(self) -> None:
        with pytest.raises(SchemaSourceError) as raised:
            _map(max_patterns=16)
        message = str(raised.value)
        assert "17 patterns" in message and "max_patterns (16)" in message
        assert f"{EX}tradesWith (9)" in message.split("most come from")[1][:60]

    def test_pattern_cap_does_not_apply_without_patterns(self) -> None:
        assert _map(max_patterns=1, patterns="none").report.patterns == 0

    def test_an_untyped_property_is_a_relationship_type(self) -> None:
        # An rdf:Property nothing types further (schema.org's properties).
        model = self._model(
            _prop(
                "knows", kind="untyped", domains=ends("Person"), ranges=ends("Person")
            ),
            _prop("age", kind="datatype", domains=ends("Person")),
            _prop("note", kind="annotation"),
        )
        mapping = _map(model, classes=["ex:Person"])
        assert _labels(mapping) == (["PERSON"], ["KNOWS"])
        assert _patterns(mapping) == ["PERSON -KNOWS-> PERSON"]
        assert [(n.iri, n.detail) for n in mapping.report.skipped] == [
            (EX + "age", "datatype property"),
            (EX + "note", "annotation property"),
        ]

    def test_every_domain_statement_pushes_down(self) -> None:
        # Both ⊑ Vendor ⊓ Buyer: below each statement; Vendor alone is not.
        model = self._model(
            _prop("brokers", domains=ends("Vendor") | ends("Buyer")),
        )
        mapping = _map(model, exclude_classes=["ex:Person"])
        assert _patterns(mapping) == [
            "BOTH -BROKERS-> ORG",
            "BOTH -BROKERS-> BUYER",
            "BOTH -BROKERS-> BOTH",
            "BOTH -BROKERS-> VENDOR",
        ]
        assert mapping.report.lifted == ()

    def test_a_lift_needs_an_ancestor_of_every_statement(self) -> None:
        # Person and Both are both gone; Org is above Both but not Person.
        model = self._model(
            _prop(
                "audits",
                domains=ends("Both") | ends("Person"),
                ranges=ends("Org"),
            )
        )
        mapping = _map(model, exclude_classes=["ex:Both", "ex:Person"])
        assert _patterns(mapping) == []
        assert mapping.report.dropped[0].detail == ("no selected class fits its domain")

    def test_a_lift_is_reported_for_each_lifted_end(self) -> None:
        model = self._model(
            _prop("audits", domains=ends("Person"), ranges=ends("Both"))
        )
        mapping = _map(model, exclude_classes=["ex:Both"])
        assert [(n.iri, n.detail) for n in mapping.report.lifted] == [
            (EX + "audits", f"range lifted to {EX}Buyer, {EX}Vendor")
        ]

    def test_an_inverse_gives_its_other_end(self) -> None:
        # owner: Person -> Buyer; ownedBy is its inverse with nothing stated.
        model = self._model(
            _prop("owner", domains=ends("Person"), ranges=ends("Buyer")),
            _prop("ownedBy", inverses=frozenset({EX + "owner"})),
        )
        mapping = _map(model, properties=["ex:ownedBy"])
        assert _patterns(mapping) == [
            "BUYER -OWNED_BY-> PERSON",
            "BOTH -OWNED_BY-> PERSON",
        ]

    def test_an_inverse_stated_on_the_other_side_counts_both_ways(self) -> None:
        # Only owner says it is ownedBy's inverse: ownedBy still gets its ends.
        model = self._model(
            _prop(
                "owner",
                domains=ends("Person"),
                ranges=ends("Buyer"),
                inverses=frozenset({EX + "ownedBy"}),
            ),
            _prop("ownedBy", domains=ends("Org")),
        )
        mapping = _map(model, properties=["ex:ownedBy"])
        assert _patterns(mapping) == [
            "BUYER -OWNED_BY-> PERSON",
            "BOTH -OWNED_BY-> PERSON",
        ]

    def test_one_stated_end_leaves_the_other_open(self) -> None:
        # A domain only: every selected class is a range.
        model = self._model(_prop("owns", domains=ends("Person")))
        mapping = _map(model, classes=["ex:Person", "ex:Org"], max_depth=0)
        assert _patterns(mapping) == ["PERSON -OWNS-> ORG", "PERSON -OWNS-> PERSON"]

    def test_a_symmetric_end_filled_from_the_closure_reaches_inheritors(
        self,
    ) -> None:
        # meets inherits its only domain from relates; the filled range
        # reaches greets, below meets, too.
        model = self._model(
            _prop("relates", domains=ends("Person")),
            _prop("meets", symmetric=True, parents=frozenset({EX + "relates"})),
            _prop("greets", parents=frozenset({EX + "meets"})),
        )
        mapping = _map(model, properties=["ex:greets"])
        assert _patterns(mapping) == ["PERSON -GREETS-> PERSON"]

    def test_a_filled_symmetric_end_reaches_an_inverse_of_the_other_end(
        self,
    ) -> None:
        # meets has a range only; its filled domain is metBy's range.
        model = self._model(
            _prop("meets", symmetric=True, ranges=ends("Person")),
            _prop("metBy", inverses=frozenset({EX + "meets"})),
        )
        mapping = _map(model, properties=["ex:metBy"])
        assert _patterns(mapping) == ["PERSON -MET_BY-> PERSON"]

    def test_an_end_equivalent_to_a_selected_class_is_one_type(self) -> None:
        # Firm ≡ Org: a domain of Firm allows Org and what is below it.
        model = self._model(
            _prop("hires", domains=ends("Firm"), ranges=ends("Person")),
            classes=(
                _cls("Org", equivalents=frozenset({EX + "Firm"})),
                _cls("Firm"),
                _cls("Vendor", "Org"),
                _cls("Person"),
            ),
        )
        assert _patterns(_map(model)) == [
            "FIRM -HIRES-> PERSON",
            "VENDOR -HIRES-> PERSON",
        ]


class TestSelection:
    def test_equivalent_classes_are_one_type(self) -> None:
        model = OntologyModel(
            classes=(
                _cls("Firm", equivalents=frozenset({EX + "Company"})),
                _cls("Company", annotations=labels("Company")),
                _cls("Shop", "Firm"),
            )
        )
        mapping = _map(model)
        # Named by the smallest IRI, labelled from either member's label.
        assert [(s.label, s.iri) for s in mapping.document.entity_types or ()] == [
            ("COMPANY", EX + "Company"),
            ("SHOP", EX + "Shop"),
        ]
        assert mapping.report.merged == ((EX + "Company", EX + "Firm"),)

    def test_subclass_cycle_collapses(self) -> None:
        model = OntologyModel(classes=(_cls("A", "B"), _cls("B", "A"), _cls("C", "B")))
        mapping = _map(model)
        assert _labels(mapping)[0] == ["A", "C"]

    def test_representative_is_the_smallest_selected_iri(self) -> None:
        model = OntologyModel(
            classes=(
                _cls("Org", ns=GEO, equivalents=frozenset({EX + "Org"})),
                _cls("Org"),
            )
        )
        mapping = _map(model, namespaces=[EX])
        assert mapping.document.entity_types[0].iri == EX + "Org"

    def test_equivalent_properties_are_one_type(self) -> None:
        model = OntologyModel(
            classes=(_cls("A"),),
            properties=(
                _prop("likes", equivalents=frozenset({EX + "fancies"})),
                _prop("fancies", domains=ends("A")),
            ),
        )
        mapping = _map(model)
        assert _labels(mapping)[1] == ["FANCIES"]
        assert _patterns(mapping) == ["A -FANCIES-> A"]

    def test_deprecated_left_out_but_closed_over(self) -> None:
        model = OntologyModel(
            classes=(
                _cls("Root"),
                _cls("Old", "Root", deprecated=True),
                _cls("Leaf", "Old"),
            ),
            properties=(_prop("gone", deprecated=True),),
        )
        mapping = _map(model)
        assert _labels(mapping) == (["ROOT", "LEAF"], [])
        assert mapping.document.entity_types[1].description == "Kind of ROOT."
        assert mapping.report.deprecated == (EX + "Old", EX + "gone")
        assert _labels(_map(model, include_deprecated=True)) == (
            ["ROOT", "OLD", "LEAF"],
            ["GONE"],
        )
        # Leaf is below Root through Old: not a root of max_depth.
        assert _labels(_map(model, max_depth=0)) == (["ROOT"], [])

    def test_max_depth_counts_from_the_top_without_roots(self) -> None:
        assert _labels(_map(max_depth=0))[0] == ["ORGANIZATION", "PERSON"]

    def test_exclude_removes_the_subtree(self) -> None:
        mapping = _map(
            exclude_classes=["ex:Organization"], exclude_properties=["ex:tradesWith"]
        )
        assert _labels(mapping) == (["PERSON"], [])

    def test_exclude_properties_keeps_the_inverse(self) -> None:
        relations = _labels(_map(exclude_properties=[EX + "suppliedBy"]))[1]
        assert "SUPPLIED_BY" not in relations and "SUPPLIES" in relations

    def test_namespaces_filter(self) -> None:
        model = OntologyModel(
            classes=(_cls("Org"), _cls("Place", ns=GEO)), prefixes={"geo": GEO}
        )
        assert _labels(_map(model, namespaces=["geo:"]))[0] == ["PLACE"]

    def test_core_and_descriptive_namespaces_are_left_out(self) -> None:
        model = OntologyModel(
            classes=(
                _cls("Org"),
                OntologyClass(iri=OWL + "Thing"),
                OntologyClass(iri="http://www.w3.org/2004/02/skos/core#Concept"),
            )
        )
        assert _labels(_map(model))[0] == ["ORG"]

    @pytest.mark.parametrize(
        ("options", "match"),
        [
            (
                {"classes": ["ex:Organisation"]},
                r"classes: 'ex:Organisation' names no class",
            ),
            (
                {"exclude_classes": ["ex:tradesWith"]},
                r"exclude_classes: .* names no class",
            ),
            ({"properties": ["ex:Vendor"]}, r"properties: .* names no property"),
            ({"exclude_properties": ["ex:nope"]}, "exclude_properties"),
            (
                {"labels": {"ex:Nope": "X"}},
                r"labels: 'ex:Nope' names no class or property",
            ),
            (
                {"namespaces": ["https://example.org/other#"]},
                "holds no class or property",
            ),
            (
                {"classes": ["foo:Vendor"]},
                r"neither an IRI nor a CURIE of a bound prefix",
            ),
            ({"classes": ["Vendor"]}, "neither an IRI nor a CURIE"),
            ({"description_properties": ["dc:description"]}, "description_properties"),
            (
                {"description_properties": ["rdfs:comment", "ex:remark"]},
                r"description_properties: 'ex:remark' is neither a property",
            ),
        ],
    )
    def test_terms_that_name_nothing_fail(self, options: dict, match: str) -> None:
        with pytest.raises(SchemaSourceError, match=match):
            _map(**options)

    def test_conflicting_prefix_binding_fails_unless_set(self) -> None:
        model = OntologyModel(
            classes=(OntologyClass(iri="http://schema.org/Thing2"),),
            prefixes={"schema": "http://schema.org/"},
        )
        with pytest.raises(SchemaSourceError, match="prefix 'schema' is bound to"):
            _map(model, classes=["schema:Thing2"])
        mapping = _map(
            model, classes=["schema:Thing2"], prefixes={"schema": "http://schema.org/"}
        )
        assert _labels(mapping)[0] == ["THING2"]

    def test_nothing_selected_fails(self) -> None:
        with pytest.raises(
            SchemaSourceError, match="no class of the ontology is selected"
        ):
            _map(exclude_classes=["ex:Organization", "ex:Person"])

    def test_entity_cap_fails(self) -> None:
        with pytest.raises(
            SchemaSourceError, match=r"4 classes .* max_entity_types \(3\)"
        ):
            _map(max_entity_types=3)

    def test_relationship_cap_fails(self) -> None:
        with pytest.raises(
            SchemaSourceError, match=r"5 properties .* max_relationship_types \(4\)"
        ):
            _map(max_relationship_types=4)

    def test_each_cap_allows_exactly_its_value(self) -> None:
        # The worked example: 4 entity types, 5 relationship types, 17 patterns.
        report = _map(
            max_entity_types=4, max_relationship_types=5, max_patterns=17
        ).report
        assert (report.entity_types, report.relationship_types, report.patterns) == (
            4,
            5,
            17,
        )

    def _properties(self) -> OntologyModel:
        return OntologyModel(
            classes=(_cls("Org"),),
            properties=(
                _prop("rel", domains=ends("Org"), ranges=ends("Org")),
                _prop("sub", parents=frozenset({EX + "rel"})),
                _prop("subSub", parents=frozenset({EX + "sub"})),
                _prop("other", domains=ends("Org")),
            ),
            prefixes={"ex": EX},
        )

    def test_properties_selects_the_subtree(self) -> None:
        mapping = _map(self._properties(), properties=["ex:rel"])
        assert _labels(mapping)[1] == ["REL", "SUB", "SUB_SUB"]

    def test_exclude_properties_removes_the_subtree(self) -> None:
        mapping = _map(self._properties(), exclude_properties=["ex:sub"])
        assert _labels(mapping)[1] == ["OTHER", "REL"]

    def test_max_depth_counts_from_the_given_roots(self) -> None:
        model = OntologyModel(
            classes=(_cls("A"), _cls("B", "A"), _cls("C", "B"), _cls("D", "C")),
            prefixes={"ex": EX},
        )
        assert _labels(_map(model, classes=["ex:B"], max_depth=1))[0] == ["B", "C"]
        assert _labels(_map(model, max_depth=2))[0] == ["A", "B", "C"]

    def test_a_group_reports_under_its_smallest_iri(self) -> None:
        # zeta ≡ alpha: listed where alpha sorts, not where zeta does.
        model = OntologyModel(
            classes=(_cls("A"),),
            properties=(
                _prop("zeta", equivalents=frozenset({EX + "alpha"})),
                _prop("alpha"),
                _prop("mid"),
            ),
        )
        mapping = _map(model)
        assert mapping.report.unrestricted == (EX + "alpha", EX + "mid")
        assert mapping.report.merged == ((EX + "alpha", EX + "zeta"),)


class TestLabels:
    @pytest.mark.parametrize(
        ("text", "label"),
        [
            ("supplyChain", "SUPPLY_CHAIN"),
            ("HTTPServer", "HTTP_SERVER"),
            ("Level2Item", "LEVEL2_ITEM"),
            ("ISO9001Cert", "ISO9001_CERT"),
            ("B2B", "B2B"),
            ("works for", "WORKS_FOR"),
            ("part-of__whole", "PART_OF_WHOLE"),
            ("공급 업체", "공급_업체"),
            ("공급업체", "공급업체"),
            ("AI반도체 Vendor", "AI반도체_VENDOR"),
            ("ＶＥＮＤＯＲ", "VENDOR"),
            ("Café", "CAFÉ"),
            ("(n/a)", "N_A"),
            ("!!!", ""),
        ],
    )
    def test_upper_snake(self, text: str, label: str) -> None:
        assert upper_snake(text) == label

    @pytest.mark.parametrize(
        ("text", "label"),
        [
            ("Vendor (EU)/Asia", "Vendor EU Asia"),
            ("  works   for ", "works for"),
            ("part-of_whole", "part-of_whole"),
            ("ｓｕｐｐｌｉｅｓ", "supplies"),
            ("공급 업체", "공급 업체"),
        ],
    )
    def test_preserve(self, text: str, label: str) -> None:
        assert preserve_case(text) == label

    @pytest.mark.parametrize(
        ("iri", "name"),
        [
            ("https://example.org/a#Vendor", "Vendor"),
            ("https://example.org/a/Vendor", "Vendor"),
            ("urn:example:Vendor", "Vendor"),
            ("https://example.org/a/%EA%B3%B5%EA%B8%89", "공급"),
            ("Vendor", "Vendor"),
        ],
    )
    def test_local_name(self, iri: str, name: str) -> None:
        assert local_name(iri) == name

    def test_label_case_preserve(self) -> None:
        entities, relations = _labels(_map(label_case="preserve"))
        assert entities == ["Organization", "Buyer", "Vendor", "Person"]
        assert "supplies" in relations and "suppliedBy" in relations

    def test_label_source_local_name(self) -> None:
        model = OntologyModel(
            classes=(_cls("supplyChainNode", annotations=labels("Node of the chain")),)
        )
        assert _labels(_map(model))[0] == ["NODE_OF_THE_CHAIN"]
        assert _labels(_map(model, label_source="local_name"))[0] == [
            "SUPPLY_CHAIN_NODE"
        ]

    def test_auto_prefers_pref_label_and_label_ignores_it(self) -> None:
        model = OntologyModel(
            classes=(
                _cls(
                    "Org",
                    annotations={
                        RDFS_LABEL: (Text(value="Organisation", language="en"),),
                        SKOS_PREF_LABEL: (Text(value="Company", language="en"),),
                    },
                ),
            )
        )
        assert _labels(_map(model))[0] == ["COMPANY"]
        assert _labels(_map(model, label_source="label"))[0] == ["ORGANISATION"]

    def test_language_before_property(self) -> None:
        # A Korean prefLabel loses to an English rdfs:label under [en].
        model = OntologyModel(
            classes=(
                _cls(
                    "Org",
                    annotations={
                        RDFS_LABEL: (Text(value="Organisation", language="en"),),
                        SKOS_PREF_LABEL: (Text(value="조직", language="ko"),),
                    },
                ),
            )
        )
        assert _labels(_map(model))[0] == ["ORGANISATION"]

    @pytest.mark.parametrize(
        ("texts", "languages", "best"),
        [
            ([("b", "en-US"), ("a", "en")], ["en"], "a"),
            ([("b", "en-US"), ("a", "de")], ["en"], "b"),
            ([("b", "EN-us"), ("a", "")], ["en-us"], "b"),
            ([("b", "de"), ("a", "")], ["en", ""], "a"),
            ([("b", ""), ("a", "ko")], ["ko", ""], "a"),
            ([("z", "fr"), ("y", "de")], ["en"], "y"),
            ([("b", "fr"), ("a", "")], ["en"], "a"),
            ([("  ", "en"), ("a", "de")], ["en"], "a"),
            ([("b", "en"), ("a", "en")], ["en"], "a"),
        ],
    )
    def test_language_chain(
        self, texts: list[tuple[str, str]], languages: list[str], best: str
    ) -> None:
        values = {RDFS_LABEL: tuple(Text(value=v, language=lang) for v, lang in texts)}
        assert best_text(values, [RDFS_LABEL], languages) == best

    def test_untagged_comes_before_other_tags_when_not_listed(self) -> None:
        values = {RDFS_LABEL: (Text(value="de", language="de"), Text(value="none"))}
        assert best_text(values, [RDFS_LABEL], ["ko"]) == "none"

    def test_override_is_used_as_written(self) -> None:
        mapping = _map(labels={"ex:Vendor": "Supplier Co", EX + "supplies": "SELLS_TO"})
        entities, relations = _labels(mapping)
        assert "Supplier Co" in entities and "SELLS_TO" in relations
        assert "Supplier Co -SELLS_TO-> BUYER" in _patterns(mapping)

    def test_override_conflict_on_equivalents_fails(self) -> None:
        model = OntologyModel(
            classes=(_cls("A", equivalents=frozenset({EX + "B"})), _cls("B")),
            prefixes={"ex": EX},
        )
        with pytest.raises(SchemaSourceError, match="equivalent class terms"):
            _map(model, labels={"ex:A": "X", "ex:B": "Y"})

    def test_too_long_or_empty_label_fails_naming_the_iri(self) -> None:
        long_label = OntologyModel(classes=(_cls("A", annotations=labels("x" * 65)),))
        with pytest.raises(
            SchemaSourceError, match=rf"{EX}A has no usable label .*longer"
        ):
            _map(long_label)
        assert _labels(_map(long_label, labels={EX + "A": "A"}))[0] == ["A"]
        empty = OntologyModel(classes=(OntologyClass(iri="https://example.org/a#!!"),))
        with pytest.raises(SchemaSourceError, match="has no usable label .*empty"):
            _map(empty)

    def test_override_longer_than_the_limit_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            MappingOptions(labels={"ex:A": "x" * 65})

    def test_punctuation_only_label_falls_back_to_the_local_name(self) -> None:
        model = OntologyModel(classes=(_cls("Vendor", annotations=labels("???")),))
        assert _labels(_map(model))[0] == ["VENDOR"]


class TestCollisions:
    def _model(self, reverse: bool = False) -> OntologyModel:
        classes = [
            _cls("Bank", annotations=labels("Bank")),
            _cls("Bank", ns=GEO, annotations=labels("bank")),
            _cls("River", ns=GEO),
        ]
        return OntologyModel(
            classes=tuple(reversed(classes) if reverse else classes),
            prefixes={"ex": EX, "geo": GEO},
        )

    def test_collision_fails_listing_the_iris(self) -> None:
        with pytest.raises(SchemaSourceError) as raised:
            _map(self._model())
        message = str(raised.value)
        assert "entity type labels collide" in message
        assert EX + "Bank" in message and GEO + "Bank" in message
        assert "on_collision: qualify" in message

    @pytest.mark.parametrize("reverse", [False, True])
    def test_qualify_suffixes_every_member(self, reverse: bool) -> None:
        mapping = _map(self._model(reverse), on_collision="qualify")
        assert _labels(mapping)[0] == ["BANK_EX", "BANK_GEO", "RIVER"]
        assert [note.detail for note in mapping.report.qualified] == [
            "BANK_GEO",
            "BANK_EX",
        ]

    def test_unbound_namespace_gets_a_hash(self) -> None:
        model = OntologyModel(
            classes=(
                _cls("Bank"),
                OntologyClass(iri="https://other.example/x/Bank"),
            ),
            prefixes={"ex": EX},
        )
        entities = _labels(_map(model, on_collision="qualify"))[0]
        assert "BANK_EX" in entities
        (hashed,) = set(entities) - {"BANK_EX"}
        assert re.fullmatch("BANK_[0-9A-F]{6}", hashed)

    def test_the_options_prefixes_name_the_suffix_first(self) -> None:
        # The files bind "a" to EX; pinned as "ex" in the options, the
        # suffix no longer follows how the files are serialised.
        model = self._model().model_copy(update={"prefixes": {"a": EX, "geo": GEO}})
        assert _labels(_map(model, on_collision="qualify"))[0][:2] == [
            "BANK_A",
            "BANK_GEO",
        ]
        pinned = _map(model, on_collision="qualify", prefixes={"ex": EX})
        assert _labels(pinned)[0][:2] == ["BANK_EX", "BANK_GEO"]

    def test_labels_differing_in_case_only_collide(self) -> None:
        # label_case: preserve keeps "Vendor" and "vendor", one schema key.
        model = OntologyModel(
            classes=(
                _cls("V", annotations=labels("Vendor")),
                _cls("V", ns=GEO, annotations=labels("vendor")),
            ),
            prefixes={"ex": EX, "geo": GEO},
        )
        with pytest.raises(
            SchemaSourceError,
            match=f"entity type labels collide: 'vendor' names {GEO}V, {EX}V",
        ):
            _map(model, label_case="preserve")
        mapping = _map(model, label_case="preserve", on_collision="qualify")
        assert _labels(mapping)[0] == ["Vendor_EX", "vendor_GEO"]

    def test_relationship_labels_differing_in_case_only_collide(self) -> None:
        model = OntologyModel(
            classes=(_cls("A"),),
            properties=(
                _prop("rel", annotations=labels("Supplies")),
                _prop("rel", ns=GEO, annotations=labels("supplies")),
            ),
            prefixes={"ex": EX, "geo": GEO},
        )
        with pytest.raises(SchemaSourceError, match="relationship type labels collide"):
            _map(model, label_case="preserve")
        mapping = _map(model, label_case="preserve", on_collision="qualify")
        assert _labels(mapping)[1] == ["Supplies_EX", "supplies_GEO"]

    def test_the_smallest_of_several_prefixes_names_the_suffix(self) -> None:
        model = self._model().model_copy(
            update={"prefixes": {"zz": EX, "ex": EX, "geo": GEO}}
        )
        assert _labels(_map(model, on_collision="qualify"))[0][:2] == [
            "BANK_EX",
            "BANK_GEO",
        ]

    def test_a_well_known_prefix_names_the_suffix_before_the_files(self) -> None:
        schema_org = "https://schema.org/"
        model = OntologyModel(
            classes=(
                _cls("Bank"),
                OntologyClass(iri=schema_org + "Bank"),
            ),
            # The files bind schema.org as "s"; options.prefixes come first.
            prefixes={"ex": EX, "s": schema_org},
        )
        assert _labels(_map(model, on_collision="qualify"))[0] == [
            "BANK_EX",
            "BANK_SCHEMA",
        ]
        pinned = _map(model, on_collision="qualify", prefixes={"sdo": schema_org})
        assert _labels(pinned)[0] == ["BANK_EX", "BANK_SDO"]

    def test_the_empty_prefix_never_names_the_suffix(self) -> None:
        # Turtle's "@prefix : <...>" binds "": the namespace is unnamed.
        model = self._model().model_copy(update={"prefixes": {"": EX, "geo": GEO}})
        entities = _labels(_map(model, on_collision="qualify"))[0]
        assert "BANK_GEO" in entities
        (hashed,) = set(entities) - {"BANK_GEO", "RIVER"}
        assert re.fullmatch("BANK_[0-9A-F]{6}", hashed)

    def test_same_namespace_collision_survives_qualify(self) -> None:
        model = OntologyModel(classes=(_cls("Bank"), _cls("bank")), prefixes={"ex": EX})
        with pytest.raises(SchemaSourceError, match="still collide after qualifying"):
            _map(model, on_collision="qualify")

    def test_qualified_label_over_the_limit_fails(self) -> None:
        long = "x" * 62
        model = OntologyModel(
            classes=(
                _cls("A", annotations=labels(long)),
                _cls("A", ns=GEO, annotations=labels(long)),
            ),
            prefixes={"ex": EX, "geo": GEO},
        )
        with pytest.raises(
            SchemaSourceError, match="qualified label .* longer than 64"
        ):
            _map(model, on_collision="qualify")

    def test_an_override_is_not_qualified(self) -> None:
        mapping = _map(
            self._model(), on_collision="qualify", labels={"ex:Bank": "BANK"}
        )
        assert _labels(mapping)[0] == ["BANK", "BANK_GEO", "RIVER"]

    def test_entity_and_relationship_labels_may_coincide(self) -> None:
        model = OntologyModel(
            classes=(_cls("Owner"),),
            properties=(_prop("owner", domains=ends("Owner")),),
        )
        assert _labels(_map(model)) == (["OWNER"], ["OWNER"])


class TestDescriptions:
    def test_truncate_at_a_word_boundary(self) -> None:
        assert truncate("alpha beta gamma", 12) == "alpha beta…"
        assert truncate("alpha beta gamma", 10) == "alpha beta…"
        assert truncate("alpha  beta", 20) == "alpha beta"
        assert truncate("supercalifragilistic", 5) == "super…"
        assert truncate("공급업체는물건을판다", 4) == "공급업체…"

    def test_language_outer_property_inner(self) -> None:
        model = OntologyModel(
            classes=(
                _cls(
                    "A",
                    annotations={
                        RDFS_COMMENT: (Text(value="comment en", language="en"),),
                        SKOS_DEFINITION: (
                            Text(value="definition ko", language="ko"),
                            Text(value="definition untagged"),
                        ),
                    },
                ),
            )
        )

        def describe(**options: object) -> str:
            return _map(model, **options).document.entity_types[0].description

        assert describe() == "comment en"
        assert describe(languages=["ko"]) == "definition ko"
        assert describe(languages=[""]) == "definition untagged"
        assert (
            describe(description_properties=["rdfs:comment"], languages=["ko"])
            == "comment en"
        )
        assert describe(max_description_chars=0) == ""
        assert describe(max_description_chars=7) == "comment…"

    def test_a_description_property_must_occur_unless_a_default(self) -> None:
        note = EX + "note"
        model = OntologyModel(
            classes=(_cls("A", annotations={note: (Text(value="a note"),)}),),
            prefixes={"ex": EX},
        )
        # rdfs:comment and skos:definition occur nowhere: still accepted.
        mapping = _map(
            model, description_properties=["rdfs:comment", "ex:note", "skos:definition"]
        )
        assert mapping.document.entity_types[0].description == "a note"
        with pytest.raises(SchemaSourceError, match="'ex:remark' is neither"):
            _map(model, description_properties=["ex:remark"])
        # A property the ontology declares may go unused.
        declared = model.model_copy(
            update={"properties": (_prop("remark", kind="annotation"),)}
        )
        assert _map(declared, description_properties=["ex:remark"]).document


class TestScale:
    """Deep hierarchies cost time and memory linear in their size (a set
    of every ancestor per class would grow with the depth squared)."""

    @pytest.mark.timeout(10)
    def test_a_deep_subclass_chain(self) -> None:
        depth = 20_000
        classes = [_cls("C0")] + [_cls(f"C{i}", f"C{i - 1}") for i in range(1, depth)]
        model = OntologyModel(
            classes=tuple(classes),
            properties=(_prop("p", domains=ends("C0"), ranges=ends(f"C{depth - 1}")),),
            prefixes={"ex": EX},
        )
        mapping = _map(model, classes=[f"ex:C{depth - 3}"])
        assert mapping.report.entity_types == 3
        assert _patterns(mapping) == [
            f"C{depth - 3} -P-> C{depth - 1}",
            f"C{depth - 2} -P-> C{depth - 1}",
            f"C{depth - 1} -P-> C{depth - 1}",
        ]

    @pytest.mark.timeout(10)
    def test_a_long_sub_property_chain(self) -> None:
        length = 2_000
        properties = [_prop("p0", domains=ends("A"))] + [
            _prop(
                f"p{i}",
                parents=frozenset({EX + f"p{i - 1}"}),
                ranges=ends("A" if i % 2 else "B"),
            )
            for i in range(1, length)
        ]
        model = OntologyModel(
            classes=(_cls("A"), _cls("B", "A")),
            properties=tuple(properties),
            prefixes={"ex": EX},
        )
        mapping = _map(model, max_relationship_types=length, max_patterns=10 * length)
        # p0 and p1 join A and B to either; every later one, with B among
        # its inherited ranges, to B.
        assert mapping.report.patterns == 4 + 4 + 2 * (length - 2)
        hints = {
            spec.label: spec.description for spec in mapping.document.relationship_types
        }
        assert hints[f"P{length - 1}"] == f"Narrower than P{length - 2}."


class TestModelAndOptions:
    def test_duplicate_term_is_refused_but_a_pun_is_not(self) -> None:
        with pytest.raises(ValidationError, match="declared twice"):
            OntologyModel(classes=(_cls("A"), _cls("A")))
        OntologyModel(classes=(_cls("A"),), properties=(_prop("A"),))

    def test_iri_with_whitespace_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            OntologyClass(iri="https://example.org/a b")

    def test_options_forbid_unknown_keys_and_lowercase_languages(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            MappingOptions.model_validate({"clases": []})
        assert MappingOptions(languages=["EN-US", ""]).languages == ("en-us", "")
        with pytest.raises(ValidationError):
            MappingOptions(languages=["english language"])
        with pytest.raises(ValidationError):
            MappingOptions(max_depth=-1)

    def test_report_lines(self) -> None:
        report = _map(classes=["ex:Organization"]).report
        assert report.summary().startswith(
            "3 entity types, 3 relationship types and 11 patterns"
        )
        assert (
            f"dropped {EX}employs: no selected class fits its range" in report.details()
        )
        assert f"skipped {EX}name: datatype property" in report.details()

    def test_a_reader_extends_the_options(self) -> None:
        class ReaderOptions(MappingOptions):
            format: str = "auto"

        options = ReaderOptions.model_validate({"format": "turtle", "max_depth": 1})
        assert (
            map_ontology(procure_model(), options).document
            == _map(max_depth=1).document
        )
        with pytest.raises(ValidationError, match="extra"):
            ReaderOptions.model_validate({"fromat": "turtle"})

    def test_mapping_modules_do_not_import_rdflib(self) -> None:
        code = (
            "import sys\n"
            "import unified_kg_rag.adapters.schema_sources.ontology_mapping\n"
            "sys.exit('rdflib' in sys.modules)\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)
