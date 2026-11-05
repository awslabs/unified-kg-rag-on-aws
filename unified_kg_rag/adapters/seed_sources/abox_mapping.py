# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""RDF statements of an A-Box (the instances of an ontology) as a
``SeedGraph``. Pure: no RDF library; a reader hands over its triples as
:class:`RdfTerm` tuples (:mod:`.rdf_reader` does, from rdflib).

:func:`map_abox` translates, it does not reason: no inverse, symmetric or
sub-property edge, no type a subclass statement implies.

Instances.
    Named subjects with at least one ``rdf:type`` outside the RDF, RDFS,
    OWL, XSD and SKOS vocabularies, minus the T-Box: a class (an object of
    ``rdf:type``, or what is typed with an RDF, RDFS or OWL term other than
    ``owl:NamedIndividual``), a property (a predicate, or an end of a
    sub-property, domain, range or inverse statement), and a reified
    statement (``rdf:Statement``). A node's ``id`` and ``iri`` are its IRI.
    Blank nodes, reified statements, RDF-star triple terms and
    ``owl:sameAs`` are skipped and counted, never followed.

Names and descriptions.
    The preferred literal of ``name_properties`` (``languages`` in order,
    then untagged, then any other tag: ``best_text``), else the IRI's
    percent-decoded local name as written; the description likewise from
    ``description_properties``.

Types.
    Each ``rdf:type`` is looked up in the schema's entity type IRIs; when
    several map, the one the schema declares last wins (an owl schema lists
    types in a depth-first preorder, so the last is the most specific).
    With none mapped, the smallest IRI of the instance's other types is
    kept as ``type_iri`` with the label ``labels`` gives it, else its local
    name in ``UPPER_SNAKE``.

Edges.
    A triple whose object is an instance, unless its predicate is an RDF,
    RDFS or OWL term. Typed by the relationship type whose IRI is the
    predicate, else ``labels``, else the predicate's local name in
    ``UPPER_SNAKE``; ``predicate_iri`` is the predicate. A named object
    that is not an instance is skipped and counted: an edge always joins
    two stated nodes, so ``dangling`` never applies.

Properties.
    Every other literal of an instance, keyed by its predicate's CURIE (the
    options' ``prefixes``, then the well-known ones, then the files'
    bindings; the full IRI when none binds its namespace), narrowed by
    ``properties`` and ``exclude_properties``. Typed literals become
    ``int``, ``float`` or ``bool`` (an ill-typed one stays its lexical
    form); a predicate with several values keeps the preferred string
    (by language, as names) or else the smallest value.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, NamedTuple, TypeVar

from pydantic import ConfigDict, Field, ValidationError

from unified_kg_rag.adapters.source_files.locations import SourceTerms
from unified_kg_rag.adapters.source_files.rdf_terms import (
    OWL,
    RDF,
    RDFS,
    SCHEMA,
    SKOS,
    WELL_KNOWN_PREFIXES,
    XSD,
    LabelOverride,
    LanguageTag,
    OptionTerm,
    PrefixConflictError,
    PrefixName,
    Text,
    best_text,
    local_name,
    namespace_of,
    resolve_curie,
    upper_snake,
)
from unified_kg_rag.domain.models.schema import ExtractionSchema
from unified_kg_rag.domain.models.seed_graph import (
    SeedEdge,
    SeedGraph,
    SeedLoadReport,
    SeedNode,
    SeedPropertyValue,
)

from .registry import SeedSourceOptions, statement_problems

_StatementT = TypeVar("_StatementT", SeedNode, SeedEdge)

RDF_TYPE = RDF + "type"
OWL_SAME_AS = OWL + "sameAs"
OWL_IMPORTS = OWL + "imports"
OWL_ONTOLOGY = OWL + "Ontology"
OWL_VERSION_IRI = OWL + "versionIRI"
RDF_STATEMENT = RDF + "Statement"
SCHEMA_HTTP = "http://schema.org/"
# The vocabularies whose types make no instance.
_VOCABULARIES = (RDF, RDFS, OWL, XSD, SKOS)
# The ontology languages' own terms: their types make a T-Box term, their
# predicates no edge.
_CORE = (RDF, RDFS, OWL)
# Core types an instance may carry.
_INSTANCE_TYPES = frozenset({OWL + "NamedIndividual", OWL + "Thing"})
# Predicates whose ends are classes or properties.
_TBOX_PREDICATES = frozenset(
    {
        RDFS + "subClassOf",
        RDFS + "subPropertyOf",
        RDFS + "domain",
        RDFS + "range",
        OWL + "equivalentClass",
        OWL + "equivalentProperty",
        OWL + "inverseOf",
        OWL + "disjointWith",
        OWL + "complementOf",
        OWL + "propertyDisjointWith",
    }
)
_REIFICATION = frozenset({RDF + "subject", RDF + "predicate", RDF + "object"})

_STRING_TYPES = (None, XSD + "string", RDF + "langString")
_INTEGER_TYPES = frozenset(
    XSD + name
    for name in (
        "integer",
        "int",
        "long",
        "short",
        "byte",
        "nonNegativeInteger",
        "nonPositiveInteger",
        "positiveInteger",
        "negativeInteger",
        "unsignedLong",
        "unsignedInt",
        "unsignedShort",
        "unsignedByte",
    )
)
_FLOAT_TYPES = frozenset(XSD + name for name in ("decimal", "double", "float"))
_INTEGER = re.compile(r"^[+-]?\d+$")
_BOOLEANS = {"true": True, "1": True, "false": False, "0": False}

TermKind = Literal["iri", "blank", "literal", "triple"]


class RdfTerm(NamedTuple):
    """One position of a triple. ``value`` is the IRI, the blank node's
    label or the literal's lexical form; ``datatype`` and ``language`` are
    the literal's (``language`` ``""`` when untagged). ``triple`` is an
    RDF-star triple term (or any term that is none of the others)."""

    kind: TermKind
    value: str = ""
    datatype: str | None = None
    language: str = ""


Triple = tuple[RdfTerm, RdfTerm, RdfTerm]


def iri(value: str) -> RdfTerm:
    """The IRI term ``value``."""
    return RdfTerm("iri", value)


def literal(value: str, datatype: str | None = None, language: str = "") -> RdfTerm:
    """The literal term ``value``."""
    return RdfTerm("literal", value, datatype, language)


class AboxMappingOptions(SeedSourceOptions):
    """``processing.seed_graphs[].options`` of kind ``rdf``: how instances
    become nodes and edges (and the options every kind takes). Terms are
    full IRIs or CURIEs (``prefix:name``, read with ``prefixes``, then the
    well-known prefixes and the files' own bindings, which must agree)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prefixes: dict[PrefixName, OptionTerm] = Field(
        default_factory=dict,
        description="Prefix -> namespace IRI for the CURIEs of these options "
        "and of the property keys",
    )
    name_properties: tuple[OptionTerm, ...] = Field(
        # Full IRIs: a file binding one of these prefixes elsewhere makes
        # a CURIE ambiguous, not the defaults.
        default=(
            SKOS + "prefLabel",
            RDFS + "label",
            SCHEMA + "name",
            SCHEMA_HTTP + "name",
        ),
        description="Literal properties a node's name is read from, in the "
        "preferred language (else the IRI's local name)",
    )
    description_properties: tuple[OptionTerm, ...] = Field(
        default=(
            SKOS + "definition",
            RDFS + "comment",
            SCHEMA + "description",
            SCHEMA_HTTP + "description",
        ),
        description="Literal properties a node's description is read from, "
        "in the preferred language",
    )
    languages: tuple[LanguageTag, ...] = Field(
        default=("en", ""),
        description="Preferred language tags of names, descriptions and "
        'string properties, in order ("" = untagged); any other tag comes '
        "after them",
    )
    labels: dict[OptionTerm, LabelOverride] = Field(
        default_factory=dict,
        description="Class or property IRI (or CURIE) no schema type declares "
        "-> the type label to use (default: its local name in UPPER_SNAKE)",
    )
    properties: tuple[OptionTerm, ...] | None = Field(
        default=None,
        description="The literal properties kept as node properties "
        "(default: every one that is not a name or description property)",
    )
    exclude_properties: tuple[OptionTerm, ...] = Field(
        default=(), description="Literal properties never kept"
    )


class AboxMapping(NamedTuple):
    """The seed graph, the instance IRIs relative to their file (``file:``
    or ``s3:``, which moving the file renames) and the ``owl:imports`` not
    followed."""

    graph: SeedGraph
    relative_iris: list[str]
    unfollowed_imports: list[str]


def unfollowed_imports(triples: Iterable[Triple]) -> list[str]:
    """The ``owl:imports`` objects no ontology of ``triples`` is (by its IRI
    or ``owl:versionIRI``), sorted."""
    known: set[str] = set()
    imported: set[str] = set()
    for subject, predicate, obj in triples:
        if predicate.value == OWL_IMPORTS and obj.kind == "iri":
            imported.add(obj.value)
        elif predicate.value == OWL_VERSION_IRI and obj.kind == "iri":
            known.add(obj.value)
        elif (
            predicate.value == RDF_TYPE
            and obj.value == OWL_ONTOLOGY
            and subject.kind == "iri"
        ):
            known.add(subject.value)
    return sorted(imported - known)


def _is_string(term: RdfTerm) -> bool:
    return term.datatype in _STRING_TYPES


def literal_value(term: RdfTerm) -> SeedPropertyValue:
    """The literal ``term`` as a property value: an ``int``, ``float`` or
    ``bool`` for the XSD numeric and boolean types, else (an ill-typed or
    other literal too) its lexical form."""
    lexical, datatype = term.value, term.datatype
    if datatype in _INTEGER_TYPES and _INTEGER.match(lexical.strip()):
        return int(lexical.strip())
    if datatype in _FLOAT_TYPES:
        try:
            number = float(lexical.strip())
        except ValueError:
            return lexical
        return number if math.isfinite(number) else lexical
    if datatype == XSD + "boolean" and lexical.strip() in _BOOLEANS:
        return _BOOLEANS[lexical.strip()]
    return lexical


def _value_order(value: SeedPropertyValue) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


class _Mapper:
    def __init__(
        self,
        options: AboxMappingOptions,
        schema: ExtractionSchema,
        file_prefixes: Mapping[str, str],
        terms: SourceTerms,
    ) -> None:
        self.options = options
        self.terms = terms
        self.file_prefixes = dict(file_prefixes)
        self.skipped: Counter[str] = Counter()
        # IRI -> (position in the schema, label): the later position wins.
        self.entity_iris = {
            spec.iri: (index, spec.label)
            for index, spec in enumerate(schema.entity_types)
            if spec.iri
        }
        self.relationship_iris = {
            spec.iri: spec.label for spec in schema.relationship_types if spec.iri
        }
        self.name_properties = self._resolved(
            options.name_properties, "name_properties"
        )
        self.description_properties = self._resolved(
            options.description_properties, "description_properties"
        )
        self.properties = (
            None
            if options.properties is None
            else set(self._resolved(options.properties, "properties"))
        )
        self.exclude = set(
            self._resolved(options.exclude_properties, "exclude_properties")
        )
        self.labels = {
            self._resolve(term, "labels"): label
            for term, label in options.labels.items()
        }

    # Terms of the options.

    def _resolve(self, term: str, option: str) -> str:
        try:
            resolved = resolve_curie(term, self.options.prefixes, self.file_prefixes)
        except PrefixConflictError as e:
            raise self.terms(
                f"options.{option}: {term!r}: {e}; say which in "
                f"options.prefixes.{e.prefix}"
            ) from e
        if resolved is None:
            raise self.terms(
                f"options.{option}: {term!r} is neither an IRI nor a CURIE of a "
                "bound prefix; bind its prefix in options.prefixes or write the "
                "full IRI"
            )
        return resolved

    def _resolved(self, terms: Sequence[str], option: str) -> list[str]:
        return list(dict.fromkeys(self._resolve(term, option) for term in terms))

    def curie(self, iri: str) -> str:
        """``iri`` as ``prefix:local``: the options' prefixes, then the
        well-known ones, then the files' bindings (the smallest prefix of a
        namespace bound twice); the full IRI when none binds it."""
        namespace = namespace_of(iri)
        if namespace:
            for bound in (
                self.options.prefixes,
                WELL_KNOWN_PREFIXES,
                self.file_prefixes,
            ):
                names = sorted(p for p, ns in bound.items() if p and ns == namespace)
                if names:
                    return f"{names[0]}:{iri[len(namespace):]}"
        return iri

    def label(self, iri: str) -> str | None:
        return self.labels.get(iri) or upper_snake(local_name(iri)) or None

    # The statements.

    def map(self, triples: Iterable[Triple]) -> tuple[list[SeedNode], list[SeedEdge]]:
        types: dict[str, set[str]] = defaultdict(set)
        # Subject IRI -> its other statements.
        statements: dict[str, list[tuple[str, RdfTerm]]] = defaultdict(list)
        tbox: set[str] = set()
        reified: set[str] = set()
        blank_typed: set[str] = set()
        for subject, predicate, obj in triples:
            if "triple" in (subject.kind, predicate.kind, obj.kind):
                self.skipped["rdf_star_triples"] += 1
                continue
            p = predicate.value
            tbox.add(p)
            if p in _TBOX_PREDICATES:
                tbox.update(t.value for t in (subject, obj) if t.kind == "iri")
            if p in _REIFICATION or (p == RDF_TYPE and obj.value == RDF_STATEMENT):
                reified.add(subject.value)
            if subject.kind == "blank":
                if (
                    p == RDF_TYPE
                    and obj.kind == "iri"
                    and not obj.value.startswith(_VOCABULARIES)
                ):
                    blank_typed.add(subject.value)
                continue
            if p == RDF_TYPE and obj.kind == "iri":
                types[subject.value].add(obj.value)
                tbox.add(obj.value)
                if obj.value.startswith(_CORE) and obj.value not in _INSTANCE_TYPES:
                    tbox.add(subject.value)
                continue
            statements[subject.value].append((p, obj))
        self.skipped["reified_statements"] += len(reified)
        self.skipped["blank_nodes"] += len(blank_typed - reified)
        instances = (
            {
                s
                for s, typed in types.items()
                if any(not t.startswith(_VOCABULARIES) for t in typed)
            }
            - tbox
            - reified
        )
        # Named subjects untyped, or typed with vocabulary terms only.
        self.skipped["subjects_not_instances"] += len(
            (set(statements) | set(types)) - instances - tbox - reified
        )
        nodes = [
            self.node(s, types[s], statements.get(s, [])) for s in sorted(instances)
        ]
        # Sorted: the parsers' order is no order.
        edges = sorted(
            (
                edge
                for s in instances
                for edge in self.edges(s, statements.get(s, []), instances)
            ),
            key=lambda e: (e.source, e.predicate_iri or "", e.target),
        )
        return nodes, edges

    def node_type(self, typed: set[str]) -> tuple[str | None, str | None]:
        """``(type label, type_iri)`` of an instance with the types
        ``typed``: the label is None when the IRI maps (the conversion maps
        it)."""
        mapped = [t for t in typed if t in self.entity_iris]
        if mapped:
            chosen = max(mapped, key=lambda t: self.entity_iris[t][0])
            self.skipped["types_not_kept"] += len(
                {t for t in typed if not t.startswith(_VOCABULARIES)} - {chosen}
            )
            return None, chosen
        others = sorted(t for t in typed if not t.startswith(_VOCABULARIES))
        self.skipped["types_not_kept"] += len(others) - 1
        return self.label(others[0]), others[0]

    def node(
        self, subject: str, typed: set[str], statements: list[tuple[str, RdfTerm]]
    ) -> SeedNode:
        texts: dict[str, list[Text]] = defaultdict(list)
        values: dict[str, list[RdfTerm]] = defaultdict(list)
        for predicate, obj in statements:
            if obj.kind != "literal":
                continue
            if predicate in self.name_properties or predicate in (
                self.description_properties
            ):
                if _is_string(obj):
                    texts[predicate].append(
                        Text(value=obj.value, language=obj.language)
                    )
                continue
            if predicate in self.exclude or (
                self.properties is not None and predicate not in self.properties
            ):
                self.skipped["properties_not_selected"] += 1
                continue
            values[predicate].append(obj)
        name = best_text(texts, self.name_properties, self.options.languages)
        description = best_text(
            texts, self.description_properties, self.options.languages
        )
        type_label, type_iri = self.node_type(typed)
        return self.validated(
            subject,
            SeedNode,
            {
                "id": subject,
                "iri": subject,
                "name": name or local_name(subject) or subject,
                "type": type_label,
                "type_iri": type_iri,
                "description": description,
                "properties": {
                    self.curie(predicate): self.property_value(found)
                    for predicate, found in sorted(values.items())
                },
            },
        )

    def validated(
        self, subject: str, model: type[_StatementT], data: dict[str, Any]
    ) -> _StatementT:
        """``data`` as ``model``; an invalid one (a lone surrogate, an IRI or
        a label over its cap) fails naming ``subject``."""
        try:
            return model.model_validate(data)
        except ValidationError as e:
            kind = "node" if model is SeedNode else "an edge from"
            raise self.terms(
                f"{kind} <{subject[:200]}>: {statement_problems(e)}"
            ) from e

    def property_value(self, found: list[RdfTerm]) -> SeedPropertyValue:
        self.skipped["property_values_not_kept"] += len(found) - 1
        strings = [
            Text(value=t.value, language=t.language) for t in found if _is_string(t)
        ]
        chosen = best_text({"": strings}, [""], self.options.languages)
        if chosen is not None:
            return chosen
        return min((literal_value(t) for t in found), key=_value_order)

    def edges(
        self,
        subject: str,
        statements: list[tuple[str, RdfTerm]],
        instances: set[str],
    ) -> Iterable[SeedEdge]:
        for predicate, obj in statements:
            if obj.kind == "literal":
                continue
            if obj.kind == "blank":
                self.skipped["blank_node_objects"] += 1
            elif predicate == OWL_SAME_AS:
                self.skipped["owl_same_as"] += 1
            elif predicate.startswith(_CORE):
                self.skipped["vocabulary_statements"] += 1
            elif obj.value not in instances:
                self.skipped["objects_not_instances"] += 1
            else:
                mapped = predicate in self.relationship_iris
                yield self.validated(
                    subject,
                    SeedEdge,
                    {
                        "source": subject,
                        "target": obj.value,
                        "type": (
                            None
                            if mapped
                            else self.label(predicate) or upper_snake(predicate)
                        ),
                        "predicate_iri": predicate,
                    },
                )


def map_abox(
    triples: Sequence[Triple],
    options: AboxMappingOptions,
    schema: ExtractionSchema,
    *,
    prefixes: Mapping[str, str] | None = None,
    terms: SourceTerms,
    files: int = 0,
    bytes_read: int = 0,
) -> AboxMapping:
    """The seed graph ``triples`` state (the rules in the module
    docstring), against the run's resolved ``schema``. ``prefixes`` are the
    files' own bindings; ``files`` and ``bytes_read`` go to the report.
    The same statements give the same graph, in any order.

    Raises:
        terms.error: A term of the options is neither an IRI nor a CURIE of
            a bound prefix.
    """
    mapper = _Mapper(options, schema, prefixes or {}, terms)
    nodes, edges = mapper.map(triples)
    relative = sorted(n.id for n in nodes if n.id.startswith(("file:", "s3:")))
    return AboxMapping(
        graph=SeedGraph(
            nodes=tuple(nodes),
            edges=tuple(edges),
            report=SeedLoadReport(
                files=files,
                bytes_read=bytes_read,
                skipped={k: v for k, v in sorted(mapper.skipped.items()) if v},
            ),
        ),
        relative_iris=relative,
        unfollowed_imports=unfollowed_imports(triples),
    )
