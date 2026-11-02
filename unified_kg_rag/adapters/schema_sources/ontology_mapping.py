# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An ontology's classes and properties as an extraction schema.

:func:`map_ontology` turns an :class:`.ontology_model.OntologyModel` into a
``SchemaDocument``: each selected named class becomes an entity type, each
selected object property a relationship type, and the domains and ranges
the patterns between them. It translates, it does not reason: a fixed set
of RDFS-style rules is applied once, offline, and nothing else is inferred.

Closure.
    ``rdfs:subClassOf`` is transitive over the whole ontology, selected or
    not; a ``subClassOf`` cycle or an ``owl:equivalentClass`` makes one
    type (named by its smallest selected IRI), and likewise for properties.
    A property's domain is what its own ``rdfs:domain`` statements, its
    super-properties' and its inverses' ranges all allow (the range
    mirrors it); ``owl:unionOf`` and ``schema:domainIncludes`` allow any of
    their classes; ``owl:Thing`` and ``rdfs:Resource`` allow anything.
    Transitivity, functionality and cardinality are ignored.

Selection.
    Every class and object property outside the RDF, RDFS, OWL, XSD, SKOS
    and Dublin Core namespaces (or inside ``namespaces``), not deprecated;
    narrowed to the subtrees of ``classes`` (down to ``max_depth``) and
    ``properties``, minus the subtrees of ``exclude_classes`` and
    ``exclude_properties``. A class's selected ancestors are not added for
    it; the closure still goes through the unselected ones.

Flat hierarchy.
    Every selected class is a type of its own. A property's patterns join
    the selected classes its domain and range allow, pushed down to the
    selected subclasses that satisfy every domain statement; when none
    does, lifted to the nearest selected ancestors of a class each statement
    allows; when neither exists, the property is dropped (the report says
    why). A property with neither a domain nor a range gets no pattern.
    A symmetric property joins every class either end allows to every
    other one; an inverse pair is two types, each with its own patterns.

Labels.
    ``skos:prefLabel``, then ``rdfs:label``, in the preferred language
    (``languages`` in order, a tag before its subtags, then untagged, then
    any other tag), else the IRI's local name; written ``UPPER_SNAKE``
    (``supplyChain`` → ``SUPPLY_CHAIN``, "공급 업체" → ``공급_업체``) or as is
    with ``label_case: preserve``; ``labels`` overrides any of them. Two
    terms of one kind with the same label fail the mapping, or with
    ``on_collision: qualify`` take their namespace's prefix as a suffix.

Order.
    Types in a depth-first walk of the selected hierarchy, siblings by
    label; patterns by relationship type, then source, then target. Only
    the statements decide it, never the order a reader produced them in.

Caps.
    More entity types, relationship types or patterns than the ``max_*``
    options allow fail the mapping (truncating would change the prompt and
    the cache keys arbitrarily); the message says how to narrow it.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Annotated, Literal
from urllib.parse import unquote

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
)

from unified_kg_rag.domain.models.schema import (
    EntityTypeItem,
    EntityTypeSpec,
    RelationshipPattern,
    RelationshipPatternItem,
    RelationshipTypeItem,
    RelationshipTypeSpec,
    SchemaDocument,
    schema_key,
    schema_problems,
)
from unified_kg_rag.shared.exceptions import SchemaSourceError

from .ontology_model import (
    DCTERMS,
    OWL,
    RDF,
    RDFS,
    RDFS_LABEL,
    SCHEMA,
    SKOS,
    SKOS_PREF_LABEL,
    XSD,
    OntologyModel,
    OntologyTerm,
    Text,
)

# The prefixes a CURIE of the options may use without binding them.
WELL_KNOWN_PREFIXES: dict[str, str] = {
    "dcterms": DCTERMS,
    "owl": OWL,
    "rdf": RDF,
    "rdfs": RDFS,
    "schema": SCHEMA,
    "skos": SKOS,
    "xsd": XSD,
}
# The ontology languages' own terms: never a type.
_CORE_NAMESPACES = (RDF, RDFS, OWL, XSD)
# Vocabularies an ontology uses to describe itself: left out unless
# ``namespaces`` names them.
_DESCRIPTIVE_NAMESPACES = (SKOS, DCTERMS, "http://purl.org/dc/elements/1.1/")
# Domains and ranges that allow anything.
_UNRESTRICTED = frozenset({OWL + "Thing", RDFS + "Resource"})
# The longest label a type may get (a longer one needs a ``labels`` entry).
MAX_LABEL_CHARS = 64

_Option = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_PrefixName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][\w.-]*$")]
_Language = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, pattern=r"^(|[A-Za-z]{1,8}(-[A-Za-z0-9]{1,8})*)$"
    ),
    AfterValidator(str.lower),
]
_Override = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_LABEL_CHARS),
]


class MappingOptions(BaseModel):
    """How an ontology becomes a schema: what is selected, how it is named
    and described, and the caps. Terms are full IRIs or CURIEs
    (``prefix:name``, read with ``prefixes``, then the well-known prefixes,
    then the ontology's own bindings); a term that names nothing in the
    ontology fails the mapping."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    namespaces: tuple[_Option, ...] = Field(
        default=(),
        description="IRI prefixes of the classes and properties to map "
        "(default: every namespace but RDF, RDFS, OWL, XSD, SKOS and Dublin "
        "Core)",
    )
    classes: tuple[_Option, ...] = Field(
        default=(),
        description="Root classes: each is mapped with its subclasses "
        "(default: every class)",
    )
    exclude_classes: tuple[_Option, ...] = Field(
        default=(), description="Classes left out, with their subclasses"
    )
    max_depth: int | None = Field(
        default=None,
        ge=0,
        description="How many subclass levels below the roots to map "
        "(0: the roots only; default: all)",
    )
    properties: tuple[_Option, ...] = Field(
        default=(),
        description="Root properties: each is mapped with its sub-properties "
        "(default: every object property)",
    )
    exclude_properties: tuple[_Option, ...] = Field(
        default=(), description="Properties left out, with their sub-properties"
    )
    include_deprecated: bool = Field(
        default=False, description="Map terms marked owl:deprecated too"
    )
    prefixes: dict[_PrefixName, _Option] = Field(
        default_factory=dict,
        description="Prefix -> namespace IRI for the CURIEs of these options",
    )
    label_source: Literal["auto", "label", "local_name"] = Field(
        default="auto",
        description="auto: skos:prefLabel, then rdfs:label; label: rdfs:label "
        "only; local_name: the IRI's local name only. Without a label the "
        "local name is used",
    )
    label_case: Literal["upper_snake", "preserve"] = Field(
        default="upper_snake",
        description="upper_snake: SUPPLY_CHAIN; preserve: the label as "
        "written, punctuation made spaces",
    )
    languages: tuple[_Language, ...] = Field(
        default=("en", ""),
        description="Preferred language tags of labels and descriptions, in "
        'order ("" = untagged); any other tag comes after them',
    )
    labels: dict[_Option, _Override] = Field(
        default_factory=dict,
        description="IRI or CURIE -> the label to use, as written",
    )
    on_collision: Literal["error", "qualify"] = Field(
        default="error",
        description="Two terms with one label: fail (error), or suffix each "
        "label with its namespace's prefix (qualify)",
    )
    description_properties: tuple[_Option, ...] = Field(
        default=("skos:definition", "rdfs:comment"),
        description="Annotation properties a type's description is read "
        "from, first match wins (within the preferred language)",
    )
    max_description_chars: int = Field(
        default=300,
        ge=0,
        description="Longer descriptions are cut at a word boundary (0: no "
        "description from the ontology)",
    )
    hierarchy_hints: bool = Field(
        default=True,
        description='Append "Kind of X.", "Narrower than P." and "Inverse '
        'of Q." to the descriptions',
    )
    patterns: Literal["closure", "none"] = Field(
        default="closure",
        description="closure: patterns from the domains and ranges; none: "
        "no patterns",
    )
    max_entity_types: int = Field(
        default=64, ge=1, description="Fail when more classes are selected"
    )
    max_relationship_types: int = Field(
        default=64, ge=1, description="Fail when more properties are selected"
    )
    max_patterns: int = Field(
        default=256, ge=1, description="Fail when more patterns result"
    )


class MappingNote(BaseModel):
    """One term the mapping treated specially, and how."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    iri: str
    detail: str


class MappingReport(BaseModel):
    """What the mapping selected, and what it left out or changed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    classes: int = Field(description="Classes in the ontology")
    properties: int = Field(description="Properties in the ontology")
    entity_types: int
    relationship_types: int
    patterns: int
    merged: tuple[tuple[str, ...], ...] = Field(
        default=(), description="Equivalent terms mapped as one type"
    )
    deprecated: tuple[str, ...] = Field(
        default=(), description="Deprecated terms left out"
    )
    skipped: tuple[MappingNote, ...] = Field(
        default=(), description="Datatype and annotation properties left out"
    )
    dropped: tuple[MappingNote, ...] = Field(
        default=(), description="Selected properties no selected class can carry"
    )
    lifted: tuple[MappingNote, ...] = Field(
        default=(), description="Property ends lifted to a selected ancestor"
    )
    unrestricted: tuple[str, ...] = Field(
        default=(), description="Properties with neither domain nor range"
    )
    qualified: tuple[MappingNote, ...] = Field(
        default=(), description="Labels suffixed to resolve a collision"
    )

    def summary(self) -> str:
        """One line for the log."""
        return (
            f"{self.entity_types} entity types, {self.relationship_types} "
            f"relationship types and {self.patterns} patterns from "
            f"{self.classes} classes and {self.properties} properties; "
            f"{len(self.dropped)} properties dropped, {len(self.skipped)} "
            f"datatype/annotation properties skipped, {len(self.deprecated)} "
            f"deprecated terms left out, {len(self.lifted)} ends lifted, "
            f"{len(self.merged)} equivalences merged, {len(self.qualified)} "
            "labels qualified"
        )

    def details(self) -> list[str]:
        """One line per term listed, for a debug log."""
        return (
            [f"merged: {', '.join(group)}" for group in self.merged]
            + [f"deprecated: {iri}" for iri in self.deprecated]
            + [f"skipped {note.iri}: {note.detail}" for note in self.skipped]
            + [f"dropped {note.iri}: {note.detail}" for note in self.dropped]
            + [f"lifted {note.iri}: {note.detail}" for note in self.lifted]
            + [f"unrestricted: {iri}" for iri in self.unrestricted]
            + [f"qualified {note.iri}: {note.detail}" for note in self.qualified]
        )


class OntologyMapping(BaseModel):
    """The schema an ontology maps to, and the report of how."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document: SchemaDocument
    report: MappingReport


def map_ontology(
    model: OntologyModel, options: MappingOptions | None = None
) -> OntologyMapping:
    """The extraction schema ``model`` maps to under ``options``.

    Raises:
        SchemaSourceError: An option names nothing in the ontology or uses
            an unbound prefix, nothing is selected, a cap is exceeded, a
            term has no usable label, or two terms share a label (and
            ``on_collision`` is ``error``, or qualifying them does not tell
            them apart).
    """
    return _Mapper(model, options or MappingOptions()).run()


# -- Labels ------------------------------------------------------------------


def local_name(iri: str) -> str:
    """The part of ``iri`` after its last ``#``, else after its last ``/``,
    else after its last ``:``, percent-decoded."""
    return unquote(iri[len(namespace_of(iri)) :])


def namespace_of(iri: str) -> str:
    """``iri`` up to and including the separator before its local name."""
    for separator in ("#", "/", ":"):
        index = iri.rfind(separator)
        if index != -1 and index < len(iri) - 1:
            return iri[: index + 1]
    return ""


def _is_upper(char: str) -> bool:
    return unicodedata.category(char) in ("Lu", "Lt")


def _is_lower(char: str) -> bool:
    return unicodedata.category(char) == "Ll"


def _kept(char: str) -> bool:
    # Letters, digits and combining marks (Hangul, CJK and accents included).
    return unicodedata.category(char)[0] in "LNM"


_UNDERSCORES = re.compile(r"_+")


def upper_snake(text: str) -> str:
    """``text`` as an ``UPPER_SNAKE`` label.

    NFKC-normalized; a word break before an upper-case letter that follows
    a lower-case one (``supplyChain``) or that starts a word after an
    acronym or a number (``HTTPServer``, ``Level2Item``; not ``B2B``); every
    character but letters, digits and marks a break; breaks written ``_``,
    collapsed and trimmed; upper-cased.
    Scripts without case (Hangul, CJK) stay as they are: "공급 업체" →
    ``공급_업체``.
    """
    text = unicodedata.normalize("NFKC", text)
    out: list[str] = []
    for index, char in enumerate(text):
        if not _kept(char):
            out.append("_")
            continue
        if index and _is_upper(char):
            previous = text[index - 1]
            following = text[index + 1] if index + 1 < len(text) else ""
            starts_word = bool(following) and _is_lower(following)
            if _is_lower(previous) or (
                starts_word and (_is_upper(previous) or previous.isdigit())
            ):
                out.append("_")
        out.append(char)
    return _UNDERSCORES.sub("_", "".join(out)).strip("_").upper()


def preserve_case(text: str) -> str:
    """``text`` NFKC-normalized, every character but letters, digits,
    marks, ``_`` and ``-`` a space, whitespace collapsed."""
    text = unicodedata.normalize("NFKC", text)
    kept = "".join(char if _kept(char) or char in "_-" else " " for char in text)
    return " ".join(kept.split())


def _language_rank(tag: str, languages: Sequence[str]) -> tuple[int, int, int, str]:
    tag = tag.lower()
    for index, language in enumerate(languages):
        if tag == language:
            return (0, index, 0, "")
        if language and tag.startswith(language + "-"):
            return (0, index, 1, "")
    if not tag:
        return (1, 0, 0, "")
    return (2, 0, 0, tag)


def best_text(
    values: Mapping[str, Sequence[Text]],
    properties: Sequence[str],
    languages: Sequence[str],
) -> str | None:
    """The preferred non-blank value of ``properties`` in ``values``:
    language first (``languages`` in order, an exact tag before its
    subtags, then untagged, then other tags alphabetically), then the order
    of ``properties``, then the smallest NFKC string."""
    ranked = [
        (
            _language_rank(text.language, languages),
            position,
            unicodedata.normalize("NFKC", text.value),
            text.value,
        )
        for position, prop in enumerate(properties)
        for text in values.get(prop, ())
        if text.value.strip()
    ]
    return min(ranked)[3] if ranked else None


def truncate(text: str, limit: int) -> str:
    """``text`` with whitespace collapsed, cut to ``limit`` characters at a
    word boundary (mid-word only when one word is longer) with "…"."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if text[limit] != " " and " " in cut:
        cut = cut[: cut.rindex(" ")]
    return cut.rstrip() + "…"


# -- Hierarchy ---------------------------------------------------------------


class _Hierarchy:
    """A subsumption graph with its cycles and equivalences collapsed.

    A group is the set of terms one cycle or chain of equivalences joins,
    identified by its smallest IRI; ``ancestors[g]`` holds ``g`` and every
    group above it.
    """

    def __init__(
        self,
        nodes: Iterable[str],
        parents: Mapping[str, Iterable[str]],
        equivalents: Mapping[str, Iterable[str]],
    ) -> None:
        edges: dict[str, set[str]] = {node: set() for node in nodes}
        for node, above in parents.items():
            edges.setdefault(node, set()).update(above)
        for node, same in equivalents.items():
            for other in same:
                edges.setdefault(node, set()).add(other)
                edges.setdefault(other, set()).add(node)
        for above in list(edges.values()):
            for node in above:
                edges.setdefault(node, set())
        self.group_of = _strongly_connected(edges)
        self.members: dict[str, list[str]] = {}
        for node in sorted(edges):
            self.members.setdefault(self.group_of[node], []).append(node)
        self.parents: dict[str, set[str]] = {group: set() for group in self.members}
        self.children: dict[str, set[str]] = {group: set() for group in self.members}
        for node, above in edges.items():
            group = self.group_of[node]
            for parent in above:
                other = self.group_of[parent]
                if other != group:
                    self.parents[group].add(other)
                    self.children[other].add(group)
        self.ancestors = self._ancestors()

    def _ancestors(self) -> dict[str, frozenset[str]]:
        ancestors: dict[str, frozenset[str]] = {}
        # Parents before children.
        self.order: list[str] = []
        waiting = {group: len(above) for group, above in self.parents.items()}
        ready = deque(sorted(group for group, count in waiting.items() if not count))
        while ready:
            group = ready.popleft()
            self.order.append(group)
            ancestors[group] = frozenset({group}).union(
                *(ancestors[parent] for parent in self.parents[group])
            )
            for child in sorted(self.children[group]):
                waiting[child] -= 1
                if not waiting[child]:
                    ready.append(child)
        return ancestors

    def descendants(
        self, roots: Iterable[str], max_depth: int | None = None
    ) -> set[str]:
        """``roots`` and the groups below them, down to ``max_depth`` levels.

        A group's level is its longest chain of subsumptions up to a root,
        so a statement the closure implies (``C ⊑ A`` beside ``C ⊑ B ⊑ A``)
        does not move it.
        """
        roots = set(roots)
        below = set(roots)
        queue = deque(sorted(roots))
        while queue:
            for child in sorted(self.children[queue.popleft()]):
                if child not in below:
                    below.add(child)
                    queue.append(child)
        if max_depth is None:
            return below
        depth: dict[str, int] = {}
        for group in self.order:
            if group in roots:
                depth[group] = 0
            elif group in below:
                depth[group] = 1 + max(
                    depth[parent] for parent in self.parents[group] if parent in below
                )
        return {group for group, level in depth.items() if level <= max_depth}

    def nearest(self, group: str, among: set[str]) -> set[str]:
        """The groups of ``among`` just above ``group``: its proper
        ancestors in ``among`` that are below no other one."""
        above = (self.ancestors[group] & among) - {group}
        return {
            candidate
            for candidate in above
            if not any(
                candidate in self.ancestors[other] for other in above - {candidate}
            )
        }


def _strongly_connected(edges: Mapping[str, set[str]]) -> dict[str, str]:
    """Node -> the smallest node of its strongly connected component
    (Tarjan's algorithm, iterative, nodes and edges visited sorted)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    component: dict[str, str] = {}
    counter = 0
    for start in sorted(edges):
        if start in index:
            continue
        work = [(start, iter(sorted(edges[start])))]
        index[start] = low[start] = counter
        counter += 1
        stack.append(start)
        on_stack.add(start)
        while work:
            node, successors = work[-1]
            advanced = False
            for successor in successors:
                if successor not in index:
                    index[successor] = low[successor] = counter
                    counter += 1
                    stack.append(successor)
                    on_stack.add(successor)
                    work.append((successor, iter(sorted(edges[successor]))))
                    advanced = True
                    break
                if successor in on_stack:
                    low[node] = min(low[node], index[successor])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                members = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    members.append(member)
                    if member == node:
                        break
                representative = min(members)
                for member in members:
                    component[member] = representative
    return component


# -- Mapping -----------------------------------------------------------------

_Ends = frozenset[frozenset[str]]


class _Mapper:
    def __init__(self, model: OntologyModel, options: MappingOptions) -> None:
        self.model = model
        self.options = options
        self.classes = {term.iri: term for term in model.classes}
        self.properties = {term.iri: term for term in model.properties}
        self.namespaces = tuple(
            self._resolve(term, "namespaces") for term in options.namespaces
        )
        for namespace, term in zip(self.namespaces, options.namespaces, strict=True):
            if not any(
                iri.startswith(namespace) for iri in (*self.classes, *self.properties)
            ):
                raise SchemaSourceError(
                    f"schema_source.options.namespaces: {term!r} holds no "
                    "class or property of the ontology"
                )
        class_nodes = set(self.classes)
        for prop in model.properties:
            for ends in (prop.domains, prop.ranges):
                for union in ends:
                    class_nodes.update(union)
        self.class_tree = _Hierarchy(
            class_nodes,
            {term.iri: term.parents for term in model.classes},
            {term.iri: term.equivalents for term in model.classes},
        )
        self.property_tree = _Hierarchy(
            self.properties,
            {term.iri: term.parents for term in model.properties},
            {term.iri: term.equivalents for term in model.properties},
        )
        self.description_properties = [
            self._resolve(term, "description_properties")
            for term in options.description_properties
        ]
        for term in options.labels:
            iri = self._resolve(term, "labels")
            if (
                iri not in self.class_tree.group_of
                and iri not in self.property_tree.group_of
            ):
                raise SchemaSourceError(
                    f"schema_source.options.labels: {term!r} names no class "
                    "or property of the ontology"
                )
        self.deprecated: set[str] = set()
        self.notes: dict[str, list[MappingNote]] = {
            "skipped": [],
            "dropped": [],
            "lifted": [],
            "qualified": [],
        }
        self.unrestricted: list[str] = []

    # Terms of the options.

    def _binding(self, prefix: str) -> str | None:
        if prefix in self.options.prefixes:
            return self.options.prefixes[prefix]
        bound = {
            namespace
            for namespace in (
                WELL_KNOWN_PREFIXES.get(prefix),
                self.model.prefixes.get(prefix),
            )
            if namespace is not None
        }
        if len(bound) > 1:
            raise SchemaSourceError(
                f"schema_source.options: prefix '{prefix}' is bound to "
                f"{' and '.join(sorted(bound))}; say which in "
                f"schema_source.options.prefixes.{prefix}"
            )
        return bound.pop() if bound else None

    def _resolve(self, term: str, option: str) -> str:
        if "://" in term or term.startswith("urn:"):
            return term
        prefix, separator, name = term.partition(":")
        namespace = self._binding(prefix) if separator else None
        if namespace is None:
            raise SchemaSourceError(
                f"schema_source.options.{option}: {term!r} is neither an IRI "
                "nor a CURIE of a bound prefix; bind its prefix in "
                "schema_source.options.prefixes or write the full IRI"
            )
        return namespace + name

    def _known(
        self, terms: Iterable[str], option: str, tree: _Hierarchy, what: str
    ) -> set[str]:
        groups: set[str] = set()
        for term in terms:
            iri = self._resolve(term, option)
            if iri not in tree.group_of:
                raise SchemaSourceError(
                    f"schema_source.options.{option}: {term!r} names no "
                    f"{what} of the ontology"
                )
            groups.add(tree.group_of[iri])
        return groups

    def _in_scope(self, iri: str) -> bool:
        if iri.startswith(_CORE_NAMESPACES):
            return False
        if self.namespaces:
            return iri.startswith(self.namespaces)
        return not iri.startswith(_DESCRIPTIVE_NAMESPACES)

    def _candidates(
        self, terms: Iterable[str], deprecated: Callable[[str], bool]
    ) -> set[str]:
        selected = set()
        for iri in terms:
            if not self._in_scope(iri):
                continue
            if deprecated(iri) and not self.options.include_deprecated:
                self.deprecated.add(iri)
                continue
            selected.add(iri)
        return selected

    # Selection.

    def _select_classes(self) -> dict[str, list[str]]:
        """Selected class group -> its selected members, sorted."""
        tree = self.class_tree
        options = self.options
        members = self._candidates(
            self.classes, lambda iri: self.classes[iri].deprecated
        )
        candidates = {tree.group_of[iri] for iri in members}
        if options.classes:
            roots = self._known(options.classes, "classes", tree, "class")
        else:
            roots = {
                group
                for group in candidates
                if not (tree.ancestors[group] - {group}) & candidates
            }
        if options.classes or options.max_depth is not None:
            candidates &= tree.descendants(roots, options.max_depth)
        excluded = self._known(
            options.exclude_classes, "exclude_classes", tree, "class"
        )
        candidates -= tree.descendants(excluded)
        if not candidates:
            raise SchemaSourceError(
                f"schema_source.options: no class of the ontology is selected "
                f"({len(self.classes)} declared); check namespaces, classes, "
                "exclude_classes, max_depth and include_deprecated"
            )
        if len(candidates) > options.max_entity_types:
            raise SchemaSourceError(
                f"schema_source.options: {len(candidates)} classes are "
                f"selected, more than max_entity_types ({options.max_entity_types}); "
                "narrow the selection with namespaces, classes, exclude_classes "
                "or max_depth, or raise max_entity_types"
            )
        return {
            group: sorted(iri for iri in tree.members[group] if iri in members)
            for group in candidates
        }

    def _select_properties(self) -> dict[str, list[str]]:
        """Selected property group -> its selected members, sorted (before
        the endpoint check)."""
        tree = self.property_tree
        options = self.options
        usable = set()
        for iri in sorted(self.properties):
            prop = self.properties[iri]
            if prop.kind in ("object", "untyped"):
                usable.add(iri)
            elif self._in_scope(iri):
                self.notes["skipped"].append(
                    MappingNote(iri=iri, detail=f"{prop.kind} property")
                )
        members = self._candidates(usable, lambda iri: self.properties[iri].deprecated)
        candidates = {tree.group_of[iri] for iri in members}
        if options.properties:
            roots = self._known(options.properties, "properties", tree, "property")
            candidates &= tree.descendants(roots)
        excluded = self._known(
            options.exclude_properties, "exclude_properties", tree, "property"
        )
        candidates -= tree.descendants(excluded)
        return {
            group: sorted(iri for iri in tree.members[group] if iri in members)
            for group in candidates
        }

    # Closure.

    def _class_unions(self, ends: _Ends) -> set[frozenset[str]]:
        group_of = self.class_tree.group_of
        unions = set()
        for union in ends:
            if union & _UNRESTRICTED:
                continue
            groups = frozenset(group_of[iri] for iri in union)
            if groups & {group_of[iri] for iri in _UNRESTRICTED if iri in group_of}:
                continue
            unions.add(groups)
        return unions

    def _closure(
        self,
    ) -> tuple[
        dict[str, set[frozenset[str]]],
        dict[str, set[frozenset[str]]],
        dict[str, set[str]],
        set[str],
    ]:
        """Per property group: the domain and range statements in effect,
        the inverse groups and whether it is symmetric."""
        tree = self.property_tree
        domains: dict[str, set[frozenset[str]]] = {
            group: set() for group in tree.members
        }
        ranges: dict[str, set[frozenset[str]]] = {
            group: set() for group in tree.members
        }
        inverses: dict[str, set[str]] = {group: set() for group in tree.members}
        symmetric: set[str] = set()
        for prop in self.model.properties:
            group = tree.group_of[prop.iri]
            domains[group] |= self._class_unions(prop.domains)
            ranges[group] |= self._class_unions(prop.ranges)
            if prop.symmetric:
                symmetric.add(group)
            for other in prop.inverses:
                if other not in tree.group_of:
                    continue
                inverse = tree.group_of[other]
                inverses[group].add(inverse)
                inverses[inverse].add(group)
        for group in tree.members:
            if group in inverses[group]:
                # Its own inverse: symmetric.
                inverses[group].discard(group)
                symmetric.add(group)
        changed = True
        while changed:
            changed = False
            for group in sorted(tree.members):
                domain = set(domains[group])
                range_ = set(ranges[group])
                for above in tree.ancestors[group]:
                    domain |= domains[above]
                    range_ |= ranges[above]
                for inverse in inverses[group]:
                    domain |= ranges[inverse]
                    range_ |= domains[inverse]
                if domain != domains[group] or range_ != ranges[group]:
                    domains[group], ranges[group] = domain, range_
                    changed = True
        return domains, ranges, inverses, symmetric

    def _match(
        self, unions: set[frozenset[str]], selected: set[str]
    ) -> tuple[set[str], bool]:
        """The selected classes ``unions`` allows, and whether they were
        lifted to ancestors."""
        if not unions:
            return set(selected), False
        ancestors = self.class_tree.ancestors
        down = {
            group
            for group in selected
            if all(ancestors[group] & union for union in unions)
        }
        if down:
            return down, False
        up = {
            group
            for group in selected
            if all(
                any(group in ancestors[allowed] for allowed in union)
                for union in unions
            )
        }
        return {
            group
            for group in up
            if not any(group in ancestors[other] for other in up - {group})
        }, True

    # Labels.

    def _label(
        self, iri: str, members: Sequence[str], terms: Mapping[str, OntologyTerm]
    ) -> str:
        options = self.options
        normalise = (
            upper_snake if options.label_case == "upper_snake" else preserve_case
        )
        label = ""
        if options.label_source != "local_name":
            sources = (
                [SKOS_PREF_LABEL, RDFS_LABEL]
                if options.label_source == "auto"
                else [RDFS_LABEL]
            )
            text = best_text(
                self._annotations(members, terms), sources, options.languages
            )
            if text is not None:
                label = normalise(text)
        if not label:
            label = normalise(local_name(iri))
        if not label or len(label) > MAX_LABEL_CHARS:
            raise SchemaSourceError(
                f"schema_source: {iri} has no usable label "
                f"({'empty' if not label else f'longer than {MAX_LABEL_CHARS} characters'}); "
                "give it one in schema_source.options.labels"
            )
        return label

    @staticmethod
    def _annotations(
        members: Sequence[str], terms: Mapping[str, OntologyTerm]
    ) -> dict[str, list[Text]]:
        merged: dict[str, list[Text]] = {}
        for iri in members:
            term = terms[iri]
            for prop, values in term.annotations.items():
                merged.setdefault(prop, []).extend(values)
        return merged

    def _overrides(self, tree: _Hierarchy, what: str) -> dict[str, str]:
        overrides: dict[str, str] = {}
        for term, label in sorted(self.options.labels.items()):
            iri = self._resolve(term, "labels")
            group = tree.group_of.get(iri)
            if group is None:
                continue
            if overrides.get(group, label) != label:
                raise SchemaSourceError(
                    f"schema_source.options.labels: equivalent {what} terms "
                    f"get the labels {overrides[group]!r} and {label!r}"
                )
            overrides[group] = label
        return overrides

    def _labels(
        self,
        selected: Mapping[str, list[str]],
        terms: Mapping[str, OntologyTerm],
        overrides: Mapping[str, str],
        what: str,
    ) -> dict[str, str]:
        """Selected group -> its label, collisions resolved."""
        labels = {
            group: overrides.get(group) or self._label(members[0], members, terms)
            for group, members in selected.items()
        }
        clashes = _clashes(labels)
        if clashes and self.options.on_collision == "error":
            raise SchemaSourceError(
                f"schema_source: {what} labels collide: "
                + "; ".join(
                    f"{labels[groups[0]]!r} names "
                    + ", ".join(selected[group][0] for group in groups)
                    for groups in clashes
                )
                + " (set schema_source.options.on_collision: qualify, or "
                "give them labels in schema_source.options.labels)"
            )
        for groups in clashes:
            for group in groups:
                if group in overrides:
                    continue
                iri = selected[group][0]
                labels[group] = f"{labels[group]}_{self._qualifier(iri)}"
                self.notes["qualified"].append(
                    MappingNote(iri=iri, detail=labels[group])
                )
        clashes = _clashes(labels)
        if clashes:
            raise SchemaSourceError(
                f"schema_source: {what} labels still collide after "
                "qualifying: "
                + "; ".join(
                    f"{labels[groups[0]]!r} names "
                    + ", ".join(selected[group][0] for group in groups)
                    for groups in clashes
                )
                + " (give them labels in schema_source.options.labels)"
            )
        for group, label in labels.items():
            if len(label) > MAX_LABEL_CHARS:
                raise SchemaSourceError(
                    f"schema_source: the qualified label {label!r} of "
                    f"{selected[group][0]} is longer than {MAX_LABEL_CHARS} "
                    "characters; give it one in schema_source.options.labels"
                )
        return labels

    def _qualifier(self, iri: str) -> str:
        namespace = namespace_of(iri)
        bound = {**WELL_KNOWN_PREFIXES, **self.model.prefixes, **self.options.prefixes}
        prefixes = sorted(
            prefix for prefix, value in bound.items() if prefix and value == namespace
        )
        if prefixes:
            return upper_snake(prefixes[0]) or prefixes[0].upper()
        return hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:6].upper()

    def _description(
        self,
        members: Sequence[str],
        terms: Mapping[str, OntologyTerm],
        hints: list[str],
    ) -> str:
        options = self.options
        parts = []
        if options.max_description_chars:
            text = best_text(
                self._annotations(members, terms),
                self.description_properties,
                options.languages,
            )
            if text is not None and text.split():
                parts.append(truncate(text, options.max_description_chars))
        if options.hierarchy_hints:
            parts.extend(hints)
        return " ".join(parts)

    # The run.

    def run(self) -> OntologyMapping:
        options = self.options
        classes = self._select_classes()
        selected = set(classes)
        properties = self._select_properties()
        domains, ranges, inverses, symmetric = self._closure()

        ends: dict[str, tuple[set[str], set[str]] | None] = {}
        for group in sorted(properties):
            iri = properties[group][0]
            domain, range_ = domains[group], ranges[group]
            if group in symmetric:
                domain, range_ = domain or range_, range_ or domain
            if not domain and not range_:
                ends[group] = None
                self.unrestricted.append(iri)
                continue
            sources, sources_lifted = self._match(domain, selected)
            targets, targets_lifted = self._match(range_, selected)
            if group in symmetric:
                sources = targets = sources | targets
            missing = [
                side
                for side, found in (("domain", sources), ("range", targets))
                if not found
            ]
            if missing:
                self.notes["dropped"].append(
                    MappingNote(
                        iri=iri,
                        detail=f"no selected class fits its {' or '.join(missing)}",
                    )
                )
                continue
            for side, lifted, found in (
                ("domain", sources_lifted, sources),
                ("range", targets_lifted, targets),
            ):
                if lifted:
                    self.notes["lifted"].append(
                        MappingNote(
                            iri=iri,
                            detail=f"{side} lifted to "
                            + ", ".join(sorted(classes[group][0] for group in found)),
                        )
                    )
            ends[group] = (sources, targets)

        if len(ends) > options.max_relationship_types:
            raise SchemaSourceError(
                f"schema_source.options: {len(ends)} properties are selected, "
                f"more than max_relationship_types ({options.max_relationship_types}); "
                "narrow the selection with namespaces, properties or "
                "exclude_properties, or raise max_relationship_types"
            )
        counts = {
            group: len(pair[0]) * len(pair[1]) if pair else 0
            for group, pair in ends.items()
        }
        if (
            options.patterns == "closure"
            and sum(counts.values()) > options.max_patterns
        ):
            largest = sorted(counts, key=lambda group: (-counts[group], group))[:5]
            raise SchemaSourceError(
                f"schema_source.options: the selection yields "
                f"{sum(counts.values())} patterns, more than max_patterns "
                f"({options.max_patterns}); the most come from "
                + ", ".join(
                    f"{properties[group][0]} ({counts[group]})" for group in largest
                )
                + ". Narrow classes or properties, exclude some, set "
                "patterns: none, or raise max_patterns"
            )

        emitted = {group: properties[group] for group in ends}
        class_labels = self._labels(
            classes,
            self.classes,
            self._overrides(self.class_tree, "class"),
            "entity type",
        )
        property_labels = self._labels(
            emitted,
            self.properties,
            self._overrides(self.property_tree, "property"),
            "relationship type",
        )
        class_order = _forest(self.class_tree, set(classes), class_labels, classes)
        property_order = _forest(
            self.property_tree, set(emitted), property_labels, emitted
        )
        class_index = {group: index for index, group in enumerate(class_order)}
        property_index = {group: index for index, group in enumerate(property_order)}

        entity_types = []
        for group in class_order:
            above = sorted(
                self.class_tree.nearest(group, selected), key=class_index.__getitem__
            )
            hints = (
                [f"Kind of {', '.join(class_labels[g] for g in above)}."]
                if above
                else []
            )
            entity_types.append(
                EntityTypeSpec(
                    label=class_labels[group],
                    description=self._description(classes[group], self.classes, hints),
                    iri=classes[group][0],
                )
            )
        relationship_types = []
        for group in property_order:
            above = sorted(
                self.property_tree.nearest(group, set(emitted)),
                key=property_index.__getitem__,
            )
            inverse = sorted(
                (g for g in inverses[group] if g in emitted and g != group),
                key=property_index.__getitem__,
            )
            hints = []
            if above:
                hints.append(
                    f"Narrower than {', '.join(property_labels[g] for g in above)}."
                )
            if inverse:
                hints.append(
                    f"Inverse of {', '.join(property_labels[g] for g in inverse)}."
                )
            relationship_types.append(
                RelationshipTypeSpec(
                    label=property_labels[group],
                    description=self._description(
                        emitted[group], self.properties, hints
                    ),
                    iri=emitted[group][0],
                )
            )
        patterns: list[RelationshipPattern] = []
        if options.patterns == "closure":
            triples = sorted(
                (property_index[group], class_index[source], class_index[target])
                for group, pair in ends.items()
                if pair
                for source in pair[0]
                for target in pair[1]
            )
            patterns = [
                RelationshipPattern(
                    source_type=entity_types[source].label,
                    relation=relationship_types[relation].label,
                    target_type=entity_types[target].label,
                )
                for relation, source, target in triples
            ]

        problems = schema_problems(entity_types, relationship_types, patterns)
        if problems:  # pragma: no cover - the steps above rule each one out
            raise SchemaSourceError("schema_source: " + "; ".join(problems))
        try:
            entity_items: list[EntityTypeItem] = list(entity_types)
            relationship_items: list[RelationshipTypeItem] = list(relationship_types)
            pattern_items: list[RelationshipPatternItem] = list(patterns)
            document = SchemaDocument(
                entity_types=entity_items,
                relationship_types=relationship_items,
                relationship_patterns=pattern_items,
            )
        except ValidationError as e:  # pragma: no cover - specs are valid
            raise SchemaSourceError(f"schema_source: {e}") from e
        merged = sorted(
            tuple(members)
            for members in (*classes.values(), *emitted.values())
            if len(members) > 1
        )
        report = MappingReport(
            classes=len(self.classes),
            properties=len(self.properties),
            entity_types=len(entity_types),
            relationship_types=len(relationship_types),
            patterns=len(patterns),
            merged=tuple(merged),
            deprecated=tuple(sorted(self.deprecated)),
            skipped=tuple(self.notes["skipped"]),
            dropped=tuple(self.notes["dropped"]),
            lifted=tuple(self.notes["lifted"]),
            unrestricted=tuple(self.unrestricted),
            qualified=tuple(sorted(self.notes["qualified"], key=lambda n: n.iri)),
        )
        return OntologyMapping(document=document, report=report)


def _clashes(labels: Mapping[str, str]) -> list[list[str]]:
    """The groups sharing a label key, each sorted, by smallest group."""
    by_key: dict[str, list[str]] = {}
    for group in sorted(labels):
        by_key.setdefault(schema_key(labels[group]), []).append(group)
    return sorted(groups for groups in by_key.values() if len(groups) > 1)


def _forest(
    tree: _Hierarchy,
    nodes: set[str],
    labels: Mapping[str, str],
    members: Mapping[str, list[str]],
) -> list[str]:
    """``nodes`` in depth-first preorder of their hierarchy: roots and
    siblings by (label key, IRI), a node under several parents at its first
    visit."""

    def key(group: str) -> tuple[str, str]:
        return (schema_key(labels[group]), members[group][0])

    children: dict[str, list[str]] = {node: [] for node in nodes}
    roots = []
    for node in nodes:
        parents = tree.nearest(node, nodes)
        if not parents:
            roots.append(node)
        for parent in parents:
            children[parent].append(node)
    order: list[str] = []
    seen: set[str] = set()
    stack = sorted(roots, key=key, reverse=True)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        order.append(node)
        stack.extend(sorted(children[node], key=key, reverse=True))
    return order
