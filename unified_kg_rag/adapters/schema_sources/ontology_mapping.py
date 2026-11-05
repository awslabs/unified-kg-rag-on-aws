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
    mirrors it); a symmetric property with statements on one end only
    takes them for the other, and what inherits from it sees both.
    ``owl:unionOf`` and ``schema:domainIncludes`` allow any of their
    classes; ``owl:Thing`` and ``rdfs:Resource`` allow anything.
    Transitivity, functionality and cardinality are ignored. Each of these
    is one pass over the hierarchy: time and memory grow about linearly
    with its size, however deep.

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
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal, NamedTuple, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from unified_kg_rag.adapters.source_files.rdf_terms import (
    DCTERMS,
    MAX_LABEL_CHARS,
    OWL,
    RDF,
    RDFS,
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
    preserve_case,
    resolve_curie,
    upper_snake,
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
    RDFS_LABEL,
    SKOS_PREF_LABEL,
    OntologyModel,
    OntologyTerm,
)

# The ontology languages' own terms: never a type.
_CORE_NAMESPACES = (RDF, RDFS, OWL, XSD)
# Vocabularies an ontology uses to describe itself: left out unless
# ``namespaces`` names them.
_DESCRIPTIVE_NAMESPACES = (SKOS, DCTERMS, "http://purl.org/dc/elements/1.1/")
# Domains and ranges that allow anything.
_UNRESTRICTED = frozenset({OWL + "Thing", RDFS + "Resource"})
# The default ``description_properties``, which an ontology need not use.
_DEFAULT_DESCRIPTIONS = frozenset({SKOS + "definition", RDFS + "comment"})


class MappingOptions(BaseModel):
    """How an ontology becomes a schema: what is selected, how it is named
    and described, and the caps. Terms are full IRIs or CURIEs
    (``prefix:name``, read with ``prefixes``, then the well-known prefixes,
    then the ontology's own bindings); a term that names nothing in the
    ontology fails the mapping."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    namespaces: tuple[OptionTerm, ...] = Field(
        default=(),
        description="IRI prefixes of the classes and properties to map "
        "(default: every namespace but RDF, RDFS, OWL, XSD, SKOS and Dublin "
        "Core)",
    )
    classes: tuple[OptionTerm, ...] = Field(
        default=(),
        description="Root classes: each is mapped with its subclasses "
        "(default: every class)",
    )
    exclude_classes: tuple[OptionTerm, ...] = Field(
        default=(), description="Classes left out, with their subclasses"
    )
    max_depth: int | None = Field(
        default=None,
        ge=0,
        description="How many subclass levels below the roots to map "
        "(0: the roots only; default: all)",
    )
    properties: tuple[OptionTerm, ...] = Field(
        default=(),
        description="Root properties: each is mapped with its sub-properties "
        "(default: every object property)",
    )
    exclude_properties: tuple[OptionTerm, ...] = Field(
        default=(), description="Properties left out, with their sub-properties"
    )
    include_deprecated: bool = Field(
        default=False, description="Map terms marked owl:deprecated too"
    )
    prefixes: dict[PrefixName, OptionTerm] = Field(
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
    languages: tuple[LanguageTag, ...] = Field(
        default=("en", ""),
        description="Preferred language tags of labels and descriptions, in "
        'order ("" = untagged); any other tag comes after them',
    )
    labels: dict[OptionTerm, LabelOverride] = Field(
        default_factory=dict,
        description="IRI or CURIE -> the label to use, as written",
    )
    on_collision: Literal["error", "qualify"] = Field(
        default="error",
        description="Two terms with one label: fail (error), or suffix each "
        "label with its namespace's prefix (qualify): one prefixes binds, "
        "else a well-known one, else the ontology's, else a hash",
    )
    description_properties: tuple[OptionTerm, ...] = Field(
        default=("skos:definition", "rdfs:comment"),
        description="Annotation properties a type's description is read "
        "from, first match wins (within the preferred language); one other "
        "than the defaults must occur in the ontology",
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

# A term or group (``str``), or one end of a property group (``(group,
# DOMAIN)`` or ``(group, RANGE)``).
_Node = TypeVar("_Node", str, tuple[str, int])


class _Hierarchy:
    """A subsumption graph with its cycles and equivalences collapsed.

    A group is the set of terms one cycle or chain of equivalences joins,
    identified by its smallest IRI; ``order`` lists the groups parents
    first. Nothing here holds a group's full set of ancestors (on a deep
    hierarchy those sets alone grow with the square of its depth): what is
    above or below a group is asked of the few groups that matter
    (:meth:`reach`).
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
        self.order = _topological(self.parents, self.children)

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

    def topmost(self, among: set[str]) -> set[str]:
        """The groups of ``among`` below no other one."""
        # Whether a group is below one of ``among``, parents first.
        covered: dict[str, bool] = {}
        for group in self.order:
            covered[group] = any(
                parent in among or covered[parent] for parent in self.parents[group]
            )
        return {group for group in among if not covered[group]}

    def reach(self, among: Iterable[str]) -> _Reach:
        """What of ``among`` is at or above, and at or below, each group."""
        return _Reach(self, among)


class _Reach:
    """For every group of a hierarchy, the groups of ``among`` at or above
    it (``up``) and at or below it (``down``), as bit sets over ``among``
    parents first (a higher bit is never above a lower one). A group's set
    is its parents' (children's) joined, so a chain shares one integer and
    the cost is the edges times the size of ``among`` in machine words,
    not the square of the depth."""

    def __init__(self, tree: _Hierarchy, among: Iterable[str]) -> None:
        among = set(among)
        self.groups = [group for group in tree.order if group in among]
        self.bit = {group: 1 << index for index, group in enumerate(self.groups)}
        self.all = (1 << len(self.groups)) - 1
        self.up = self._join(tree.order, tree.parents)
        self.down = self._join(reversed(tree.order), tree.children)

    def _join(
        self, order: Iterable[str], before: Mapping[str, set[str]]
    ) -> dict[str, int]:
        masks: dict[str, int] = {}
        for group in order:
            mask = self.bit.get(group, 0)
            for other in before[group]:
                # Shared, not copied, while nothing is added to it.
                mask = mask | masks[other] if mask else masks[other]
            masks[group] = mask
        return masks

    def of(self, mask: int) -> set[str]:
        """The groups of ``among`` in ``mask``."""
        found = set()
        while mask:
            low = mask & -mask
            found.add(self.groups[low.bit_length() - 1])
            mask ^= low
        return found

    def lowest(self, mask: int) -> set[str]:
        """The groups of ``mask`` above no other one of it."""
        found = set()
        while mask:
            # The highest bit is below no other one; drop it and what is
            # above it.
            group = self.groups[mask.bit_length() - 1]
            found.add(group)
            mask &= ~self.up[group]
        return found

    def nearest(self, group: str) -> set[str]:
        """The groups of ``among`` just above ``group``: its proper
        ancestors in ``among`` that are below no other one."""
        return self.lowest(self.up[group] & ~self.bit.get(group, 0))


def _strongly_connected(edges: Mapping[_Node, set[_Node]]) -> dict[_Node, _Node]:
    """Node -> the smallest node of its strongly connected component
    (Tarjan's algorithm, iterative, nodes and edges visited sorted)."""
    index: dict[_Node, int] = {}
    low: dict[_Node, int] = {}
    on_stack: set[_Node] = set()
    stack: list[_Node] = []
    component: dict[_Node, _Node] = {}
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


def _topological(
    parents: Mapping[_Node, set[_Node]], children: Mapping[_Node, set[_Node]]
) -> list[_Node]:
    """The nodes of an acyclic graph, parents before children (Kahn's
    algorithm, nodes visited sorted)."""
    order: list[_Node] = []
    waiting = {node: len(above) for node, above in parents.items()}
    ready = deque(sorted(node for node, count in waiting.items() if not count))
    while ready:
        node = ready.popleft()
        order.append(node)
        for child in sorted(children[node]):
            waiting[child] -= 1
            if not waiting[child]:
                ready.append(child)
    return order


# -- Mapping -----------------------------------------------------------------

_Ends = frozenset[frozenset[str]]
DOMAIN, RANGE = 0, 1


class _Fit(NamedTuple):
    """What a set of domain (or range) statements allows of the selected
    classes, as bit sets of a :class:`_Reach`: the classes at or below a
    class of every statement (``down``), and those at or above one
    (``up``). Folding statements in is an intersection, so a property's
    fit is its own statements' joined with those it inherits."""

    stated: bool
    down: int
    up: int

    def join(self, other: _Fit) -> _Fit:
        return _Fit(
            self.stated or other.stated, self.down & other.down, self.up & other.up
        )


def _propagate(
    sources: Mapping[tuple[str, int], set[tuple[str, int]]],
    own: Mapping[tuple[str, int], _Fit],
) -> dict[tuple[str, int], _Fit]:
    """Each property end's own fit joined with that of every end it
    inherits from, directly or not (``sources`` holds the direct ones):
    one pass over the strongly connected components, sources first."""
    component = _strongly_connected(sources)
    members: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for node in sorted(sources):
        members.setdefault(component[node], []).append(node)
    before: dict[tuple[str, int], set[tuple[str, int]]] = {c: set() for c in members}
    after: dict[tuple[str, int], set[tuple[str, int]]] = {c: set() for c in members}
    for node, above in sources.items():
        for source in above:
            if component[source] != component[node]:
                before[component[node]].add(component[source])
                after[component[source]].add(component[node])
    fits: dict[tuple[str, int], _Fit] = {}
    for current in _topological(before, after):
        fit = own[members[current][0]]
        for node in members[current][1:]:
            fit = fit.join(own[node])
        for source in before[current]:
            fit = fit.join(fits[source])
        fits[current] = fit
    return {node: fits[component[node]] for node in sources}


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
        if "description_properties" in options.model_fields_set:
            # The defaults may be absent; another property named here may not.
            used = set(self.properties).union(
                _DEFAULT_DESCRIPTIONS,
                *(term.annotations for term in (*model.classes, *model.properties)),
            )
            for term, iri in zip(
                options.description_properties, self.description_properties, strict=True
            ):
                if iri not in used:
                    raise SchemaSourceError(
                        f"schema_source.options.description_properties: {term!r} "
                        "is neither a property of the ontology nor a literal "
                        "annotation of any of its terms"
                    )
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

    def _resolve(self, term: str, option: str) -> str:
        try:
            resolved = resolve_curie(term, self.options.prefixes, self.model.prefixes)
        except PrefixConflictError as e:
            raise SchemaSourceError(
                f"schema_source.options: {e}; say which in "
                f"schema_source.options.prefixes.{e.prefix}"
            ) from e
        if resolved is None:
            raise SchemaSourceError(
                f"schema_source.options.{option}: {term!r} is neither an IRI "
                "nor a CURIE of a bound prefix; bind its prefix in "
                "schema_source.options.prefixes or write the full IRI (a "
                "prefix the ontology's files bind to different namespaces is "
                "not bound: a WARNING names those when the files are read)"
            )
        return resolved

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
            roots = tree.topmost(candidates)
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
            groups = frozenset(group_of[iri] for iri in union)
            if groups & {group_of[iri] for iri in _UNRESTRICTED if iri in group_of}:
                continue
            unions.add(groups)
        return unions

    def _fit(self, union: frozenset[str], reach: _Reach) -> _Fit:
        down = up = 0
        for group in union:
            down |= reach.down[group]
            up |= reach.up[group]
        return _Fit(True, down, up)

    def _closure(
        self, reach: _Reach
    ) -> tuple[dict[str, tuple[_Fit, _Fit]], dict[str, set[str]], set[str]]:
        """Per property group: the fits of the domain and range statements
        in effect, the inverse groups and whether it is symmetric.

        An end inherits its super-properties' same end and its inverses'
        other end. A symmetric property with statements on one end only
        then takes them for the other too, and what inherits from it sees
        both; the closure is taken again after that, so it does not depend
        on the order the ends are visited in.
        """
        tree = self.property_tree
        anything = _Fit(False, reach.all, reach.all)
        own: dict[tuple[str, int], _Fit] = {
            (group, side): anything
            for group in tree.members
            for side in (DOMAIN, RANGE)
        }
        inverses: dict[str, set[str]] = {group: set() for group in tree.members}
        symmetric: set[str] = set()
        fits: dict[frozenset[str], _Fit] = {}
        for prop in self.model.properties:
            group = tree.group_of[prop.iri]
            for side, ends in ((DOMAIN, prop.domains), (RANGE, prop.ranges)):
                for union in self._class_unions(ends):
                    if union not in fits:
                        fits[union] = self._fit(union, reach)
                    own[(group, side)] = own[(group, side)].join(fits[union])
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
        sources = {
            (group, side): {(parent, side) for parent in tree.parents[group]}
            | {(inverse, 1 - side) for inverse in inverses[group]}
            for group in tree.members
            for side in (DOMAIN, RANGE)
        }
        closed = _propagate(sources, own)
        filled = False
        for group in sorted(symmetric):
            domain, range_ = closed[(group, DOMAIN)], closed[(group, RANGE)]
            if domain.stated != range_.stated:
                own[(group, DOMAIN)] = own[(group, RANGE)] = (
                    domain if domain.stated else range_
                )
                filled = True
        if filled:
            closed = _propagate(sources, own)
        return (
            {
                group: (closed[(group, DOMAIN)], closed[(group, RANGE)])
                for group in tree.members
            },
            inverses,
            symmetric,
        )

    @staticmethod
    def _match(fit: _Fit, reach: _Reach) -> tuple[set[str], bool]:
        """The selected classes ``fit`` allows, and whether they were
        lifted to ancestors: those below every statement, else the lowest
        of those above one class of each."""
        if not fit.stated:
            return set(reach.groups), False
        if fit.down:
            return reach.of(fit.down), False
        return reach.lowest(fit.up), True

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
        # The options' prefixes first, then the well-known ones: only when
        # neither names the namespace do the files' own bindings, which
        # re-serialising may change, decide.
        namespace = namespace_of(iri)
        for bound in (self.options.prefixes, WELL_KNOWN_PREFIXES, self.model.prefixes):
            prefixes = sorted(
                prefix
                for prefix, value in bound.items()
                if prefix and value == namespace
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
        reach = self.class_tree.reach(selected)
        fits, inverses, symmetric = self._closure(reach)

        ends: dict[str, tuple[set[str], set[str]] | None] = {}
        for group in sorted(properties):
            iri = properties[group][0]
            domain, range_ = fits[group]
            if not domain.stated and not range_.stated:
                ends[group] = None
                self.unrestricted.append(iri)
                continue
            sources, sources_lifted = self._match(domain, reach)
            targets, targets_lifted = self._match(range_, reach)
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
        property_reach = self.property_tree.reach(emitted)
        class_order = _forest(reach, class_labels, classes)
        property_order = _forest(property_reach, property_labels, emitted)
        class_index = {group: index for index, group in enumerate(class_order)}
        property_index = {group: index for index, group in enumerate(property_order)}

        entity_types = []
        for group in class_order:
            above = sorted(reach.nearest(group), key=class_index.__getitem__)
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
                property_reach.nearest(group), key=property_index.__getitem__
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
    reach: _Reach,
    labels: Mapping[str, str],
    members: Mapping[str, list[str]],
) -> list[str]:
    """The groups of ``reach`` in depth-first preorder of their hierarchy:
    roots and siblings by (label key, IRI), a node under several parents at
    its first visit."""

    def key(group: str) -> tuple[str, str]:
        return (schema_key(labels[group]), members[group][0])

    children: dict[str, list[str]] = {node: [] for node in reach.groups}
    roots = []
    for node in reach.groups:
        parents = reach.nearest(node)
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
