# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A seed graph as virtual documents of the corpus.

:func:`convert_seed` turns a :class:`~unified_kg_rag.domain.models.seed_graph.SeedGraph`
into shard documents whose text is a lossless, canonical rendering of the
statements, against the run's extraction schema. Everything is a pure
function of the seed, the settings and the schema, and does not depend on
the order of the nodes or edges:

1. Checks: a duplicate node id fails, as does a seed over ``max_nodes``,
   ``max_edges`` or ``max_property_keys``; an edge naming an unknown node
   fails or is skipped (``dangling``); a self-loop is dropped.
2. Types: a ``type_iri`` / ``predicate_iri`` that a schema type declares as
   its ``iri`` names that type (the type declared last when several do);
   else the label as written, as extraction writes it
   (:meth:`SchemaConformance.entity_type`). An edge without a type fails.
   Edges with the same endpoints and type key merge (weights add,
   descriptions union).
3. Conformance: :meth:`SchemaConformance.conform` over the whole seed at
   once, so a pattern check sees both endpoint types. ``align`` counts,
   ``strict`` drops what extraction would drop, and the edges of a dropped
   node.
4. Name collisions: nodes whose names share an ``entity_key`` (one entity)
   but neither an id nor an IRI fail (``error``), stay one entity
   (``merge``) or get `` (<qualifier>)`` appended (``qualify``).
5. Shards: a node goes to shard ``int(sha256(id)[:8], 16) % shards``, an
   edge to its source node's. A node card (the node and its outgoing
   edges, in name order) is packed with others into text units of at most
   ``max_items_per_unit`` statements; a larger card (a hub) is split.
6. Rendering: one document per non-empty shard,
   ``seed_graphs/<name>/<k>-of-<n>.jsonl``: a header line (name, format,
   filters), then each unit's statement lines, a blank line before each
   unit. Statement lines are canonical JSON.

A shard document becomes text units (:func:`seed_text_units`, ids
``sg-<uuid>``, which sort after every document unit id) and each unit's
extraction record is parsed from its text (:func:`parse_seed_unit`): no
model call, and equal texts give equal records, keyed by
:data:`SEED_RECORD_FINGERPRINT`.

Merged with extracted items, documents come first: where a merge picks one
value (type, name, confidence) the statements of document units decide and
a seed's only fill a gap (:func:`document_first`); descriptions are
unioned, document lines first (records merge in unit-id order); an entity's
IRI is the smallest one stated (:func:`merged_iri`). With no seed unit,
every rule is the one it was before seed graphs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Discriminator, Field, TypeAdapter

from unified_kg_rag.domain.ingestion.base_processor import (
    SEED_PROPERTIES_ATTRIBUTE,
    BaseProcessor,
    edge_weight,
)
from unified_kg_rag.domain.ingestion.descriptions import merge_descriptions
from unified_kg_rag.domain.ingestion.document_identity import (
    RELATIVE_PATH_KEY,
    compute_document_id,
    compute_text_hash,
)
from unified_kg_rag.domain.ingestion.relationship_weights import (
    apply_text_unit_weights,
)
from unified_kg_rag.domain.ingestion.schema_conformance import (
    SCHEMA_COUNTERS,
    SchemaConformance,
)
from unified_kg_rag.domain.models import (
    Constants,
    Document,
    Entity,
    Relationship,
    TextUnit,
    TextUnitExtraction,
)
from unified_kg_rag.domain.models.schema import ExtractionSchema, schema_key
from unified_kg_rag.domain.models.seed_graph import (
    SeedEdge,
    SeedGraph,
    SeedNode,
    SeedPropertyValue,
)
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.exceptions import DataProcessingError, SeedGraphError
from unified_kg_rag.shared.utils import compute_hash, entity_key, generate_stable_id

logger = get_logger(__name__)

# Bump when the rendering or parsing rules change: it keys every seed
# unit's record (SEED_RECORD_FINGERPRINT) and is in every shard's header.
SEED_FORMAT_VERSION = 1

# Document metadata "source_kind" and text-unit attribute of seed items.
SEED_SOURCE_KIND = "seed_graph"
# The source kind of every other text unit, in an item's ``source_kinds``.
DOCUMENT_SOURCE_KIND = "document"
# The document filter every item of a seed carries: its name.
SEED_FILTER_KEY = "seed_graph"
# Relative path of the shard documents: <root>/<name>/<k>-of-<n>.jsonl.
SEED_DOCUMENT_ROOT = "seed_graphs"
# Prefix of a seed text unit's id. Document unit ids are lowercase hex
# UUIDs, so "sg-..." sorts after every one of them.
SEED_UNIT_PREFIX = "sg-"
# A shard over this many units fails: a shard is one registry record, and
# its text-unit list must stay well under the record size limit.
MAX_UNITS_PER_SHARD = 2000

# The extraction-store fingerprint of a seed unit's record: it depends only
# on the format, never on model settings (the length of
# cache_keys.unit_record_fingerprint).
SEED_RECORD_FINGERPRINT = compute_hash(
    json.dumps(
        {"kind": SEED_SOURCE_KIND, "format": SEED_FORMAT_VERSION}, sort_keys=True
    ),
    length=12,
)

SeedNameCollision = Literal["error", "merge", "qualify"]
SeedDangling = Literal["error", "skip"]
SeedFilterValue = str | int | float | bool

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_UNIT_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class SeedConversionSettings:
    """How one seed graph is converted (built from its configuration).

    The values a seed's virtual documents depend on; where a source reads
    the seed from is not one of them.
    """

    name: str
    filters: Mapping[str, SeedFilterValue] = field(default_factory=dict)
    on_name_collision: SeedNameCollision = "error"
    # Property whose value qualifies a colliding name; else the IRI's local
    # name, else the node id.
    qualifier_property: str | None = None
    description_properties: tuple[str, ...] = ()
    confidence: float = 1.0
    shards: int = 16
    max_items_per_unit: int = 50
    max_nodes: int = 100_000
    max_edges: int = 500_000
    max_property_keys: int = 64
    max_description_chars: int = 1000
    dangling: SeedDangling = "error"

    def __post_init__(self) -> None:
        problems = []
        if not _NAME.match(self.name):
            problems.append(
                f"name {self.name!r} is not lowercase letters, digits, '-' and '_'"
            )
        if SEED_FILTER_KEY in self.filters:
            problems.append(f"filters: {SEED_FILTER_KEY!r} is reserved")
        if any(
            not isinstance(value, SeedFilterValue) for value in self.filters.values()
        ):
            problems.append("filters: every value is a string, number or boolean")
        if not 0.0 <= self.confidence <= 1.0:
            problems.append("confidence is not between 0 and 1")
        for name, low in (
            ("shards", 1),
            ("max_items_per_unit", 1),
            ("max_nodes", 1),
            ("max_edges", 1),
            ("max_property_keys", 0),
            ("max_description_chars", 1),
        ):
            if getattr(self, name) < low:
                problems.append(f"{name} is below {low}")
        if problems:
            raise SeedGraphError(f"seed graph {self.name!r}: " + "; ".join(problems))
        object.__setattr__(
            self, "filters", MappingProxyType(dict(sorted(self.filters.items())))
        )
        object.__setattr__(
            self, "description_properties", tuple(self.description_properties)
        )

    @property
    def document_filters(self) -> dict[str, SeedFilterValue]:
        """The document filters of the seed's items: its ``filters`` and
        ``seed_graph: <name>``."""
        return {**self.filters, SEED_FILTER_KEY: self.name}


class _Canonical(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SeedNodeStatement(_Canonical):
    """A node line of a seed unit: the entity it states, as converted."""

    kind: Literal["node"] = "node"
    id: str = Field(description="The node's id in the seed")
    name: str = Field(description="The entity's name (qualified if it collided)")
    type: str | None = Field(default=None, description="The type, as converted")
    iri: str | None = Field(default=None, description="The node's IRI")
    description: str | None = Field(
        default=None, description="Description and description properties, cut"
    )
    confidence: float = Field(description="The statement's confidence")
    properties: dict[str, SeedPropertyValue] = Field(default_factory=dict)


class SeedEdgeStatement(_Canonical):
    """An edge line of a seed unit: the relationship it states."""

    kind: Literal["edge"] = "edge"
    source: str = Field(description="Name of the source entity")
    target: str = Field(description="Name of the target entity")
    type: str = Field(description="The type, as converted")
    predicate_iri: str | None = Field(default=None, description="The edge's IRI")
    description: str | None = Field(default=None)
    weight: float = Field(description="The edge's weight")
    properties: dict[str, SeedPropertyValue] = Field(default_factory=dict)


SeedStatement = Annotated[SeedNodeStatement | SeedEdgeStatement, Discriminator("kind")]
_STATEMENT: TypeAdapter[SeedNodeStatement | SeedEdgeStatement] = TypeAdapter(
    SeedStatement
)


class SeedConversionStats(_Canonical):
    """What a conversion read, kept, dropped and merged (counts only)."""

    nodes_read: int = 0
    edges_read: int = 0
    nodes: int = Field(default=0, description="Node statements written")
    edges: int = Field(default=0, description="Edge statements written")
    units: int = 0
    documents: int = 0
    dangling_edges_skipped: int = 0
    self_loops_dropped: int = 0
    duplicate_edges_merged: int = 0
    name_collisions_merged: int = Field(
        default=0, description="Name keys several distinct nodes share, merged"
    )
    names_qualified: int = 0
    nodes_unmapped_type_iri: int = Field(
        default=0, description="Nodes whose type_iri no schema type declares"
    )
    edges_unmapped_predicate_iri: int = 0
    descriptions_truncated: int = 0
    entities_off_schema: int = 0
    entities_dropped_off_schema: int = 0
    relationships_off_schema: int = 0
    relationships_dropped_off_schema: int = 0
    relationships_dropped_off_schema_endpoint: int = 0
    relationships_off_pattern: int = 0
    relationships_dropped_off_pattern: int = 0


class SeedShard(_Canonical):
    """One shard document's path and text."""

    index: int = Field(description="Shard number, from 0")
    relative_path: str = Field(description="seed_graphs/<name>/<k>-of-<n>.jsonl")
    page_content: str = Field(description="Header line and unit lines")


class SeedConversion(_Canonical):
    """A converted seed graph: its shard documents and counts."""

    name: str
    shards: int = Field(description="The number of shards the seed is split into")
    filters: dict[str, SeedFilterValue] = Field(
        description="Document filters of its items (seed_graph included)"
    )
    documents_content: tuple[SeedShard, ...] = Field(
        description="The non-empty shards, by shard number"
    )
    stats: SeedConversionStats

    @property
    def digest(self) -> str:
        """SHA-256 (hex) of the shard paths and texts: equal for two
        conversions that give the same documents."""
        return seed_digest(self.documents_content)

    def documents(self) -> list[Document]:
        """New virtual documents of the shards (callers may edit them)."""
        return [
            seed_document(shard, self.name, self.filters)
            for shard in self.documents_content
        ]


# -- Rendering and parsing ----------------------------------------------------


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def render_statement(statement: SeedNodeStatement | SeedEdgeStatement) -> str:
    """The canonical JSON line of ``statement`` (unset fields left out)."""
    data = statement.model_dump(mode="json", exclude_none=True)
    if not data.get("properties"):
        data.pop("properties", None)
    return _canonical_json(data)


def parse_statement(line: str) -> SeedNodeStatement | SeedEdgeStatement:
    """The statement of a seed unit line.

    Raises:
        DataProcessingError: ``line`` is not a seed statement.
    """
    try:
        return _STATEMENT.validate_json(line)
    except ValueError as e:
        raise DataProcessingError(f"not a seed graph statement: {e}") from e


def render_unit(statements: Iterable[SeedNodeStatement | SeedEdgeStatement]) -> str:
    """The text of a seed unit: one line per statement."""
    return "\n".join(render_statement(statement) for statement in statements)


def parse_unit_statements(text: str) -> list[SeedNodeStatement | SeedEdgeStatement]:
    """The statements of a seed unit's text (:func:`render_unit`)."""
    # "\n" only: ensure_ascii=False lets U+2028 and friends through, which
    # str.splitlines would split on.
    return [parse_statement(line) for line in text.split("\n") if line]


def render_header(name: str, filters: Mapping[str, SeedFilterValue]) -> str:
    """The first line of a shard document: name, format and filters."""
    return _canonical_json(
        {"seed_graph": name, "format": SEED_FORMAT_VERSION, "filters": dict(filters)}
    )


def seed_digest(shards: Iterable[SeedShard]) -> str:
    """SHA-256 (hex) over the shards' paths and texts, in shard order."""
    payload = _canonical_json(
        [
            [shard.relative_path, shard.page_content]
            for shard in sorted(shards, key=lambda s: s.index)
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def seed_relative_path(name: str, index: int, shards: int) -> str:
    """``seed_graphs/<name>/<k>-of-<n>.jsonl`` of shard ``index``."""
    return f"{SEED_DOCUMENT_ROOT}/{name}/{index:04d}-of-{shards:04d}.jsonl"


def seed_document(
    shard: SeedShard, name: str, filters: Mapping[str, SeedFilterValue]
) -> Document:
    """The virtual document of a shard."""
    path = shard.relative_path
    return Document(
        page_content=shard.page_content,
        document_id=compute_document_id(path, compute_text_hash(shard.page_content)),
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
        metadata={
            "source_kind": SEED_SOURCE_KIND,
            SEED_FILTER_KEY: name,
            Constants.FILTERS.value: dict(filters),
            RELATIVE_PATH_KEY: path,
        },
    )


def is_seed_document(document: Document) -> bool:
    """Whether ``document`` is a seed graph's shard."""
    return document.metadata.get("source_kind") == SEED_SOURCE_KIND


# A shard document's relative path (seed_relative_path), as a registry
# record keeps it.
_SEED_PATH = re.compile(
    rf"^{SEED_DOCUMENT_ROOT}/[a-z0-9][a-z0-9_-]*/[0-9]+-of-[0-9]+\.jsonl$"
)


def is_seed_path(path: str | None) -> bool:
    """Whether ``path`` (a registry record's ``file_path``) names a seed
    graph's shard document."""
    return bool(path) and _SEED_PATH.match(path or "") is not None


def is_seed_unit(unit_id: str) -> bool:
    """Whether ``unit_id`` names a seed text unit (the one test anywhere)."""
    return unit_id.startswith(SEED_UNIT_PREFIX)


def seed_only(text_unit_ids: Iterable[str] | None) -> bool:
    """Whether an item citing ``text_unit_ids`` cites seed units only: some,
    and no document unit. An item citing no unit is not seed-only."""
    units = list(text_unit_ids or ())
    return bool(units) and all(is_seed_unit(unit_id) for unit_id in units)


_Cited = TypeVar("_Cited", Entity, Relationship)


def document_first(items: Sequence[_Cited]) -> list[_Cited]:
    """The ``items`` some document states (that are not :func:`seed_only`),
    or all of them when none is: where a merge picks one value (type, name,
    confidence), documents decide and seed statements only fill a gap. With
    no seed unit, every item."""
    stated = [item for item in items if not seed_only(item.text_unit_ids)]
    return stated or list(items)


def merged_iri(iris: Iterable[str | None]) -> tuple[str | None, bool]:
    """The IRI of an entity merged from statements giving ``iris``: the
    smallest one given (whatever their order), and whether they gave more
    than one."""
    given = {iri for iri in iris if iri}
    return (min(given) if given else None), len(given) > 1


def annotate_sources(item: Entity | Relationship) -> list[str] | None:
    """The ``source_kinds`` of ``item``: ``None`` when none of its text units
    is a seed unit (a document-only item), else the sorted kinds of its
    units (``["document", "seed_graph"]`` or ``["seed_graph"]``)."""
    units = item.text_unit_ids or []
    if not any(is_seed_unit(unit_id) for unit_id in units):
        return None
    kinds = {SEED_SOURCE_KIND}
    if not seed_only(units):
        kinds.add(DOCUMENT_SOURCE_KIND)
    return sorted(kinds)


def encoded_source_kinds(kinds: Iterable[str]) -> str:
    """``source_kinds`` as a graph edge stores it: the sorted kinds as a JSON
    list (the graph codec's encoding of a list property, ``json.dumps``)."""
    return json.dumps(sorted(kinds))


# The stored property naming an item's source kinds (Entity.source_kinds).
SOURCE_KINDS_FIELD = "source_kinds"
# The stored ``source_kinds`` of a seed-only edge: a graph read of the
# document-backed edges leaves out exactly the edges holding this value
# (an edge without the property is document-only).
SEED_ONLY_SOURCE_KINDS = encoded_source_kinds([SEED_SOURCE_KIND])


def is_document_backed(relationship: Relationship) -> bool:
    """Whether ``relationship`` counts toward ranks: unless every one of its
    text units is a seed unit (see :func:`seed_only`)."""
    return not seed_only(relationship.text_unit_ids)


def seed_unit_id(document_id: str, n: int) -> str:
    """Id of unit ``n`` of a shard document: the chunker's rule, prefixed."""
    return SEED_UNIT_PREFIX + generate_stable_id(f"text_unit:{document_id}:{n}")


def _placement(values: Mapping[str, Any]) -> dict[str, Any]:
    """The index suffix and document filters of a document or unit."""
    keys: tuple[str, str] = (Constants.INDEX.value, Constants.FILTERS.value)
    return {key: values[key] for key in keys if key in values}


def seed_text_units(
    document: Document, *, count_tokens: Callable[[str], int] | None = None
) -> list[TextUnit]:
    """The text units of a shard document, in order (a pure function of it).

    ``count_tokens`` sets ``n_tokens`` (the chunker's estimate); ``None``
    leaves it unset.

    Raises:
        DataProcessingError: The document is not a seed shard.
    """
    header, *units = document.page_content.split(_UNIT_SEPARATOR)
    try:
        parsed = json.loads(header)
    except ValueError:
        parsed = None
    if not isinstance(parsed, dict) or SEED_FILTER_KEY not in parsed:
        raise DataProcessingError(
            f"document '{document.file_path}' is not a seed graph shard"
        )
    placement = _placement(document.metadata)
    result = []
    for n, text in enumerate(units):
        unit_id = seed_unit_id(document.document_id, n)
        result.append(
            TextUnit(
                id=unit_id,
                short_id=unit_id[len(SEED_UNIT_PREFIX) :][:8],
                text=text,
                document_ids=[document.document_id],
                entity_ids=[],
                relationship_ids=[],
                community_ids=[],
                n_tokens=count_tokens(text) if count_tokens else None,
                attributes={
                    "file_name": document.file_name,
                    "file_path": document.file_path,
                    "chunk_id": n,
                    "total_chunks": len(units),
                    "chunking_method": SEED_SOURCE_KIND,
                    "source_kind": SEED_SOURCE_KIND,
                    **placement,
                },
            )
        )
    return result


def _item_attributes(
    properties: Mapping[str, SeedPropertyValue], placement: Mapping[str, Any]
) -> dict[str, Any] | None:
    attributes: dict[str, Any] = {}
    if properties:
        attributes[SEED_PROPERTIES_ATTRIBUTE] = dict(properties)
    attributes.update(placement)
    return attributes or None


def parse_seed_unit(unit: TextUnit) -> TextUnitExtraction:
    """The extraction record of a seed unit: a pure function of its text,
    stamped with the unit's id and placement (index, filters) as
    ``contributions.record_from_payload`` stamps a stored record.

    Ids are the extractor's (``BaseProcessor``), so a seed node and a
    document mention with the same name key are one entity.

    Raises:
        DataProcessingError: The text is not seed statements.
    """
    placement = _placement(unit.attributes or {})
    entities: list[Entity] = []
    relationships: list[Relationship] = []
    for statement in parse_unit_statements(unit.text):
        if isinstance(statement, SeedNodeStatement):
            entity_id = BaseProcessor._generate_entity_id(statement.name)
            entities.append(
                Entity(
                    id=entity_id,
                    short_id=entity_id[:8],
                    name=statement.name,
                    type=statement.type or "",
                    description=statement.description or "",
                    text_unit_ids=[unit.id],
                    confidence=statement.confidence,
                    iri=statement.iri,
                    attributes=_item_attributes(statement.properties, placement),
                )
            )
            continue
        rel_id = BaseProcessor._generate_relationship_id(
            statement.source, statement.target, statement.type
        )
        rel = Relationship(
            id=rel_id,
            short_id=rel_id[:8],
            source_id=BaseProcessor._generate_entity_id(statement.source),
            source_name=statement.source,
            target_id=BaseProcessor._generate_entity_id(statement.target),
            target_name=statement.target,
            type=statement.type,
            weight=statement.weight,
            description=statement.description or "",
            text_unit_ids=[unit.id],
            attributes=_item_attributes(statement.properties, placement),
        )
        apply_text_unit_weights(rel, {unit.id: statement.weight})
        relationships.append(rel)
    return TextUnitExtraction(
        text_unit_id=unit.id, entities=entities, relationships=relationships
    )


# -- Conversion ---------------------------------------------------------------


@dataclass
class _Node:
    id: str
    name: str
    type: str | None
    iri: str | None
    description: str | None
    confidence: float
    properties: dict[str, SeedPropertyValue]


@dataclass
class _Edge:
    source: str
    target: str
    type: str
    predicate_iri: str | None
    description: str | None
    weight: float
    properties: dict[str, SeedPropertyValue]


def shard_of(node_id: str, shards: int) -> int:
    """The shard of a node: a stable hash of its id."""
    digest = hashlib.sha256(node_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % shards


def _cut(text: str, limit: int) -> tuple[str, bool]:
    """``text`` with blank lines dropped and spaces collapsed per line, cut
    to ``limit`` characters at a word boundary (mid-word only when one word
    is longer) with "…"; and whether it was cut."""
    text = "\n".join(
        " ".join(line.split()) for line in text.split("\n") if line.strip()
    )
    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    if not text[limit].isspace() and re.search(r"\s", cut):
        cut = cut[: max(cut.rfind(" "), cut.rfind("\n"))]
    return cut.rstrip() + "…", True


def _property_text(value: SeedPropertyValue) -> str:
    return value if isinstance(value, str) else _canonical_json(value)


def _local_name(iri: str) -> str:
    return re.split(r"[#/:]", iri.rstrip("#/"))[-1] or iri


class _Converter:
    def __init__(
        self,
        graph: SeedGraph,
        settings: SeedConversionSettings,
        schema: ExtractionSchema,
    ) -> None:
        self.graph = graph
        self.settings = settings
        self.conformance = SchemaConformance(schema)
        self.stats: Counter[str] = Counter(
            nodes_read=len(graph.nodes), edges_read=len(graph.edges)
        )
        # IRI -> declared label; a later declaration wins (an owl schema
        # lists types in preorder, so the last is the most specific).
        self.entity_iris = {
            spec.iri: spec.label for spec in schema.entity_types if spec.iri
        }
        self.relationship_iris = {
            spec.iri: spec.label for spec in schema.relationship_types if spec.iri
        }

    def fail(self, message: str) -> SeedGraphError:
        return SeedGraphError(f"seed graph {self.settings.name!r}: {message}")

    # 1. Checks

    def check_caps(self) -> None:
        settings, graph = self.settings, self.graph
        if len(graph.nodes) > settings.max_nodes:
            raise self.fail(
                f"{len(graph.nodes)} nodes exceed max_nodes ({settings.max_nodes})"
            )
        if len(graph.edges) > settings.max_edges:
            raise self.fail(
                f"{len(graph.edges)} edges exceed max_edges ({settings.max_edges})"
            )
        keys = {key for item in (*graph.nodes, *graph.edges) for key in item.properties}
        if len(keys) > settings.max_property_keys:
            raise self.fail(
                f"{len(keys)} distinct property keys exceed max_property_keys "
                f"({settings.max_property_keys})"
            )

    def unique_nodes(self) -> dict[str, SeedNode]:
        nodes: dict[str, SeedNode] = {}
        duplicates: set[str] = set()
        for node in self.graph.nodes:
            if node.id in nodes:
                duplicates.add(node.id)
            nodes[node.id] = node
        if duplicates:
            raise self.fail(f"duplicate node ids {_listed(duplicates)}")
        return nodes

    def resolved_edges(self, nodes: Mapping[str, SeedNode]) -> list[SeedEdge]:
        edges: list[SeedEdge] = []
        dangling: set[str] = set()
        for edge in self.graph.edges:
            unknown = {edge.source, edge.target} - nodes.keys()
            if unknown:
                dangling |= unknown
                self.stats["dangling_edges_skipped"] += 1
                continue
            if edge.source == edge.target:
                self.stats["self_loops_dropped"] += 1
                continue
            edges.append(edge)
        if dangling and self.settings.dangling == "error":
            raise self.fail(f"edges name unknown nodes {_listed(dangling)}")
        return edges

    # 2. Types

    def node(self, node: SeedNode) -> _Node:
        label = None
        if node.type_iri:
            label = self.entity_iris.get(node.type_iri)
            if label is None:
                self.stats["nodes_unmapped_type_iri"] += 1
        if label is None and node.type and node.type.strip():
            label = self.conformance.entity_type(node.type.strip())
        confidence = (
            node.confidence if node.confidence is not None else self.settings.confidence
        )
        return _Node(
            id=node.id,
            name=node.name,
            type=label,
            iri=node.iri,
            description=node.description,
            confidence=confidence,
            properties=dict(node.properties),
        )

    def edge(self, edge: SeedEdge) -> _Edge:
        label = None
        if edge.predicate_iri:
            label = self.relationship_iris.get(edge.predicate_iri)
            if label is None:
                self.stats["edges_unmapped_predicate_iri"] += 1
        if label is None and edge.type and edge.type.strip():
            label = self.conformance.relationship_type(edge.type.strip())
        if label is None:
            raise self.fail(
                f"the edge from {edge.source!r} to {edge.target!r} has no type "
                "(type, or a predicate_iri a relationship type declares)"
            )
        return _Edge(
            source=edge.source,
            target=edge.target,
            type=label,
            predicate_iri=edge.predicate_iri,
            description=edge.description,
            weight=edge.weight,
            properties=dict(edge.properties),
        )

    def merged_edges(self, edges: list[_Edge]) -> list[_Edge]:
        groups: dict[tuple[str, str, str], list[_Edge]] = defaultdict(list)
        for edge in edges:
            groups[(edge.source, edge.target, schema_key(edge.type))].append(edge)
        merged: list[_Edge] = []
        for group in groups.values():
            # A canonical order, so the merge does not depend on the input's.
            group.sort(key=lambda e: _canonical_json(vars(e)))
            self.stats["duplicate_edges_merged"] += len(group) - 1
            first = group[0]
            iris = [e.predicate_iri for e in group if e.predicate_iri]
            properties: dict[str, SeedPropertyValue] = {}
            for e in group:
                properties.update(e.properties)
            merged.append(
                _Edge(
                    source=first.source,
                    target=first.target,
                    type=min(e.type for e in group),
                    predicate_iri=min(iris) if iris else None,
                    description=merge_descriptions(e.description for e in group),
                    weight=math.fsum(e.weight for e in group),
                    properties=properties,
                )
            )
        return merged

    # 3. Conformance

    def conform(
        self, nodes: dict[str, _Node], edges: list[_Edge]
    ) -> tuple[dict[str, _Node], list[_Edge]]:
        entities = [
            Entity(id=node.id, name=node.name, type=node.type or "")
            for node in nodes.values()
        ]
        relationships = [
            Relationship(
                id=str(position),
                source_id=edge.source,
                source_name=nodes[edge.source].name,
                target_id=edge.target,
                target_name=nodes[edge.target].name,
                type=edge.type,
            )
            for position, edge in enumerate(edges)
        ]
        result = self.conformance.conform(
            entities, relationships, unit=f"seed graph '{self.settings.name}'"
        )
        for counter in SCHEMA_COUNTERS:
            self.stats[counter] += getattr(result, counter)
        kept_nodes = {entity.id: nodes[entity.id] for entity in result.entities}
        kept_edges = []
        for rel in result.relationships:
            edge = edges[int(rel.id)]
            # A node strict dropped whose name another, kept node states is
            # still dropped, and so are its edges.
            if edge.source in kept_nodes and edge.target in kept_nodes:
                kept_edges.append(edge)
            else:
                self.stats["relationships_dropped_off_schema_endpoint"] += 1
        return kept_nodes, kept_edges

    # 4. Name collisions

    def collisions(self, nodes: Iterable[_Node]) -> list[list[list[_Node]]]:
        """The name keys several distinct nodes share: per key, the nodes
        grouped by IRI (a node without one alone), by smallest id."""
        by_key: dict[str, dict[str, list[_Node]]] = defaultdict(dict)
        for node in nodes:
            identity = f"iri:{node.iri}" if node.iri else f"id:{node.id}"
            by_key[entity_key(node.name)].setdefault(identity, []).append(node)
        found = []
        for _, groups in sorted(by_key.items()):
            if len(groups) > 1:
                parts = [sorted(g, key=lambda n: n.id) for g in groups.values()]
                found.append(sorted(parts, key=lambda g: g[0].id))
        return found

    def resolve_names(self, nodes: dict[str, _Node]) -> None:
        collisions = self.collisions(nodes.values())
        if not collisions:
            return
        policy = self.settings.on_name_collision
        if policy == "error":
            raise self.fail(
                f"{len(collisions)} names are shared by distinct nodes (ids "
                + "; ".join(
                    ", ".join(node.id for part in parts for node in part)
                    for parts in collisions[:5]
                )
                + (" ..." if len(collisions) > 5 else "")
                + "); set on_name_collision to merge or qualify"
            )
        if policy == "merge":
            self.stats["name_collisions_merged"] += len(collisions)
            return
        for parts in collisions:
            for part in parts[1:]:
                for node in part:
                    node.name = f"{node.name} ({self.qualifier(node)})"
                    self.stats["names_qualified"] += 1
        if again := self.collisions(nodes.values()):
            raise self.fail(
                f"{len(again)} qualified names still collide (ids "
                + "; ".join(
                    ", ".join(node.id for part in parts for node in part)
                    for parts in again[:5]
                )
                + ")"
            )

    def qualifier(self, node: _Node) -> str:
        prop = self.settings.qualifier_property
        if prop and prop in node.properties:
            value = _property_text(node.properties[prop]).strip()
            if value:
                return " ".join(value.split())
        return _local_name(node.iri) if node.iri else node.id

    # 5-6. Statements, shards and units

    def description(
        self, text: str | None, properties: Mapping[str, Any]
    ) -> str | None:
        lines = [text] if text else []
        lines += [
            f"{key}: {_property_text(properties[key])}"
            for key in self.settings.description_properties
            if key in properties
        ]
        if not lines:
            return None
        cut, truncated = _cut("\n".join(lines), self.settings.max_description_chars)
        self.stats["descriptions_truncated"] += truncated
        return cut or None

    def node_statement(self, node: _Node) -> SeedNodeStatement:
        return SeedNodeStatement(
            id=node.id,
            name=node.name,
            type=node.type,
            iri=node.iri,
            description=self.description(node.description, node.properties),
            confidence=node.confidence,
            properties=node.properties,
        )

    def edge_statement(
        self, edge: _Edge, nodes: Mapping[str, _Node]
    ) -> SeedEdgeStatement:
        return SeedEdgeStatement(
            source=nodes[edge.source].name,
            target=nodes[edge.target].name,
            type=edge.type,
            predicate_iri=edge.predicate_iri,
            description=self.description(edge.description, edge.properties),
            weight=edge_weight(edge.weight),
            properties=edge.properties,
        )

    def pack(self, cards: list[list[str]]) -> list[str]:
        """Units of at most ``max_items_per_unit`` lines: cards in order, a
        card that does not fit the current unit starting the next, a card
        larger than a unit split into full units."""
        limit = self.settings.max_items_per_unit
        units: list[list[str]] = []
        current: list[str] = []
        for card in cards:
            if len(current) + len(card) <= limit:
                current += card
                continue
            if current:
                units.append(current)
            current = []
            for start in range(0, len(card), limit):
                piece = card[start : start + limit]
                if len(piece) == limit:
                    units.append(piece)
                else:
                    current = piece
        if current:
            units.append(current)
        return ["\n".join(unit) for unit in units]

    def convert(self) -> SeedConversion:
        settings = self.settings
        self.check_caps()
        seed_nodes = self.unique_nodes()
        seed_edges = self.resolved_edges(seed_nodes)
        nodes = {node_id: self.node(node) for node_id, node in seed_nodes.items()}
        edges = self.merged_edges([self.edge(edge) for edge in seed_edges])
        nodes, edges = self.conform(nodes, edges)
        self.resolve_names(nodes)
        kept = []
        for edge in edges:
            if entity_key(nodes[edge.source].name) == entity_key(
                nodes[edge.target].name
            ):
                # Two merged nodes: one entity, which a full build relates
                # to itself nowhere.
                self.stats["self_loops_dropped"] += 1
            else:
                kept.append(edge)
        outgoing: dict[str, list[_Edge]] = defaultdict(list)
        for edge in kept:
            outgoing[edge.source].append(edge)
        by_shard: dict[int, list[_Node]] = defaultdict(list)
        for node in nodes.values():
            by_shard[shard_of(node.id, settings.shards)].append(node)
        header = render_header(settings.name, settings.document_filters)
        shards: list[SeedShard] = []
        for index in sorted(by_shard):
            cards = []
            for node in sorted(
                by_shard[index], key=lambda n: (entity_key(n.name), n.id)
            ):
                card_edges = sorted(
                    outgoing[node.id],
                    key=lambda e: (
                        entity_key(nodes[e.target].name),
                        schema_key(e.type),
                        e.target,
                    ),
                )
                cards.append(
                    [render_statement(self.node_statement(node))]
                    + [
                        render_statement(self.edge_statement(edge, nodes))
                        for edge in card_edges
                    ]
                )
            units = self.pack(cards)
            if len(units) > MAX_UNITS_PER_SHARD:
                raise self.fail(
                    f"shard {index} would hold {len(units)} text units, over "
                    f"{MAX_UNITS_PER_SHARD}: raise shards (now {settings.shards}) "
                    "or max_items_per_unit"
                )
            self.stats["units"] += len(units)
            shards.append(
                SeedShard(
                    index=index,
                    relative_path=seed_relative_path(
                        settings.name, index, settings.shards
                    ),
                    page_content=_UNIT_SEPARATOR.join([header, *units]),
                )
            )
        self.stats["nodes"] = len(nodes)
        self.stats["edges"] = len(kept)
        self.stats["documents"] = len(shards)
        stats = SeedConversionStats(**self.stats)
        logger.info(
            "Seed graph '%s': %s",
            settings.name,
            ", ".join(f"{key}={value}" for key, value in stats if value),
        )
        return SeedConversion(
            name=settings.name,
            shards=settings.shards,
            filters=settings.document_filters,
            documents_content=tuple(shards),
            stats=stats,
        )


def _listed(values: Iterable[str], limit: int = 5) -> str:
    ordered = sorted(values)
    shown = ", ".join(repr(v) for v in ordered[:limit])
    return shown + (f" and {len(ordered) - limit} more" if len(ordered) > limit else "")


def convert_seed(
    graph: SeedGraph, settings: SeedConversionSettings, schema: ExtractionSchema
) -> SeedConversion:
    """The virtual documents of ``graph`` under ``settings`` and ``schema``
    (see the module docstring). Deterministic: the same seed, in any order,
    gives the same documents.

    Raises:
        SeedGraphError: The seed breaks a rule of the import (a duplicate
            node id, a dangling edge under ``dangling: error``, an edge
            without a type, a name collision under ``on_name_collision:
            error``, a cap).
    """
    return _Converter(graph, settings, schema).convert()


def seed_statements(conversion: SeedConversion) -> Sequence[str]:
    """Every statement line of a conversion, in document order."""
    return [
        line
        for shard in conversion.documents_content
        for unit in shard.page_content.split(_UNIT_SEPARATOR)[1:]
        for line in unit.split("\n")
        if line
    ]
