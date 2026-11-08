# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller filters as one backend-neutral form, and the fields they name.

``SearchQuery.filters`` is a mapping of field to value: a scalar (equality),
a list (any of its values), a mapping of range bounds (``gt``/``gte``/
``lt``/``lte``), or for ``source_kinds`` the kinds an item must hold
evidence of. :func:`parse_filters` checks and types it into clauses
(:class:`Term`, :class:`Range`, :class:`KindsRule`; :class:`AnyOf` for
``SearchQuery.any_of``), each with the field's :class:`FieldKind`, and each
store compiles the clauses into its own language: the OpenSearch query DSL
(OpenSearch and the embedded engine), Gremlin steps and the SQLite graph's
vertex predicate. A compiler refuses (``InvalidFilterError``) a clause its
store cannot apply rather than matching something else.

The field catalog (:data:`FIELD_KINDS` and the per-store
:class:`FilterFields`) is the one statement of what each store's items hold
and how a filter on a field matches, for every backend:

- ``attr_<key>`` and ``attributes.filters.<key>`` are entry ``<key>`` of an
  item's documents' ``filters`` metadata (:data:`FieldKind.DOCUMENT`). Every
  index carries them; a graph vertex holds none (the strategies check a
  graph's entities against the entities index for them).
- ``attributes.<path>`` is an item's own attribute (an entity's extracted
  ``role``; :data:`FieldKind.ATTRIBUTE`). The entities index holds every
  path of it; a graph vertex a top-level one, as ``attr_<key>``.
- ``attributes.seed_properties...`` is kept unindexed: refused.
- ``name`` is an entity's exact name; analyzed text (``description``,
  ``text``, ...) is refused: a term on it matched single analyzed tokens on
  OpenSearch and the whole text on a graph.
- the schema annotations and the numbers are typed: a string value is
  converted (a command line gives every value as one).
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from unified_kg_rag.domain.ingestion.base_processor import SEED_PROPERTIES_ATTRIBUTE
from unified_kg_rag.domain.ingestion.seed_graph import (
    DOCUMENT_SOURCE_KIND,
    SEED_SOURCE_KIND,
    SOURCE_KINDS_FIELD,
    encoded_source_kinds,
)
from unified_kg_rag.domain.models import Constants
from unified_kg_rag.shared.exceptions import InvalidFilterError

ATTRIBUTE_KEY_PREFIX = f"{Constants.ATTRIBUTE_PREFIX.value}_"
ATTRIBUTES_OBJECT_PREFIX = "attributes."
DOCUMENT_FILTERS_PREFIX = f"{ATTRIBUTES_OBJECT_PREFIX}{Constants.FILTERS.value}."
SEED_PROPERTIES_PREFIX = f"{ATTRIBUTES_OBJECT_PREFIX}{SEED_PROPERTIES_ATTRIBUTE}"
# The exact copy of an analyzed name (the entities index's sub-field).
EXACT_SUBFIELD = ".keyword"
TRANSLATED_TEXT_PREFIX = "translated_text_"

# The source kinds of an entity or relationship (``Entity.source_kinds``):
# written only on items a seed graph states, so an item without the field
# holds document evidence only. A filter on it names kinds the item must hold
# evidence of: ``seed_graph`` matches the items a seed graph states,
# ``document`` the items a document states, those without the field included.
SOURCE_KINDS = frozenset({DOCUMENT_SOURCE_KIND, SEED_SOURCE_KIND})


class FieldKind(StrEnum):
    """How a filter on a field matches, on every store."""

    # Exact equality of the stored value.
    KEYWORD = "keyword"
    # An entity's exact name (``name.keyword`` on the entities index).
    NAME = "name"
    # Analyzed text: refused.
    TEXT = "text"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"
    # ``source_kinds``: the kinds an item holds evidence of (``KindsRule``).
    SOURCE_KINDS = "source_kinds"
    # A document's ``filters`` entry: a keyword, a number or boolean held as
    # its JSON text (``attr_year: 2023`` matches ``"2023"``).
    DOCUMENT = "document"
    # An item's own attribute, mapped by its first value on an index.
    ATTRIBUTE = "attribute"
    # Kept, never indexed: refused.
    UNINDEXED = "unindexed"


_TYPED = frozenset({FieldKind.BOOLEAN, FieldKind.INTEGER, FieldKind.NUMBER})
# The keyword fields ingestion writes lowercase (an entity's and a claim's
# type, a claim's status: ``str.lower``), so a filter value matches its
# lowercase form too: ``type: PERSON`` matched nothing. The value as given
# still matches what a caller wrote to a store itself.
_LOWERCASE_FIELDS = frozenset({"type", "status"})
_REFUSED = frozenset({FieldKind.TEXT, FieldKind.UNINDEXED})

# Every fixed field a store writes on an item, by how a filter on it
# matches. The OpenSearch mappings (OpenSearchIndexer._get_*_mapping) and
# the graph codec (neptune_codec.entity_properties) write these; unit tests
# pin the two together. A field not listed (a scope key such as a text
# unit's ``source_kind``) is a keyword.
FIELD_KINDS: Mapping[str, FieldKind] = {
    **dict.fromkeys(
        (
            "id",
            "type",
            "text_unit_ids",
            "community_ids",
            "type_iri",
            "predicate_iri",
            "iri",
            "source_id",
            "target_id",
            "subject_id",
            "object_id",
            "status",
            "start_date",
            "end_date",
            "child_ids",
            "language",
            "partition",
            "community_id",
            "document_ids",
            "parent_community_id",
            "child_community_ids",
        ),
        FieldKind.KEYWORD,
    ),
    "name": FieldKind.NAME,
    f"name{EXACT_SUBFIELD}": FieldKind.NAME,
    **dict.fromkeys(
        (
            "text",
            "description",
            "source_name",
            "target_name",
            "subject_name",
            "object_name",
            "source_text",
            "summary",
            "full_content",
        ),
        FieldKind.TEXT,
    ),
    "in_schema": FieldKind.BOOLEAN,
    "lineage_complete": FieldKind.BOOLEAN,
    **dict.fromkeys(("rank", "confidence", "weight", "rating"), FieldKind.NUMBER),
    **dict.fromkeys(
        ("n_tokens", "n_leaves", "level", "depth", "max_cut_depth"),
        FieldKind.INTEGER,
    ),
    SOURCE_KINDS_FIELD: FieldKind.SOURCE_KINDS,
}


def field_kind(key: str) -> FieldKind:
    """How a filter on ``key`` matches (:data:`FIELD_KINDS`, then the
    prefixes: ``translated_text_<language>`` is text, ``attr_<key>`` and
    ``attributes.filters.<key>`` document filters, the seed properties
    unindexed, any other ``attributes.<path>`` an item's own attribute); a
    key the catalog does not know is a keyword."""
    kind = FIELD_KINDS.get(key)
    if kind is not None:
        return kind
    if key.startswith(TRANSLATED_TEXT_PREFIX):
        return FieldKind.TEXT
    if key.startswith((ATTRIBUTE_KEY_PREFIX, DOCUMENT_FILTERS_PREFIX)):
        return FieldKind.DOCUMENT
    if key == SEED_PROPERTIES_PREFIX or key.startswith(f"{SEED_PROPERTIES_PREFIX}."):
        return FieldKind.UNINDEXED
    if key.startswith(ATTRIBUTES_OBJECT_PREFIX):
        return FieldKind.ATTRIBUTE
    return FieldKind.KEYWORD


# Entity attributes a graph vertex holds as JSON or bookkeeping.
_GRAPH_INTERNAL_ATTRIBUTES = frozenset(
    {Constants.FILTERS.value, Constants.INDEX.value, SEED_PROPERTIES_ATTRIBUTE}
)


def own_attribute(key: str) -> str | None:
    """The attribute an ``attributes.<key>`` filter names when it is a
    top-level attribute of the item's own (``role`` of ``attributes.role``),
    else ``None``: a nested path, a document filter, an internal attribute
    (the suffix, the seed properties, ``_``-prefixed bookkeeping)."""
    if field_kind(key) is not FieldKind.ATTRIBUTE:
        return None
    name = key[len(ATTRIBUTES_OBJECT_PREFIX) :]
    if (
        not name
        or "." in name
        or name.startswith("_")
        or name in _GRAPH_INTERNAL_ATTRIBUTES
    ):
        return None
    return name


@dataclass(frozen=True)
class FilterFields:
    """The filter keys one store (or a union of stores) applies: its fixed
    ``fields`` and the ``prefixes`` of its dynamic ones. With
    ``nested_attributes`` off (a graph vertex), ``attributes.<path>``
    declares only an item's own top-level attribute (``own_attribute``).
    A refused kind (text, unindexed) is never declared."""

    fields: frozenset[str]
    prefixes: frozenset[str] = frozenset()
    nested_attributes: bool = True

    def declares(self, key: str) -> bool:
        if field_kind(key) in _REFUSED:
            return False
        if key in self.fields:
            return True
        if not any(key.startswith(p) for p in self.prefixes):
            return False
        # A vertex holds only an item's own top-level attributes.
        return self.nested_attributes or own_attribute(key) is not None

    def describe(self) -> list[str]:
        """Sorted keys, with each prefix rendered as ``<prefix><key>``."""
        return sorted(self.fields) + sorted(f"{p}<key>" for p in self.prefixes)


def union_filter_fields(items: Iterable[FilterFields]) -> FilterFields:
    """The keys any of ``items`` declares."""
    listed = list(items)
    return FilterFields(
        frozenset(f for item in listed for f in item.fields),
        frozenset(p for item in listed for p in item.prefixes),
        any(item.nested_attributes for item in listed) or not listed,
    )


# Document filters (``document_filters``): every index carries them.
_DOCUMENT_PREFIXES = frozenset({ATTRIBUTE_KEY_PREFIX, DOCUMENT_FILTERS_PREFIX})
# Entities also carry their own attributes (``attributes.role``), which no
# other index holds: such a key leaves the other indexes unfiltered, as
# ``type`` does, instead of emptying them.
_ENTITY_PREFIXES = frozenset({ATTRIBUTE_KEY_PREFIX, ATTRIBUTES_OBJECT_PREFIX})

# Item kind (an index of the vector store) -> the filter keys its documents
# hold. Analyzed text is no key (refused); neither is a report's ``name``,
# which has no exact copy, nor the fields only a scope reads.
ITEM_FILTER_FIELDS: Mapping[str, FilterFields] = {
    "text_units": FilterFields(
        frozenset({"id", "community_ids", "n_tokens"}), _DOCUMENT_PREFIXES
    ),
    "entities": FilterFields(
        frozenset(
            {
                "id",
                "name",
                f"name{EXACT_SUBFIELD}",
                "type",
                "rank",
                "confidence",
                "text_unit_ids",
                "in_schema",
                "type_iri",
                "iri",
                SOURCE_KINDS_FIELD,
            }
        ),
        _ENTITY_PREFIXES,
    ),
    "relationships": FilterFields(
        frozenset(
            {
                "id",
                "source_id",
                "target_id",
                "weight",
                "rank",
                "text_unit_ids",
                # No ``type``: a relationship's type is not an entity type,
                # and ``type: PERSON`` would empty this index under mix and
                # hybrid.
                "in_schema",
                "predicate_iri",
                SOURCE_KINDS_FIELD,
            }
        ),
        _DOCUMENT_PREFIXES,
    ),
    "claims": FilterFields(
        frozenset(
            {
                "id",
                "subject_id",
                "object_id",
                "type",
                "status",
                "start_date",
                "end_date",
                "text_unit_ids",
            }
        ),
        _DOCUMENT_PREFIXES,
    ),
    # The summary tree's nodes hold a document key only when all their
    # documents share its value: fail-closed.
    "summary_nodes": FilterFields(
        frozenset(
            {
                "id",
                "level",
                "text_unit_ids",
                "child_ids",
                "n_leaves",
                "n_tokens",
                "lineage_complete",
                "language",
                "partition",
            }
        ),
        _DOCUMENT_PREFIXES,
    ),
    "community_reports": FilterFields(
        frozenset(
            {
                "id",
                "community_id",
                "rank",
                "rating",
                "text_unit_ids",
                "document_ids",
                "level",
                "depth",
                "max_cut_depth",
                "parent_community_id",
                "child_community_ids",
            }
        ),
        _DOCUMENT_PREFIXES,
    ),
}

# The filter keys of a graph's entity vertices, the only vertices a graph
# filters: the entity fields a vertex stores (its exact name under both
# spellings) and its own top-level attributes. A vertex holds no document
# filter and no nested attribute path.
GRAPH_ENTITY_FILTER_FIELDS = FilterFields(
    frozenset(
        {
            "id",
            "name",
            f"name{EXACT_SUBFIELD}",
            "type",
            "rank",
            "confidence",
            "text_unit_ids",
            "community_ids",
            "in_schema",
            "type_iri",
            "iri",
            SOURCE_KINDS_FIELD,
        }
    ),
    frozenset({ATTRIBUTES_OBJECT_PREFIX}),
    nested_attributes=False,
)


# --- values -------------------------------------------------------------------

Scalar = str | int | float | bool
_SCALARS = (str, int, float, bool)
RANGE_BOUNDS = frozenset({"gt", "gte", "lt", "lte"})


def check_filter_values(
    filters: Mapping[str, Any] | None, *, allow_empty_lists: bool = False
) -> None:
    """Reject filter keys and values no store applies as documented.

    A key naming analyzed text or an unindexed path is refused
    (``FieldKind.TEXT``/``UNINDEXED``). A value is a scalar, a list of
    scalars or a mapping of range bounds (``gt``/``gte``/``lt``/``lte`` to a
    scalar). ``None`` and NaN anywhere, a tuple or set, a nested list or
    mapping, and an empty range are rejected: OpenSearch answers them with a
    400 and the embedded engine would match them instead (a null field,
    say). An empty list is rejected too (a caller's filters): OpenSearch
    refuses a search with query text under one, the embedded engine returns
    nothing. A retriever passes ``allow_empty_lists``: a strategy's own scope
    may be one (``scope_filters``: nothing the caller allows), which only a
    text-less fetch by id carries.

    Raises:
        InvalidFilterError: A key is refused or a value malformed.
    """
    for key in filters or {}:
        kind = field_kind(key)
        if kind is FieldKind.TEXT:
            raise InvalidFilterError(
                f"Filter '{key}' names an analyzed text field, which no store "
                "matches exactly; put the words in the query, or filter an "
                "exact field (id, name, type, an attribute)."
            )
        if kind is FieldKind.UNINDEXED:
            raise InvalidFilterError(
                f"Filter '{key}' names the seed properties, which are kept "
                "unindexed; no store can filter them."
            )

    def bad(value: Any) -> str | None:
        if value is None:
            return "has no value (null)"
        if isinstance(value, float) and value != value:
            return "is NaN"
        if not isinstance(value, _SCALARS):
            return "is not a scalar (a string, number or boolean)"
        return None

    for key, value in (filters or {}).items():
        reason: str | None
        if value is None:
            reason = "has no value (null); give the value to match"
        elif isinstance(value, list):
            if not value and not allow_empty_lists:
                reason = "is an empty list; give at least one value"
            else:
                reason = next(
                    (f"lists a value that {r}" for r in map(bad, value) if r), None
                )
        elif isinstance(value, dict):
            if not value:
                reason = "is an empty range; give gt, gte, lt or lte"
            elif unknown := sorted(set(value) - RANGE_BOUNDS):
                reason = (
                    f"has a range bound '{unknown[0]}'; a range takes gt, gte, "
                    "lt and lte"
                )
            else:
                reason = next(
                    (
                        f"has a range bound that {r}"
                        for r in map(bad, value.values())
                        if r
                    ),
                    None,
                )
        else:
            reason = bad(value)
        if reason:
            raise InvalidFilterError(f"Filter '{key}' {reason}.")


_TRUE = frozenset({"true", "1", "yes"})
_FALSE = frozenset({"false", "0", "no"})


def _shown(value: bool | float) -> str:
    """``value`` as a filter names it (JSON, as a query file or ``--filter``
    gives it): ``true``, not Python's ``True``."""
    return json.dumps(value)


def _number(key: str, kind: FieldKind, number: float, given: str) -> int | float:
    """``number`` as a ``kind`` (number or integer) value; ``given`` is how
    the error names the value."""
    if isinstance(number, int):
        return number
    if not math.isfinite(number) or (
        kind is FieldKind.INTEGER and not number.is_integer()
    ):
        expected = "an integer" if kind is FieldKind.INTEGER else "a number"
        raise InvalidFilterError(
            f"Filter '{key}' is {expected} field; give {expected}, not {given}."
        )
    return int(number) if kind is FieldKind.INTEGER else number


def _typed(key: str, kind: FieldKind, value: Any) -> Any:
    """``value`` as a ``kind`` value: a string converted, a number or
    boolean checked against the type (``True`` is no number, ``1`` no
    boolean, ``2.5`` no integer); any other value unchanged, for
    :func:`check_filter_values` to judge."""
    if isinstance(value, bool):
        if kind is FieldKind.BOOLEAN:
            return value
        expected = "an integer" if kind is FieldKind.INTEGER else "a number"
        raise InvalidFilterError(
            f"Filter '{key}' is {expected} field; give {expected}, not "
            f"{_shown(value)}."
        )
    if isinstance(value, int | float):
        if kind is FieldKind.BOOLEAN:
            raise InvalidFilterError(
                f"Filter '{key}' is a boolean field; give true or false, "
                f"not {_shown(value)}."
            )
        return _number(key, kind, value, _shown(value))
    if not isinstance(value, str):
        return value
    text = value.strip()
    if kind is FieldKind.BOOLEAN:
        if text.lower() in _TRUE:
            return True
        if text.lower() in _FALSE:
            return False
        raise InvalidFilterError(
            f"Filter '{key}' is a boolean field; give true or false "
            f"(or 1/0, yes/no), not '{value}'."
        )
    try:
        return int(text)
    except ValueError:
        pass
    try:
        number = float(text)
    except ValueError:
        number = math.nan
    return _number(key, kind, number, f"'{value}'")


def _source_kind(value: Any) -> Any:
    """``value`` as a source kind (any case, spaces stripped); ``None`` is
    left for :func:`check_filter_values` to judge.

    Raises:
        InvalidFilterError: ``value`` names no source kind.
    """
    if value is None:
        return value
    kind = value.strip().lower() if isinstance(value, str) else None
    if kind not in SOURCE_KINDS:
        raise InvalidFilterError(
            f"Filter '{SOURCE_KINDS_FIELD}' takes "
            f"{' or '.join(sorted(SOURCE_KINDS))} (or a list of them), not "
            f"{json.dumps(value, default=str)}."
        )
    return kind


def _lowercased(value: Any) -> Any:
    """``value`` with the lowercase form of each string beside it (a scalar
    becomes a list when the forms differ); a range is left as given."""
    if isinstance(value, str):
        return value if value == value.lower() else [value, value.lower()]
    if isinstance(value, list):
        return list(
            dict.fromkeys(
                form
                for v in value
                for form in ((v, v.lower()) if isinstance(v, str) else (v,))
            )
        )
    return value


def source_kinds_values(value: Any) -> list[str]:
    """The kinds a typed ``source_kinds`` filter value names, in order."""
    return list(value) if isinstance(value, list) else [value]


def typed_filter_values(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """``filters`` with each string value of a boolean or numeric field
    converted to that type, in a list and in range bounds too:
    ``in_schema: "false"`` becomes ``False``, ``rank: "2"`` ``2``. A command
    line gives every value as a string; OpenSearch converts one itself, but
    the graph stores compare it as written and matched nothing, so every
    store sees typed values instead. A boolean takes ``true``/``false``,
    ``1``/``0`` or ``yes``/``no`` (any case); a number is kept an integer
    when it reads as one. A value already typed must be of the field's type:
    a boolean field takes only ``True``/``False``, a numeric one no boolean,
    an integer one no fractional number (``2.0`` becomes ``2``). A
    ``source_kinds`` value (or each value of a list) is a source kind, in
    lower case, and a string of a field ingestion stores lowercase
    (``_LOWERCASE_FIELDS``) matches its lowercase form too. Values of other
    fields are unchanged.

    Raises:
        InvalidFilterError: A value is not a value of its field's type.
    """
    if not filters:
        return filters
    typed: dict[str, Any] = {}
    for key, value in filters.items():
        kind = field_kind(key)
        if kind is FieldKind.SOURCE_KINDS:
            typed[key] = (
                [_source_kind(v) for v in value]
                if isinstance(value, list)
                else _source_kind(value)
            )
        elif key in _LOWERCASE_FIELDS:
            typed[key] = _lowercased(value)
        elif kind not in _TYPED:
            typed[key] = value
        elif isinstance(value, list):
            typed[key] = [_typed(key, kind, v) for v in value]
        elif isinstance(value, dict):
            typed[key] = {op: _typed(key, kind, v) for op, v in value.items()}
        else:
            typed[key] = _typed(key, kind, value)
    return typed


def document_filters(filters: Mapping[str, Any] | None) -> dict[str, Any]:
    """The document attribute keys of ``filters``: ``attr_<key>`` and
    ``attributes.filters.<key>``, an entry of the documents' ``filters``
    metadata. Other ``attributes.<path>`` keys name an item's own
    attributes (an entity's extracted ``role``), not its documents'."""
    return {
        key: value
        for key, value in (filters or {}).items()
        if field_kind(key) is FieldKind.DOCUMENT
    }


def loose_values(values: Iterable[Scalar]) -> tuple[Scalar, ...]:
    """``values`` with each one's other spelling: a number its JSON text, a
    string that reads as a number or boolean that value.

    An index maps an item's own attribute by its first value and converts a
    term to that type (``"1999"`` matches a stored ``1999``, ``1999`` a
    stored ``"1999"``); a store that compares as stored (a graph vertex)
    matches either spelling instead, which no two values of one attribute
    share.
    """
    spelled: list[Scalar] = []
    for given in values:
        spelled.append(given)
        if isinstance(given, bool):
            spelled.append("true" if given else "false")
        elif isinstance(given, int | float):
            spelled.append(json.dumps(given))
        elif isinstance(given, str):
            text = given.strip()
            if text in ("true", "false"):
                spelled.append(text == "true")
                continue
            try:
                spelled.append(int(text))
                continue
            except ValueError:
                pass
            try:
                number = float(text)
            except ValueError:
                continue
            if math.isfinite(number):
                spelled.append(number)
    unique = dict.fromkeys((type(v), v) for v in spelled)
    return tuple(v for _, v in unique)


# --- the clauses ----------------------------------------------------------------


@dataclass(frozen=True)
class Term:
    """``field`` holds one of ``values`` (an empty tuple matches nothing);
    ``listed``: the filter gave a list (a ``terms`` query)."""

    field: str
    values: tuple[Scalar, ...]
    kind: FieldKind = FieldKind.KEYWORD
    listed: bool = False


@dataclass(frozen=True)
class Range:
    """Some stored value of ``field`` satisfies every bound."""

    field: str
    bounds: tuple[tuple[str, Scalar], ...]
    kind: FieldKind = FieldKind.KEYWORD


@dataclass(frozen=True)
class KindsRule:
    """A ``source_kinds`` filter: the item holds evidence of one of
    ``kinds``, or, when they name ``document``, holds no ``source_kinds``
    (an item no seed graph states)."""

    kinds: tuple[str, ...]
    field: str = SOURCE_KINDS_FIELD
    kind: FieldKind = FieldKind.SOURCE_KINDS

    @property
    def unstated_passes(self) -> bool:
        """Whether an item without ``source_kinds`` passes."""
        return DOCUMENT_SOURCE_KIND in self.kinds

    @property
    def narrows(self) -> bool:
        """Whether the rule leaves out an item of some kinds (it names fewer
        than every kind): a community report, which holds no kinds, cannot
        honour it."""
        return not SOURCE_KINDS <= set(self.kinds)

    def passes(self, stored: Collection[str] | None) -> bool:
        """Whether an item holding the ``stored`` kinds passes."""
        if not stored:
            return self.unstated_passes
        return bool(set(self.kinds).intersection(str(k) for k in stored))

    def encodings(self) -> list[str]:
        """The stored ``source_kinds`` of an edge (a sorted JSON list of a
        non-empty set of kinds, ``encoded_source_kinds``) that pass."""
        return [
            encoded_source_kinds(subset)
            for size in range(1, len(SOURCE_KINDS) + 1)
            for subset in itertools.combinations(sorted(SOURCE_KINDS), size)
            if set(self.kinds).intersection(subset)
        ]


@dataclass(frozen=True)
class AnyOf:
    """At least one of ``groups`` holds, each group all of its clauses."""

    groups: tuple[tuple[Term | Range | KindsRule, ...], ...]


Clause = Term | Range | KindsRule | AnyOf


def _clause(key: str, value: Any) -> Term | Range | KindsRule:
    kind = field_kind(key)
    if kind is FieldKind.SOURCE_KINDS:
        return KindsRule(tuple(dict.fromkeys(source_kinds_values(value))), key)
    if isinstance(value, dict):
        return Range(key, tuple(value.items()), kind)
    if isinstance(value, list):
        return Term(key, tuple(value), kind, listed=True)
    return Term(key, (value,), kind)


def parse_filters(
    filters: Mapping[str, Any] | None, *, allow_empty_lists: bool = False
) -> tuple[Term | Range | KindsRule, ...]:
    """``filters`` checked (``check_filter_values``), typed
    (``typed_filter_values``) and as clauses, one per key, all of which must
    hold.

    Raises:
        InvalidFilterError: A key is refused or a value malformed.
    """
    if not filters:
        return ()
    typed = typed_filter_values(dict(filters)) or {}
    check_filter_values(typed, allow_empty_lists=allow_empty_lists)
    return tuple(_clause(key, value) for key, value in typed.items())


def parse_any_of(groups: Iterable[Mapping[str, Any]] | None) -> AnyOf | None:
    """``SearchQuery.any_of`` as one clause (``None`` without groups); each
    group is parsed as filters are (``parse_filters``).

    Raises:
        InvalidFilterError: A group is empty, or a key refused or a value
            malformed.
    """
    listed = list(groups or ())
    if not listed:
        return None
    parsed = []
    for group in listed:
        if not group:
            raise InvalidFilterError(
                "An any_of group is empty (it would match every item)."
            )
        parsed.append(parse_filters(group, allow_empty_lists=True))
    return AnyOf(tuple(parsed))


def kinds_rule(filters: Mapping[str, Any] | None) -> KindsRule | None:
    """The ``source_kinds`` rule of ``filters``, ``None`` without one (or
    with an empty list, which a scope never sets on it)."""
    value = (filters or {}).get(SOURCE_KINDS_FIELD)
    if value is None or not isinstance(value, str | list) or not value:
        return None
    typed = typed_filter_values({SOURCE_KINDS_FIELD: value}) or {}
    return KindsRule(
        tuple(dict.fromkeys(source_kinds_values(typed[SOURCE_KINDS_FIELD])))
    )
