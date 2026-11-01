# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Filterable fields of the OpenSearch indexes and Neptune entity vertices.

The declared schema for ``SearchQuery.filters`` keys: the fields the indexers
write on each index / vertex label (``OpenSearchIndexer._get_*_mapping`` and
``NeptuneIndexer._build_vertex_properties``; a unit test pins the two
together). Both retrievers scope filters with it, so a key applies only to the
indexes / labels that declare it, and the chain rejects a key that no store a
strategy reads declares.

On OpenSearch, dynamically mapped document fields are declared by prefix on
the indexes whose documents carry them: ``attr_<key>`` (one per entry of an
item's ``filters`` attribute) and ``attributes.<path>`` (the dynamic
``attributes`` object; relationships and claims hold only its ``filters``).
Every index declares ``attr_<key>`` and ``attributes.filters.<key>`` (the
documents' filters), so an index whose documents lack such a field returns
nothing for that filter rather than being searched unfiltered. Only the
entities index declares the rest of ``attributes.<path>`` (an entity's own
attributes); the other indexes drop such a key, as they drop ``type``.
Graph vertices hold no document filters: under one, strategies expand over
the relationships index instead
(``BaseSearchStrategy._expand_over_passing_relationships``), or, without that
index, check the graph walk against the entities index
(``BaseSearchStrategy._vouch_by_entities_index``); under two or more keys
both also check the survivors' text units.

Neptune entity vertices carry ``attr_<key>`` for the entity's raw top-level
attributes (e.g. ``attr_role`` from LLM extraction); document attribute filters
sit inside one JSON property (``attr_filters``), so ``attr_category`` is absent
on vertices. Neptune therefore applies ``attr_*`` keys "where present": a vertex
passes when the property matches or when it has no such property
(``NeptuneRetriever._filter_predicate``). Other keys are strict on both stores.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from unified_kg_rag.domain.models import Constants
from unified_kg_rag.shared.exceptions import InvalidFilterError

ATTRIBUTE_KEY_PREFIX = f"{Constants.ATTRIBUTE_PREFIX.value}_"
ATTRIBUTES_OBJECT_PREFIX = "attributes."


@dataclass(frozen=True)
class FilterFields:
    """Filter keys one store (or a union of stores) can match."""

    fields: frozenset[str]
    prefixes: frozenset[str] = frozenset()

    def declares(self, key: str) -> bool:
        return key in self.fields or any(key.startswith(p) for p in self.prefixes)

    def describe(self) -> list[str]:
        """Sorted keys, with each prefix rendered as ``<prefix><key>``."""
        return sorted(self.fields) + sorted(f"{p}<key>" for p in self.prefixes)


DOCUMENT_FILTERS_PREFIX = f"{ATTRIBUTES_OBJECT_PREFIX}{Constants.FILTERS.value}."


DEFAULT_MAX_TERMS_COUNT = 65536
DEFAULT_MAX_RESULT_WINDOW = 10000


def index_setting(settings: dict[str, Any], name: str, default: int) -> int:
    """``settings[name]``, also when written ``index.<name>``.

    OpenSearch takes either spelling of an index setting.
    """
    return int(settings.get(name, settings.get(f"index.{name}", default)))


@dataclass(frozen=True)
class IndexLimits:
    """The hits one search returns (``max_result_window``) and the values one
    ``terms`` clause takes (``max_terms_count``) on an index."""

    max_result_window: int = DEFAULT_MAX_RESULT_WINDOW
    max_terms_count: int = DEFAULT_MAX_TERMS_COUNT

    @classmethod
    def of(cls, *settings: dict[str, Any]) -> "IndexLimits":
        """The limits in force under every one of ``settings`` (the smallest):
        ``indexing.opensearch.index_settings`` and the settings each live
        index was created with. An index keeps the settings it was created
        with, so a changed config does not change them until a reindex."""
        return cls(
            max_result_window=min(
                (
                    index_setting(s, "max_result_window", DEFAULT_MAX_RESULT_WINDOW)
                    for s in settings
                ),
                default=DEFAULT_MAX_RESULT_WINDOW,
            ),
            max_terms_count=min(
                (
                    index_setting(s, "max_terms_count", DEFAULT_MAX_TERMS_COUNT)
                    for s in settings
                ),
                default=DEFAULT_MAX_TERMS_COUNT,
            ),
        )

    @property
    def id_batch_size(self) -> int:
        """Ids one fetch by id can name: ``max_terms_count`` per ``terms``
        clause and ``max_result_window`` hits."""
        return max(1, min(self.max_terms_count, self.max_result_window))


_SCALARS = (str, int, float, bool)
_RANGE_BOUNDS = frozenset({"gt", "gte", "lt", "lte"})


def check_filter_values(
    filters: dict[str, Any] | None, *, allow_empty_lists: bool = False
) -> None:
    """Reject filter values a search engine would refuse or misread.

    A value is a scalar (a ``term``), a list of scalars (``terms``) or a
    mapping of range bounds (``range``: ``gt``/``gte``/``lt``/``lte`` to a
    scalar). ``None`` and NaN anywhere, a tuple or set, a nested list or
    mapping, and an empty range are rejected: OpenSearch answers them with a
    400 and the embedded engine would match them instead (a null field,
    say). Checked when a query is given, before any model call, and again by
    the retriever. An empty list is rejected too (a caller's filters):
    OpenSearch refuses a search with query text under one, the embedded
    engine returns nothing. The retriever passes ``allow_empty_lists``: a
    strategy's own scope may be one (``scope_filters``: nothing the caller
    allows), which only a text-less fetch by id carries.

    Raises:
        InvalidFilterError: A value is malformed.
    """

    def bad(value: Any) -> str | None:
        if value is None:
            return "has no value (None)"
        if isinstance(value, float) and value != value:
            return "is NaN"
        if not isinstance(value, _SCALARS):
            return "is not a scalar (a string, number or boolean)"
        return None

    for key, value in (filters or {}).items():
        reason: str | None
        if value is None:
            reason = "has no value (None); give the value to match"
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
            elif unknown := sorted(set(value) - _RANGE_BOUNDS):
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


# The declared fields whose stored values are not strings, by their type on
# every store that declares them (the OpenSearch mappings and the graph
# vertices; a unit test pins this to the mappings). Every other declared
# field is a keyword or text field (the claim dates included), and the
# ``attr_*`` / ``attributes.*`` values are matched as given.
_BOOLEAN = "boolean"
_INTEGER = "integer"
_NUMBER = "number"
TYPED_FILTER_FIELDS: dict[str, str] = {
    "in_schema": _BOOLEAN,
    "rank": _NUMBER,
    "confidence": _NUMBER,
    "weight": _NUMBER,
    "rating": _NUMBER,
    "n_tokens": _INTEGER,
    "level": _INTEGER,
    "depth": _INTEGER,
    "max_cut_depth": _INTEGER,
}
_TRUE = frozenset({"true", "1", "yes"})
_FALSE = frozenset({"false", "0", "no"})


def _typed(key: str, kind: str, value: Any) -> Any:
    """``value`` as a ``kind`` value when it is a string; else unchanged."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if kind == _BOOLEAN:
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
    if not math.isfinite(number) or (kind == _INTEGER and not number.is_integer()):
        expected = "an integer" if kind == _INTEGER else "a number"
        raise InvalidFilterError(
            f"Filter '{key}' is {expected} field; give {expected}, not '{value}'."
        )
    return int(number) if kind == _INTEGER else number


def typed_filter_values(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """``filters`` with each string value of a boolean or numeric field
    (``TYPED_FILTER_FIELDS``) converted to that type, in a list and in range
    bounds too: ``in_schema: "false"`` becomes ``False``, ``rank: "2"``
    ``2``. A command line gives every value as a string; OpenSearch converts
    one itself, but the graph stores compare it as written and matched
    nothing, so every store sees typed values instead. A boolean takes
    ``true``/``false``, ``1``/``0`` or ``yes``/``no`` (any case); a number
    is kept an integer when it reads as one. Values of other fields and
    values already typed are unchanged.

    Raises:
        InvalidFilterError: A string is not a value of its field's type.
    """
    if not filters:
        return filters
    typed: dict[str, Any] = {}
    for key, value in filters.items():
        kind = TYPED_FILTER_FIELDS.get(key)
        if kind is None:
            typed[key] = value
        elif isinstance(value, list):
            typed[key] = [_typed(key, kind, v) for v in value]
        elif isinstance(value, dict):
            typed[key] = {op: _typed(key, kind, v) for op, v in value.items()}
        else:
            typed[key] = _typed(key, kind, value)
    return typed


def document_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    """The document attribute keys of ``filters``: ``attr_<key>`` and
    ``attributes.filters.<key>``, an entry of the documents' ``filters``
    metadata. Other ``attributes.<path>`` keys name an item's own
    attributes (an entity's extracted ``role``), not its documents'."""
    return {
        key: value
        for key, value in (filters or {}).items()
        if key.startswith((ATTRIBUTE_KEY_PREFIX, DOCUMENT_FILTERS_PREFIX))
    }


def union_filter_fields(items: Iterable[FilterFields]) -> FilterFields:
    """The keys any of ``items`` declares."""
    fields: set[str] = set()
    prefixes: set[str] = set()
    for item in items:
        fields |= item.fields
        prefixes |= item.prefixes
    return FilterFields(frozenset(fields), frozenset(prefixes))


# Document filters (``document_filters``): every index carries them.
_OPENSEARCH_DOCUMENT_PREFIXES = frozenset(
    {ATTRIBUTE_KEY_PREFIX, DOCUMENT_FILTERS_PREFIX}
)
# Entities also carry their own attributes (``attributes.role``), which no
# other index holds: such a key leaves the other indexes unfiltered, as
# ``type`` does, instead of emptying them.
_OPENSEARCH_ENTITY_PREFIXES = frozenset(
    {ATTRIBUTE_KEY_PREFIX, ATTRIBUTES_OBJECT_PREFIX}
)


def opensearch_filter_fields(
    target_language: str, *additional_languages: str
) -> dict[str, FilterFields]:
    """Index kind -> filterable fields (one ``translated_text_<language>`` per
    translated language)."""
    translated = {
        f"translated_text_{language}"
        for language in (target_language, *additional_languages)
    }
    return {
        "text_units": FilterFields(
            frozenset({"id", "text", *translated, "community_ids", "n_tokens"}),
            _OPENSEARCH_DOCUMENT_PREFIXES,
        ),
        "entities": FilterFields(
            frozenset(
                {
                    "id",
                    "name",
                    "name.keyword",
                    "description",
                    "type",
                    "rank",
                    "confidence",
                    "text_unit_ids",
                    "in_schema",
                    "type_iri",
                }
            ),
            _OPENSEARCH_ENTITY_PREFIXES,
        ),
        "relationships": FilterFields(
            frozenset(
                {
                    "id",
                    "source_id",
                    "target_id",
                    "source_name",
                    "target_name",
                    "description",
                    "weight",
                    "rank",
                    "text_unit_ids",
                    # No ``type``: a relationship's type is not an entity
                    # type, and ``type: PERSON`` would empty this index under
                    # mix and hybrid.
                    "in_schema",
                    "predicate_iri",
                }
            ),
            _OPENSEARCH_DOCUMENT_PREFIXES,
        ),
        "claims": FilterFields(
            frozenset(
                {
                    "id",
                    "subject_id",
                    "object_id",
                    "subject_name",
                    "object_name",
                    "type",
                    "status",
                    "start_date",
                    "end_date",
                    "description",
                    "source_text",
                    "text_unit_ids",
                }
            ),
            _OPENSEARCH_DOCUMENT_PREFIXES,
        ),
        "community_reports": FilterFields(
            frozenset(
                {
                    "id",
                    "community_id",
                    "name",
                    "summary",
                    "full_content",
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
            _OPENSEARCH_DOCUMENT_PREFIXES,
        ),
    }


# The filterable properties of entity vertices, the only ones the graph
# filters. ``attr_*`` keys on Neptune apply where present (see the module
# docstring).
NEPTUNE_ENTITY_FILTER_FIELDS = FilterFields(
    frozenset(
        {
            "id",
            "name",
            "type",
            "description",
            "rank",
            "confidence",
            "text_unit_ids",
            "community_ids",
            "in_schema",
            "type_iri",
        }
    ),
    frozenset({ATTRIBUTE_KEY_PREFIX}),
)
