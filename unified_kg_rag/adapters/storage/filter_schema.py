# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Filterable fields of the OpenSearch indexes and Neptune vertex labels.

The declared schema for ``SearchQuery.filters`` keys: the fields the indexers
write on each index / vertex label (``OpenSearchIndexer._get_*_mapping`` and
``NeptuneIndexer._build_vertex_properties``; a unit test pins the two
together). Both retrievers scope filters with it, so a key applies only to the
indexes / labels that declare it, and the chain rejects a key that no store a
strategy reads declares.

On OpenSearch, dynamically mapped document fields are declared by prefix on
the indexes whose documents carry them: ``attr_<key>`` (one per entry of an
item's ``filters`` attribute) and ``attributes.<path>`` (the dynamic
``attributes`` object). A prefix key applies to every such index, so an index
whose documents lack the field returns nothing for that filter rather than
being searched unfiltered.

Neptune entity vertices carry ``attr_<key>`` for the entity's raw top-level
attributes (e.g. ``attr_role`` from LLM extraction); document attribute filters
sit inside one JSON property (``attr_filters``), so ``attr_category`` is absent
on vertices. Neptune therefore applies ``attr_*`` keys "where present": a vertex
passes when the property matches or when it has no such property
(``NeptuneRetriever._filter_predicate``). Other keys are strict on both stores.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from unified_kg_rag.domain.models import Constants

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


def union_filter_fields(items: Iterable[FilterFields]) -> FilterFields:
    """The keys any of ``items`` declares."""
    fields: set[str] = set()
    prefixes: set[str] = set()
    for item in items:
        fields |= item.fields
        prefixes |= item.prefixes
    return FilterFields(frozenset(fields), frozenset(prefixes))


_OPENSEARCH_ATTRIBUTE_PREFIXES = frozenset(
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
            _OPENSEARCH_ATTRIBUTE_PREFIXES,
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
                }
            ),
            _OPENSEARCH_ATTRIBUTE_PREFIXES,
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
                }
            )
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
                    "description",
                    "source_text",
                }
            )
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
                }
            ),
            _OPENSEARCH_ATTRIBUTE_PREFIXES,
        ),
    }


# Vertex label kind -> filterable properties. ``attr_*`` keys on Neptune apply
# where present (see the module docstring).
NEPTUNE_FILTER_FIELDS: dict[str, FilterFields] = {
    "entity": FilterFields(
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
            }
        ),
        frozenset({ATTRIBUTE_KEY_PREFIX}),
    ),
    "community": FilterFields(
        frozenset({"id", "name", "level", "parent", "size", "period", "children"})
    ),
}
