# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the OpenSearch indexes hold beyond the filter catalog, and their limits.

The filter keys every store declares, and how each matches, are the
backend-neutral catalog of ``domain.retrieval.filters`` (``ITEM_FILTER_FIELDS``
per index kind, ``GRAPH_ENTITY_FILTER_FIELDS``); a unit test pins it to the
mappings ``OpenSearchIndexer._get_*_mapping`` writes. This module holds what
only the OpenSearch retriever reads: the fields an index maps that are no
filter key, the fields an index built before them may lack (probed before a
kNN filter on one), and the limits of one search.

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
"""

from dataclasses import dataclass
from typing import Any

from unified_kg_rag.domain.ingestion.seed_graph import SOURCE_KINDS_FIELD

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


# The schema annotations (``domain.ingestion.schema_annotation``). An index
# built before they were mapped lacks them until an ingestion run maps them
# (``OpenSearchIndexer.check_index_settings``), so the retriever treats them
# as it treats ``attr_*`` keys: a probe first, and no document holding the
# field matches no value of it.
SCHEMA_ANNOTATION_FIELDS = frozenset({"in_schema", "type_iri", "predicate_iri"})
# The seed-graph fields (an entity's ``iri``, ``source_kinds``) are mapped
# as late, and probed alike.
SEED_GRAPH_FIELDS = frozenset({"iri", SOURCE_KINDS_FIELD})
LATE_MAPPED_FIELDS = SCHEMA_ANNOTATION_FIELDS | SEED_GRAPH_FIELDS

# The keyword a seed graph's text units hold (``seed_graph``; a document's
# units lack it), by which the retriever itself keeps them out of every text
# search: it is no filter key.
TEXT_UNIT_KIND_FIELD = "source_kind"
# Fields an index maps that are no filter key.
UNDECLARED_MAPPED_FIELDS: dict[str, frozenset[str]] = {
    "text_units": frozenset({TEXT_UNIT_KIND_FIELD}),
    # Rendered in the relationship's context line, but no filter key (an
    # entity type's: ``ITEM_FILTER_FIELDS``).
    "relationships": frozenset({"type"}),
    # The digest a summary is reused by.
    "summary_nodes": frozenset({"input_key"}),
    # Analyzed text without an exact copy: ``name`` is an entity's exact
    # name, which leaves the reports unfiltered as ``type`` does.
    "community_reports": frozenset({"name"}),
}
