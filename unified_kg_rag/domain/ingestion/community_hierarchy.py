# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where each community report sits in the community hierarchy.

Global search maps over the reports of one *cut* of the hierarchy, as MS
GraphRAG does (``read_indexer_reports``): the communities at depth ``D``,
plus the leaves that stop above ``D``, so every entity is covered once. The
hierarchy is built bottom-up (``Community.level`` 0 is the finest), while MS
GraphRAG counts its ``community_level`` down from the first Leiden pass over
the whole graph, so each community's depth is counted that way when the
hierarchy is built (``Community.depth``, negative for the coarser levels
above that pass) and stored on its report; a community without one (not
built by the hierarchy) is at its distance from its root.

A report belongs to cut ``D`` when ``min_cut_depth <= D <= max_cut_depth``:
``max_cut_depth`` is the report's own depth for a community with children and
:data:`LEAF_CUT_DEPTH` for a leaf (a leaf stands for every deeper cut), and
``min_cut_depth`` one below its parent's depth, or :data:`ROOT_CUT_DEPTH`
for a community without a parent (a root stands for every shallower cut): one
range filter per field, which every store applies. So every cut covers each
entity once, the coarser cuts at negative depths too, where a first-pass
community no coarser one groups (an isolated component), or a community a
coarser level carried up past one, stands for itself. In the first pass a
parent is one depth above its children, so ``min_cut_depth <= D`` reads as
``depth <= D`` for ``D >= 0``: :func:`cut_filter` filters on ``depth`` there,
which a report written without ``min_cut_depth`` holds too.
"""

from __future__ import annotations

from collections.abc import Mapping

from unified_kg_rag.domain.ingestion.base_processor import merge_attributes
from unified_kg_rag.domain.models import (
    Community,
    CommunityReport,
    Constants,
    TextUnit,
)
from unified_kg_rag.shared.utils import generate_stable_id

# A leaf stands for every deeper cut, a root for every shallower one.
LEAF_CUT_DEPTH = 1_000_000
ROOT_CUT_DEPTH = -LEAF_CUT_DEPTH
# The report fields the cuts select by.
CUT_FIELDS = frozenset({"depth", "min_cut_depth", "max_cut_depth"})


def community_report_id(community_id: str) -> str:
    """The id of the report of the community ``community_id``: one report
    per community, so a report written again replaces it."""
    return generate_stable_id(f"report:{community_id}")


def community_depths(communities: list[Community]) -> dict[str, int]:
    """Each community's depth: its ``depth`` when the hierarchy set one,
    else its distance from its root (a community with no parent, or whose
    parent is not in ``communities``)."""
    parents = {c.id: c.parent for c in communities}
    depths: dict[str, int] = {}

    def depth(community_id: str) -> int:
        chain: list[str] = []
        current = community_id
        while current not in depths:
            parent = parents.get(current)
            if not parent or parent not in parents or parent in chain:
                depths[current] = 0
                break
            chain.append(current)
            current = parent
        found = depths[current]
        for node in reversed(chain):
            found += 1
            depths[node] = found
        return depths[community_id]

    for community in communities:
        depth(community.id)
    return {c.id: c.depth if c.depth is not None else depths[c.id] for c in communities}


def attach_report_hierarchy(
    reports: list[CommunityReport] | None, communities: list[Community] | None
) -> None:
    """Record each report's level, depth, shallowest cut, parent and
    children (in place).

    A report whose community is not in ``communities`` is left as it is.
    """
    if not reports or not communities:
        return
    by_id = {c.id: c for c in communities}
    depths = community_depths(communities)
    for report in reports:
        community = by_id.get(report.community_id)
        if community is None:
            continue
        try:
            report.level = int(community.level)
        except (TypeError, ValueError):
            report.level = None
        report.depth = depths[community.id]
        parent = community.parent
        report.parent_community_id = parent if parent in by_id else None
        report.min_cut_depth = depths[parent] + 1 if parent in by_id else ROOT_CUT_DEPTH
        report.child_community_ids = [c for c in community.children if c in by_id]


def attach_report_lineage(
    reports: list[CommunityReport],
    community_units: Mapping[str, list[str]],
    text_units: list[TextUnit] | None,
) -> None:
    """Record each report's candidate source text units, documents and
    document ``filters`` (in place): those of its community's text units
    (``community_units``, by community id).

    A report is the only artifact in the index with no pointer back to its
    sources, so a retrieved report cannot be filtered by origin. Detection
    already unions its member entities' ``text_unit_ids`` onto the
    Community, so the report inherits them; the source documents come from
    those units.

    This is community MEMBERSHIP provenance, not report-input provenance:
    the prompt is capped at ``max_entities_per_report`` and a token budget,
    and the lineage deliberately ignores that cap (a community with ``e1``
    from ``d1`` and ``e2`` from ``d2`` records ``document_ids=["d1", "d2"]``
    even when only ``e1`` reached the prompt). The fields therefore may name
    material omitted from the report and do not establish sentence-level
    citation support; they exist so a consumer can filter reports by source
    or re-summarize them per reader.

    The report's ``filters`` attribute is the union of its text units'
    document filters, so an attribute filter matches a report when one of
    its documents matches, as it does an entity. Without text units, or
    when none has filters, the report has none and an attribute filter
    excludes it (fail-closed). A removal narrows a kept report the same way
    to the text units it still cites.

    ``text_units``: the corpus's (or those the units name); omit to leave
    ``document_ids`` empty.
    """
    unit_to_documents = {
        unit.id: list(unit.document_ids or []) for unit in (text_units or [])
    }
    filters_key = Constants.FILTERS.value
    unit_filters = {
        unit.id: filters
        for unit in (text_units or [])
        if isinstance(filters := (unit.attributes or {}).get(filters_key), dict)
        and filters
    }
    for report in reports:
        unit_ids = list(community_units.get(report.community_id, []))
        report.text_unit_ids = unit_ids
        document_ids: set[str] = set()
        for unit_id in unit_ids:
            document_ids.update(unit_to_documents.get(unit_id, ()))
        # Sorted for a stable, reproducible record.
        report.document_ids = sorted(document_ids)
        attributes = {
            k: v for k, v in (report.attributes or {}).items() if k != filters_key
        }
        maps = [{filters_key: unit_filters[u]} for u in unit_ids if u in unit_filters]
        filters = (merge_attributes(maps) or {}).get(filters_key)
        if filters:
            attributes[filters_key] = filters
        report.attributes = attributes


def keep_stored_lineage(report: CommunityReport, stored: CommunityReport) -> None:
    """Give ``report`` the documents and document ``filters`` of ``stored``,
    its earlier version (in place): what a report written again keeps when
    the store cannot read its text units' attributes, of which
    :func:`attach_report_lineage` would record none."""
    filters_key = Constants.FILTERS.value
    report.document_ids = list(stored.document_ids or [])
    attributes = {
        k: v for k, v in (report.attributes or {}).items() if k != filters_key
    }
    if filters := (stored.attributes or {}).get(filters_key):
        attributes[filters_key] = filters
    report.attributes = attributes


def max_cut_depth(report: CommunityReport) -> int | None:
    """The deepest cut the report belongs to (see the module docstring)."""
    if report.depth is None:
        return None
    return report.depth if report.child_community_ids else LEAF_CUT_DEPTH


def cut_filter(depth: int) -> dict[str, dict[str, int]]:
    """The store filter selecting the reports of the depth-``depth`` cut
    (see the module docstring)."""
    shallowest = "depth" if depth >= 0 else "min_cut_depth"
    return {shallowest: {"lte": depth}, "max_cut_depth": {"gte": depth}}


def coarser_cut_filter(depth: int) -> dict[str, dict[str, int]]:
    """The store filter matching the reports at ``depth`` or above that a
    cut there can select: none on a store without a coarser level, or whose
    reports were written without ``min_cut_depth``."""
    return {"depth": {"lte": depth}, "min_cut_depth": {"lte": depth}}
