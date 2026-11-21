# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An incremental run's community reports written again after a removal.

A removal keeps a community whose text units are not all removed, and its
report, which was written from the removed text too and may still state it
(``IndexBuildRecord.outdated_reports``). The rule is the summary tree's
(docs/design/retrieval-hipporag-raptor.md, "Incremental runs"): a summary
paraphrases what it covers, so one citing a removed text unit is not kept as
it is but made again from what remains, and every summary above it too. A
report's text units are its members' (a parent's hold its children's), so
the removal's search for the reports citing a removed unit finds the
community's whole chain up to its root; each is written again here.

Its community is kept as clustered (an incremental run clusters only its
delta), so the report is made from the community's members as the stores
now hold them, as a full build's detector reads them: the stored entities
whose level-0 communities (``Entity.community_ids``) lie under it, their
descriptions as the commit derived them, the relationships among them, the
claims about them, and their degrees over every stored edge (the order a
report packs its members in, and its rank). A parent's roll-up reads its
children's stored reports beside those written again with it. The answer is
kept in the extraction store by its input, as a full build's: a report whose
input did not change (its removed text named no member) takes the stored
answer, and a later full build reuses a new one.

One model call per report at most, under the run's budget
(``processing.max_run_*``). A community none of whose members remains, or
only seed-graph members, has no report in a full build: its report is
deleted. A report the run cannot write again (the model failed, the stores
could not be read, a report below it is missing, or one below it got no
report this run and it rolls its children up) stays flagged outdated and
listed in the build record, and the next run tries again.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, NamedTuple

import networkx as nx

from unified_kg_rag.adapters.ingestion.community_detector import (
    CommunityDetector,
    ReportRecords,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.ingestion.community_clustering import (
    community_label,
    community_name,
)
from unified_kg_rag.domain.ingestion.community_hierarchy import community_report_id
from unified_kg_rag.domain.ingestion.graph_builder import (
    GraphBuilder,
    edge_relationships,
)
from unified_kg_rag.domain.ingestion.seed_graph import seed_only
from unified_kg_rag.domain.models import (
    Claim,
    Community,
    CommunityReport,
    Constants,
    Entity,
    TextUnit,
)
from unified_kg_rag.domain.retrieval import source_fields
from unified_kg_rag.ports.indexer import BaseIndexer, IndexingStats
from unified_kg_rag.shared import NamespaceLeaseError, get_logger
from unified_kg_rag.shared.logging import log_error

if TYPE_CHECKING:
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

logger = get_logger(__name__)


class RefreshedReports(NamedTuple):
    """What a refresh did with the outdated reports it was given."""

    # Written again from their community's current members.
    written: set[str]
    # Deleted: their community has no document-backed member left, or
    # the report is no longer stored.
    deleted: set[str]
    # Left outdated: not written again (see the module docstring).
    kept: set[str]
    # The store writes (``IndexingStats`` by write).
    results: dict[str, IndexingStats]


class _Hierarchy(NamedTuple):
    # The stored reports read, by community id: the outdated ones and every
    # report below them.
    reports: dict[str, CommunityReport]
    # The outdated reports' communities with their level-0 communities;
    # one whose hierarchy could not be read (a report below it is missing)
    # is absent.
    leaves: dict[str, set[str]]


class ReportRefresh:
    """Writes outdated community reports again (see the module docstring),
    for the indexing stage of an incremental run.

    ``records`` keeps the report answers by their input (the extraction
    store). The detector, which builds the report chain, is made on first
    use: most runs remove nothing a report cites.
    """

    def __init__(
        self,
        manager: IndexingManager,
        providers: Providers,
        records: ReportRecords | None = None,
    ) -> None:
        self._manager = manager
        self._providers = providers
        self._records = records
        self._detector: CommunityDetector | None = None

    def _report_detector(self) -> CommunityDetector:
        if self._detector is None:
            self._detector = CommunityDetector(
                self._manager.config, show_progress=False, providers=self._providers
            )
        return self._detector

    def __call__(
        self,
        suffix: str,
        report_ids: Collection[str],
        candidates: Collection[str] = (),
    ) -> RefreshedReports:
        """Write the stored reports ``report_ids`` of ``suffix`` again.

        ``candidates``: stored entities that may be a member and cite no
        text unit of its report (a removal stripped them of their units the
        report cites), looked at beside those citing one.
        """
        ids = set(report_ids)
        if not ids:
            return RefreshedReports(set(), set(), set(), {})
        try:
            return self._refresh(suffix, ids, set(candidates))
        except NamespaceLeaseError:
            raise
        except Exception as e:  # noqa: BLE001 - left outdated, retried next run
            log_error(
                logger,
                logging.WARNING,
                "Writing %d outdated community reports of '%s' again failed (%s): "
                "they stay flagged outdated and the next run tries again",
                len(ids),
                suffix,
                error=e,
            )
            return RefreshedReports(set(), set(), ids, {})

    def _refresh(
        self, suffix: str, ids: set[str], candidates: set[str]
    ) -> RefreshedReports:
        manager = self._manager
        vector = manager.vector_indexer
        stored = vector.read_community_reports(sorted(ids), suffix)
        if stored is None:
            logger.warning(
                "The vector store cannot read community reports by id: %d "
                "outdated reports of '%s' stay outdated until indexing.reset: "
                "true writes them again",
                len(ids),
                suffix,
            )
            return RefreshedReports(set(), set(), ids, {})
        outdated = {r.community_id: r for r in stored}
        gone = ids - {r.id for r in stored}
        hierarchy = self._hierarchy(suffix, outdated)
        unread = {outdated[c].id for c in outdated if c not in hierarchy.leaves}
        if unread:
            logger.warning(
                "%d outdated community reports of '%s' have a community below "
                "them without a stored report: they stay outdated",
                len(unread),
                suffix,
            )
        read = self._members(suffix, hierarchy, candidates)
        if read is None:
            logger.warning(
                "The graph store cannot find what text units state: %d outdated "
                "community reports of '%s' stay outdated",
                len(ids),
                suffix,
            )
            return RefreshedReports(set(), set(), ids, {})
        members, graph = read
        communities: list[Community] = []
        emptied: set[str] = set()
        for community_id, leaves in sorted(hierarchy.leaves.items()):
            report = outdated[community_id]
            entity_ids = sorted(
                e.id for e in members if set(e.community_ids or ()) & leaves
            )
            units = sorted(
                {
                    u
                    for e in members
                    if e.id in entity_ids
                    for u in e.text_unit_ids or ()
                }
            )
            if not entity_ids or seed_only(units):
                emptied.add(report.id)
                continue
            communities.append(self._community(report, entity_ids, units, graph))
        results: dict[str, IndexingStats] = {}
        if emptied:
            # A full build has no report for such a community.
            results[f"vector_emptied_reports_{suffix}"] = vector.delete_by_id(
                sorted(emptied),
                manager.config.indexing.opensearch.community_reports_index_prefix,
                suffix,
            )
            emptied -= set(results[f"vector_emptied_reports_{suffix}"].failed_ids)
        written = self._write(suffix, communities, hierarchy, outdated, graph, results)
        return RefreshedReports(
            written,
            emptied | gone,
            ids - written - emptied - gone,
            results,
        )

    def _hierarchy(
        self, suffix: str, outdated: Mapping[str, CommunityReport]
    ) -> _Hierarchy:
        """The stored reports below the ``outdated`` ones (read level by
        level through their ``child_community_ids``) and the level-0
        communities of each outdated one."""
        vector = self._manager.vector_indexer
        reports: dict[str, CommunityReport] = dict(outdated)
        pending = {c for r in outdated.values() for c in r.child_community_ids}
        missing: set[str] = set()
        while pending - reports.keys() - missing:
            wanted = sorted(pending - reports.keys() - missing)
            read = vector.read_community_reports(
                [community_report_id(c) for c in wanted], suffix
            )
            found = {r.community_id: r for r in read or []}
            reports.update(found)
            missing |= set(wanted) - found.keys()
            pending = {c for r in found.values() for c in r.child_community_ids}

        leaves: dict[str, set[str] | None] = {}

        def below(community_id: str) -> set[str] | None:
            # Iterative post-order: a chain of splits can be deep.
            stack = [community_id]
            while stack:
                current = stack[-1]
                if current in leaves:
                    stack.pop()
                    continue
                report = reports.get(current)
                if report is None:
                    leaves[current] = None
                    stack.pop()
                    continue
                children = [c for c in report.child_community_ids if c not in leaves]
                if children:
                    stack.extend(children)
                    continue
                stack.pop()
                if not report.child_community_ids:
                    leaves[current] = {current}
                    continue
                parts = [leaves[c] for c in report.child_community_ids]
                leaves[current] = (
                    None
                    if any(part is None for part in parts)
                    else set().union(*(p for p in parts if p is not None))
                )
            return leaves[community_id]

        known = {c: under for c in outdated if (under := below(c)) is not None}
        return _Hierarchy(reports, known)

    def _members(
        self, suffix: str, hierarchy: _Hierarchy, candidates: set[str]
    ) -> tuple[list[Entity], nx.Graph] | None:
        """The stored members of the outdated reports' communities (of the
        entities citing a text unit one of them cites, and ``candidates``,
        those in one of their level-0 communities), and the graph their
        reports read: those entities, their stored edges, the entities at
        the other ends and, when claim extraction is on, the claims about
        them. ``None`` when the graph store cannot find what text units
        state."""
        manager = self._manager
        graph_store = manager.graph_indexer
        units = sorted(
            {
                u
                for community_id in hierarchy.leaves
                for u in hierarchy.reports[community_id].text_unit_ids or ()
            }
        )
        citing = graph_store.find_ids_citing(units, suffix=suffix) if units else {}
        if citing is None:
            return None
        wanted = sorted(set(citing) | candidates)
        entities = graph_store.read_entities(wanted, suffix=suffix) if wanted else []
        leaves = set().union(*hierarchy.leaves.values()) if hierarchy.leaves else set()
        members = [e for e in entities if set(e.community_ids or ()) & leaves]
        member_ids = sorted(e.id for e in members)
        incident = (
            graph_store.find_incident_relationship_ids(member_ids, suffix=suffix)
            if member_ids
            else []
        )
        relationships = (
            graph_store.read_relationships(sorted(set(incident)), suffix=suffix)
            if incident
            else []
        )
        ends = {r.source_id for r in relationships} | {
            r.target_id for r in relationships
        }
        known = {e.id for e in members}
        others = graph_store.read_entities(sorted(ends - known), suffix=suffix)
        claims: list[Claim] = []
        if manager.config.processing.claim_extraction.enabled and member_ids:
            claim_ids = manager._claims_about(suffix, member_ids)
            if claim_ids:
                claims = manager.vector_indexer.read_claims(claim_ids, suffix)
        nodes = [_in_suffix(e, suffix) for e in [*members, *others]]
        graph = GraphBuilder(nodes, relationships, claims).build()
        return members, graph

    @staticmethod
    def _community(
        report: CommunityReport,
        entity_ids: list[str],
        units: list[str],
        graph: nx.Graph,
    ) -> Community:
        """The community of the stored ``report`` with its current
        members, as the detector makes one."""
        level = report.level or 0
        label = community_label(level, report.community_id)
        members = set(entity_ids)
        relationship_ids = sorted(
            str(rel["id"])
            for _, _, data in graph.subgraph(members).edges(data=True)
            for rel in edge_relationships(data)
            if rel.get("id") is not None
        )
        return Community(
            id=report.community_id,
            short_id=label,
            name=community_name(level, label),
            level=str(level),
            parent=report.parent_community_id or "",
            children=list(report.child_community_ids),
            entity_ids=entity_ids,
            relationship_ids=relationship_ids,
            text_unit_ids=units,
            size=len(entity_ids),
            attributes={Constants.INDEX.value: BaseIndexer.get_suffix(report)},
        )

    def _write(
        self,
        suffix: str,
        communities: list[Community],
        hierarchy: _Hierarchy,
        outdated: Mapping[str, CommunityReport],
        graph: nx.Graph,
        results: dict[str, IndexingStats],
    ) -> set[str]:
        """Generate and write the reports of ``communities``; the ids
        written."""
        if not communities:
            return set()
        vector = self._manager.vector_indexer
        units = sorted({u for c in communities for u in c.text_unit_ids or ()})
        attributes = vector.read_text_unit_attributes(units, suffix=suffix)
        text_units = [
            TextUnit(
                id=unit_id,
                text="",
                document_ids=source_fields.document_ids({"attributes": stored}),
                attributes=stored,
            )
            for unit_id, stored in sorted(attributes.items())
        ]
        detector = self._report_detector().report_on(graph)
        for community in communities:
            # As a full build's detector makes it: the graph store's vertex
            # is written again with the members it now has.
            community.attributes = {
                **(community.attributes or {}),
                **detector.community_attributes(
                    community.entity_ids or [], community.relationship_ids or []
                ),
            }
        # A parent rolls up its children's current reports: those written
        # again beside it, and the stored ones that are not outdated (an
        # outdated one may state the removed text, an emptied one is gone).
        current = {
            community_id: report
            for community_id, report in hierarchy.reports.items()
            if community_id not in outdated
        }
        rollup = (
            self._manager.config.graph.community_detection.report_generation.enable_sub_community_rollup
        )
        reports: list[CommunityReport] = []
        failed: set[str] = set()
        waiting: set[str] = set()
        # Level by level, the finest first, as the detector rolls up: a
        # parent whose child got no report this run would roll up without
        # it and, written, never be asked again once the child is (the next
        # run writes only what stays flagged). It waits, flagged, as a
        # summary over a failed one is rebuilt (the summary tree's rule).
        for level in sorted({_level(c) for c in communities}):
            ready = []
            for community in communities:
                if _level(community) != level:
                    continue
                if rollup and (failed | waiting) & set(community.children or ()):
                    waiting.add(community.id)
                else:
                    ready.append(community)
            if not ready:
                continue
            made, none = detector.generate_reports(
                ready, text_units=text_units, records=self._records, sub_reports=current
            )
            reports += made
            failed |= set(none)
            current.update((r.community_id, r) for r in made)
        if failed:
            logger.warning(
                "%d of %d outdated community reports of '%s' got no report "
                "from the model: they stay outdated and the next run tries "
                "again",
                len(failed),
                len(communities),
                suffix,
            )
        if waiting:
            logger.warning(
                "%d outdated community reports of '%s' roll up a report that "
                "failed: they stay outdated and the next run writes them "
                "with it",
                len(waiting),
                suffix,
            )
        for report in reports:
            # The community keeps its place in the hierarchy.
            kept = outdated[report.community_id]
            report.level = kept.level
            report.depth = kept.depth
            report.min_cut_depth = kept.min_cut_depth
            report.parent_community_id = kept.parent_community_id
            report.child_community_ids = list(kept.child_community_ids)
        self._write_vertices(suffix, communities, results)
        if not reports:
            return set()
        stats = vector.upsert_community_reports(reports)
        results[f"vector_refreshed_reports_{suffix}"] = stats
        written = {r.id for r in reports} - set(stats.failed_ids)
        if stats.failed_items and not stats.failed_ids:
            # A write that failed without naming its items wrote none.
            written = set()
        logger.info(
            "Wrote %d outdated community reports of '%s' again from their "
            "communities' current members",
            len(written),
            suffix,
        )
        return written

    def _write_vertices(
        self,
        suffix: str,
        communities: list[Community],
        results: dict[str, IndexingStats],
    ) -> None:
        """Write the graph store's vertices of ``communities`` again from
        their current members (text units, size, member names), whatever
        their reports: membership does not wait for a model answer.
        Best-effort: a failed write leaves the vertex as the last full
        build wrote it, which only a removal's search reads, by live text
        units."""
        try:
            stats = self._manager.graph_indexer.upsert_communities(communities)
        except NamespaceLeaseError:
            raise
        except Exception as e:  # noqa: BLE001 - the reports still count
            log_error(
                logger,
                logging.WARNING,
                "Writing %d refreshed communities of '%s' to the graph store "
                "failed (%s): their vertices keep the removed text units",
                len(communities),
                suffix,
                error=e,
            )
            return
        results[f"graph_refreshed_communities_{suffix}"] = stats


def _level(community: Community) -> int:
    """The community's level, as the detector orders its roll-up."""
    try:
        return int(community.level)
    except (TypeError, ValueError):
        return 0


def _in_suffix(entity: Entity, suffix: str) -> Entity:
    """A copy of ``entity`` tagged with the index ``suffix`` it was read
    from (the reports' suffix)."""
    return entity.model_copy(
        update={
            "attributes": {**(entity.attributes or {}), Constants.INDEX.value: suffix}
        }
    )
