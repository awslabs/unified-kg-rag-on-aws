# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Removal of text units (removed or changed documents) from both stores.

What only the removed text units state is deleted; what other text units
state too is stripped of them and derived again (the *strip*).

A strip is all or retry. It writes in phases, each after the ones it
depends on: the claims under new ids, then the stored claims; the new
vertices (a re-keyed entity's, a split's part), the stored entities gaining
a unit, the graph edges, the vector edges, the vector entities and, last,
the kept entities' graph copies. The first phase with a rejected write stops
the strip, and the removal fails (the next run retries it). What the strip
did not write keeps its stored copy, which still cites the removed text
units (the kept entities, a split's origin, a re-keyed entity's old vertex)
or is in the pending-strip note written before the first write (the other
side of a merge, a re-keyed or re-pointed edge, an edge or claim under a new
id, a split's claims): the retry finds it there and derives it again, as
after a run killed between two phases. Written on past a rejection, a copy
it wrote no longer led a retry to what the rejected write left, and a stored
copy holding a rejected new item's lines (a split's origin, the claim a
claim was split off) was overwritten without them.
"""

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple

from unified_kg_rag.application.storage.stored_items import (
    _PENDING_STRIP,
    _STRIPPED_SUMMARIES,
    IndexingTask,
    _claim_text,
    _is_summary,
    _note_ids,
    _other_spellings_of_parts,
    _reads_back,
    _vector_stale,
    write_claims_new_ids_first,
)
from unified_kg_rag.domain.ingestion.cache_keys import unbuilt_record_prefixes
from unified_kg_rag.domain.ingestion.contributions import (
    RECORD_KEYS_ATTRIBUTE,
    RECORD_KINDS,
    attribute_record_keys,
)
from unified_kg_rag.domain.ingestion.derivation import Derived
from unified_kg_rag.domain.ingestion.merge import (
    remove_claim_text_units,
    remove_text_units,
)
from unified_kg_rag.domain.ingestion.methodologies import artifacts_to_build
from unified_kg_rag.domain.ingestion.names import compact_name_key
from unified_kg_rag.domain.models import Entity, Relationship
from unified_kg_rag.ports.indexer import IndexingStats
from unified_kg_rag.shared import NamespaceLeaseError, get_logger
from unified_kg_rag.shared.write_fence import check_write_fence

if TYPE_CHECKING:
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

logger = get_logger(__name__)


class TextUnitRemoval:
    """The removal of text units from the stores, for :class:`IndexingManager`
    (whose stores and run state it uses): what only they stated is deleted,
    what others state too is stripped of them and derived again."""

    def __init__(self, manager: "IndexingManager") -> None:
        self._m = manager

    def remove_text_units(
        self, suffix: str, text_unit_ids: list[str]
    ) -> dict[str, IndexingStats]:
        """See :meth:`IndexingManager.remove_text_units`."""
        m = self._m
        try:
            citing = self._find_citing(suffix, text_unit_ids)
        except Exception as e:  # noqa: BLE001 - reported as failed items
            logger.error("Finding the removed text units' items failed: %s", e)
            stats = IndexingStats(total_items=len(text_unit_ids))
            stats.add_error(str(e), count=stats.total_items)
            return {f"find_removed_{suffix}": stats}
        exclusive, gone = citing.exclusive, citing.gone

        outdated = sorted(i for i, units in citing.reports.items() if not gone(units))
        if outdated:
            logger.warning(
                "%d community reports in '%s' cite removed text units and other "
                "ones: they are kept, and may state the removed text until "
                "indexing.reset: true rebuilds them",
                len(outdated),
                suffix,
            )
            m.outdated_reports.update(outdated)

        shared_entities, shared_relationships, shared_claims, doomed = (
            self._shared_and_doomed(citing, text_unit_ids)
        )
        logger.info(
            "Removing %d artifacts of %d text units in '%s'; stripping %d "
            "shared entities, %d relationships and %d claims",
            len(doomed),
            len(text_unit_ids),
            suffix,
            len(shared_entities),
            len(shared_relationships),
            len(shared_claims),
        )
        # An entity only the vector store holds (an interrupted run never
        # wrote its graph copy) has its edges there only.
        found = citing.found
        vector_only = (
            {
                i
                for i, units in citing.entities.items()
                if exclusive(units) and i not in found
            }
            if found is not None and _reads_back(m.graph_indexer)
            else set()
        )
        results = m.delete_documents({suffix: doomed}, {suffix: vector_only})
        # A summary of the tree paraphrases what it covers: one citing a
        # removed text unit goes, and every summary above it (fail-closed:
        # the next summary_tree build writes the tree again).
        summaries = m.vector_indexer.delete_summary_nodes_citing(text_unit_ids, suffix)
        if summaries.total_items or summaries.failed_items:
            results[f"vector_summary_nodes_{suffix}"] = summaries
        if _rejected(results):
            # Strip nothing until the removed text units are gone: a retry
            # tells a removal that got further by their missing copies.
            return results
        deleted_communities = [i for i, u in citing.communities.items() if gone(u)]
        m.deleted_communities.setdefault(suffix, set()).update(deleted_communities)
        if deleted_communities:
            # A kept text unit may still name one: a later delta wrote the
            # community anew under its id without that unit.
            results[f"vector_text_unit_communities_{suffix}"] = (
                m.vector_indexer.prune_text_unit_communities(
                    deleted_communities, suffix
                )
            )
        if shared_entities or shared_relationships or shared_claims:
            results.update(
                m.remove_text_units_from_shared(
                    suffix,
                    entity_ids=shared_entities,
                    relationship_ids=shared_relationships,
                    text_unit_ids=text_unit_ids,
                    claim_ids=shared_claims,
                    community_ids=deleted_communities,
                    earlier_note=citing.note,
                )
            )
        elif citing.note is not None and not _rejected(results):
            # The noted items are gone: the earlier strip is finished.
            self._clear_pending_strip(suffix)
        return results

    def _find_citing(self, suffix: str, text_unit_ids: list[str]) -> "_Citing":
        """What cites the removed ``text_unit_ids`` in the stores (see
        :meth:`IndexingManager.remove_text_units`), and the note of an
        earlier strip; the extraction-store keys the units name are kept for
        :meth:`collect_extraction_records`. A read that fails raises."""
        m = self._m
        removed = set(text_unit_ids)
        graph, vector = m.graph_indexer, m.vector_indexer
        prefixes = m.config.indexing.opensearch
        found = graph.find_ids_citing(text_unit_ids, suffix=suffix)
        if found is None:
            logger.warning(
                "The graph store cannot find what text units state: the "
                "removed documents' entities and relationships stay"
            )
        # The vector store's copies are searched too: an interrupted run
        # may have written one without its graph copy.
        cited = {
            **m._vector_citing(prefixes.entities_index_prefix, text_unit_ids, suffix),
            **(found or {}),
        }
        # A vector copy the graph search missed may be stale (its write
        # was rejected) while its graph copy cites other units too (an
        # interrupted strip of these units rewrote it): the entity is
        # shared then, not vector-only, so its graph copy's units count.
        unfound = sorted(set(cited) - set(found)) if found is not None else []
        if unfound and _reads_back(graph):
            for entity in graph.read_entities(unfound, suffix=suffix):
                cited[entity.id] = sorted(
                    set(cited[entity.id]) | set(entity.text_unit_ids or [])
                )
        edges = self._graph_edges_citing(suffix, text_unit_ids, cited)
        relationships = {
            **m._vector_citing(
                prefixes.relationships_index_prefix, text_unit_ids, suffix
            ),
            **edges,
        }
        communities = (
            graph.find_ids_citing(text_unit_ids, suffix=suffix, communities=True) or {}
        )
        claims = m._vector_citing(prefixes.claims_index_prefix, text_unit_ids, suffix)
        reports = m._vector_citing(
            prefixes.community_reports_index_prefix, text_unit_ids, suffix
        )
        other_units = sorted(
            {
                unit_id
                for units in (*communities.values(), *reports.values())
                for unit_id in units
            }
            - removed
        )
        stored = vector.read_text_unit_ids(other_units, suffix=suffix)
        # A backend that cannot tell keeps every community still citing
        # other text units.
        live = set(other_units if stored is None else stored)
        m._removed_record_keys.setdefault(suffix, set()).update(
            key
            for attributes in vector.read_text_unit_attributes(
                text_unit_ids, suffix=suffix
            ).values()
            for key in attribute_record_keys(attributes).values()
        )
        # What an earlier strip merged or re-keyed and may not have
        # finished (see _note_pending_strip): the items need not cite the
        # removed units any more. Read on every removal (one record): a
        # retry need not remove the same units.
        notes = vector.read_extraction_records(
            [_PENDING_STRIP, _STRIPPED_SUMMARIES], suffix=suffix
        )
        note = notes.get(_PENDING_STRIP)
        # And the items an earlier strip noted for its commit, which may
        # not have run (see note_stripped_summaries).
        m.read_stripped_summaries(suffix, notes)
        # A claim the earlier strip re-keyed onto a noted entity cites no
        # removed unit: it is found through its subject or object.
        noted_claims = sorted(
            {
                *m._claims_about(suffix, _note_ids(note, "entity_ids")),
                *_note_ids(note, "claim_ids"),
            }
        )
        return _Citing(
            removed=removed,
            found=found,
            entities=cited,
            relationships=relationships,
            communities=communities,
            claims=claims,
            reports=reports,
            live=live,
            note=note,
            noted_claims=noted_claims,
        )

    def _graph_edges_citing(
        self, suffix: str, text_unit_ids: list[str], cited: Collection[str]
    ) -> dict[str, list[str]]:
        """``{id: text_unit_ids}`` of the graph edges citing a removed text
        unit, found among the edges of the ``cited`` entities."""
        m = self._m
        removed = set(text_unit_ids)
        graph, vector = m.graph_indexer, m.vector_indexer
        # The edges citing them are among the cited entities' edges,
        # except after an earlier removal of these units that got past
        # deleting them (their vector copies are gone) and died: it may
        # have stripped the entities and re-keyed an edge whose old graph
        # copy it did not get to delete. Only then are the graph's edges
        # looked up by text unit (on Neptune a scan of the suffix's edges).
        still = vector.read_text_unit_ids(text_unit_ids, suffix=suffix)
        resumed = still is None or len(set(still)) < len(removed)
        edges = (
            graph.find_relationship_ids_citing(text_unit_ids, suffix=suffix)
            if resumed
            else None
        )
        if edges is None:
            incident = (
                graph.find_incident_relationship_ids(sorted(cited), suffix=suffix)
                if cited
                else []
            )
            edges = {
                r.id: list(r.text_unit_ids or [])
                for r in graph.read_relationships(incident, suffix=suffix)
                if removed.intersection(r.text_unit_ids or [])
            }
        return edges

    @staticmethod
    def _shared_and_doomed(
        citing: "_Citing", text_unit_ids: list[str]
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """The entities, relationships and claims a removal strips (those
        citing other text units too, and those an earlier strip noted), and
        the ids it deletes."""
        exclusive, gone = citing.exclusive, citing.gone
        shared_entities = sorted(
            i for i, units in citing.entities.items() if not exclusive(units)
        )
        shared_relationships = sorted(
            i for i, units in citing.relationships.items() if not exclusive(units)
        )
        shared_claims = sorted(
            i for i, units in citing.claims.items() if not exclusive(units)
        )
        doomed = sorted(
            {
                *text_unit_ids,
                *(i for i, units in citing.entities.items() if exclusive(units)),
                *(i for i, units in citing.relationships.items() if exclusive(units)),
                *(i for i, units in citing.claims.items() if exclusive(units)),
                *(i for i, units in citing.communities.items() if gone(units)),
                *(i for i, units in citing.reports.items() if gone(units)),
            }
        )
        note = citing.note
        if note is not None:
            shared_entities = sorted(
                (set(shared_entities) | _note_ids(note, "entity_ids")) - set(doomed)
            )
            shared_relationships = sorted(
                (set(shared_relationships) | _note_ids(note, "relationship_ids"))
                - set(doomed)
            )
            shared_claims = sorted(
                (set(shared_claims) | set(citing.noted_claims)) - set(doomed)
            )
        return shared_entities, shared_relationships, shared_claims, doomed

    def remove_text_units_from_shared(
        self,
        suffix: str,
        *,
        entity_ids: list[str],
        relationship_ids: list[str],
        text_unit_ids: list[str],
        claim_ids: Sequence[str] = (),
        community_ids: Collection[str] = (),
        earlier_note: Mapping[str, Any] | None = None,
    ) -> dict[str, IndexingStats]:
        """See :meth:`IndexingManager.remove_text_units_from_shared`."""
        m = self._m
        results, noted = self._strip_shared(
            suffix,
            entity_ids=entity_ids,
            relationship_ids=relationship_ids,
            text_unit_ids=text_unit_ids,
            claim_ids=claim_ids,
            community_ids=community_ids,
            earlier_note=earlier_note,
        )
        # A note is kept until the old ids are deleted (a retry needs it).
        if _rejected(results):
            m.discard_retired_ids()
            return results
        results = {**results, **m.delete_retired_ids()}
        if (noted or earlier_note is not None) and not _rejected(results):
            self._clear_pending_strip(suffix)
        return results

    def _note_pending_strip(
        self,
        suffix: str,
        entity_ids: list[str],
        relationship_ids: list[str],
        claim_ids: list[str] | None = None,
    ) -> None:
        """Note the stored items a strip merges or re-keys, in one record of
        ``suffix``'s extraction store (kept until :meth:`_clear_pending_strip`),
        so a retry of the removal strips them again (:meth:`remove_text_units`),
        with the ``claim_ids`` about the entities it splits (a split keeps the
        stored id, so a retry would not find them).
        Raises when the write fails, before the strip writes anything."""
        m = self._m
        m._write_note(
            _PENDING_STRIP,
            suffix,
            {
                "entity_ids": entity_ids,
                "relationship_ids": relationship_ids,
                "claim_ids": claim_ids or [],
            },
        )

    def _clear_pending_strip(self, suffix: str) -> None:
        """Delete the note of :meth:`_note_pending_strip` (best-effort: one
        left only makes a retry strip the items again)."""
        m = self._m
        m._delete_note(_PENDING_STRIP, suffix)

    def _strip_shared(
        self,
        suffix: str,
        *,
        entity_ids: list[str],
        relationship_ids: list[str],
        text_unit_ids: list[str],
        claim_ids: Sequence[str] = (),
        community_ids: Collection[str] = (),
        earlier_note: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, IndexingStats], bool]:
        """:meth:`remove_text_units_from_shared` before the old ids of the
        re-keyed items are deleted, and whether it noted them
        (:meth:`_note_pending_strip`; with the ``earlier_note`` items, so
        the note it replaces is not lost).

        In order: the stored items are read and derived again
        (:meth:`_read_shared`, :meth:`_derive_stripped`), the run state
        records what the derivation decided (:meth:`_record_strip`), the
        note's ids (:meth:`_strip_note_ids`) and the claims that follow
        (:meth:`_claims_to_strip`) are found, the claims are stripped, the
        note is written (once, before the first write: :class:`_StripNote`)
        and the items are written (:meth:`_write_stripped`). A rejected
        write stops it (all or retry, see the module docstring)."""
        results: dict[str, IndexingStats] = {}
        note = _StripNote(self, suffix, earlier_note)
        doing = "Reading shared artifacts"
        try:
            entities, relationships, absent = self._read_shared(
                suffix, entity_ids, relationship_ids
            )
            derived, new_entities, new_relationships, made = self._derive_stripped(
                suffix, entities, relationships, text_unit_ids, community_ids
            )
            self._record_strip(
                suffix, derived, new_entities, new_relationships, absent, made
            )
            note.ids, split_claims = self._strip_note_ids(
                suffix, derived, relationships
            )
            doing = "Reading the claims of re-keyed entities"
            claim_ids = self._claims_to_strip(
                suffix, derived, entities, claim_ids, split_claims
            )
        except Exception as e:  # noqa: BLE001 - reported as failed items
            logger.error("%s in '%s' failed: %s", doing, suffix, e)
            stats = IndexingStats(total_items=len(entity_ids) + len(relationship_ids))
            stats.add_error(str(e), count=stats.total_items)
            return {**results, f"graph_read_{suffix}": stats}, note.noted

        if claim_ids:
            results.update(
                self._remove_text_units_from_claims(
                    suffix, list(claim_ids), text_unit_ids, note=note
                )
            )
            if _rejected(results):
                return results, note.noted
        try:
            note(set())
        except Exception as e:  # noqa: BLE001 - reported as failed items
            logger.error("Noting the pending strip in '%s' failed: %s", suffix, e)
            stats = IndexingStats(total_items=len(entity_ids) + len(relationship_ids))
            stats.add_error(str(e), count=stats.total_items)
            return {**results, f"graph_read_{suffix}": stats}, note.noted
        if not new_entities and not new_relationships:
            return results, note.noted
        try:
            self._m.note_stripped_summaries(suffix)
        except Exception as e:  # noqa: BLE001 - reported as failed items
            logger.error("Noting the stripped summaries in '%s' failed: %s", suffix, e)
            stats = IndexingStats(total_items=len(entity_ids) + len(relationship_ids))
            stats.add_error(str(e), count=stats.total_items)
            return {**results, f"graph_read_{suffix}": stats}, note.noted
        logger.info(
            "Removing %d text units from %d shared entities and %d shared "
            "relationships in '%s'",
            len(text_unit_ids),
            len(entities),
            len(relationships),
            suffix,
        )
        self._write_stripped(
            suffix,
            entities,
            relationships,
            new_entities,
            new_relationships,
            results,
        )
        return results, note.noted

    def _read_shared(
        self, suffix: str, entity_ids: list[str], relationship_ids: list[str]
    ) -> tuple[list[Entity], list[Relationship], tuple[set[str], set[str]]]:
        """The stored entities and relationships a strip derives again, with
        the endpoints of the edges, and the ids among ``entity_ids`` and
        ``relationship_ids`` the graph does not hold. A read that fails
        raises."""
        m = self._m
        entities = (
            m.graph_indexer.read_entities(entity_ids, suffix=suffix)
            if entity_ids
            else []
        )
        relationships = (
            m.graph_indexer.read_relationships(relationship_ids, suffix=suffix)
            if relationship_ids
            else []
        )
        # A shared id a graph store with read-back does not hold is a
        # vector copy whose graph write an interrupted run never made:
        # nothing strips it, so it goes with the retired ids (with the
        # vector edges at it, see delete_documents).
        # Retired once the derivation is known: a noted id an earlier
        # strip retired may be derived again (see _record_strip).
        absent: tuple[set[str], set[str]] = (set(), set())
        if _reads_back(m.graph_indexer):
            absent = (
                set(entity_ids) - {e.id for e in entities},
                set(relationship_ids) - {r.id for r in relationships},
            )
        # An edge is derived with its endpoints (the resolved entities map
        # to them). They cite the removed units too, unless a killed run
        # stripped them and left the edge (re-keyed, not yet deleted).
        ends = {i for r in relationships for i in (r.source_id, r.target_id)}
        ends -= {e.id for e in entities}
        if ends:
            entities += m.graph_indexer.read_entities(sorted(ends), suffix=suffix)
        entities = m._with_full_descriptions(
            entities, m.vector_indexer.read_entity_descriptions, suffix
        )
        relationships = m._with_full_descriptions(
            relationships,
            m.vector_indexer.read_relationship_descriptions,
            suffix,
        )
        return entities, relationships, absent

    def _derive_stripped(
        self,
        suffix: str,
        entities: list[Entity],
        relationships: list[Relationship],
        text_unit_ids: list[str],
        community_ids: Collection[str],
    ) -> tuple[Derived, list[Entity], list[Relationship], set[str]]:
        """The stored items stripped of ``text_unit_ids`` and derived again:
        the derivation, the entities and relationships to write
        (``_finish``) and the ids of the items it summarized."""
        m = self._m
        stripped_entities, stripped_relationships = remove_text_units(
            entities, relationships, text_unit_ids, community_ids
        )
        # Every item read back is rewritten, stripped or not: the graph
        # copy is the reference, and after a strip that was interrupted
        # or partly failed between the stores it no longer cites the
        # removed text units while the vector copy still does.
        stripped = {item.id: item for item in stripped_entities}
        stripped_rels = {item.id: item for item in stripped_relationships}
        made: set[str] = set()
        derived = m.derive(
            suffix,
            [stripped.get(e.id, e) for e in entities],
            [stripped_rels.get(r.id, r) for r in relationships],
            (),
        )
        new_entities, new_relationships = m._merge._finish(
            suffix,
            derived.entities,
            derived.relationships,
            stored_entities=entities,
            stored_relationships=relationships,
            rekey=derived.id_remap,
            rewired=set(derived.rewired),
            dropped=derived.dropped,
            split_from=derived.split_from,
            unsummarized=m._delta_ids.get(suffix, set()),
            made=made,
        )
        return derived, new_entities, new_relationships, made

    def _record_strip(
        self,
        suffix: str,
        derived: Derived,
        new_entities: list[Entity],
        new_relationships: list[Relationship],
        absent: tuple[set[str], set[str]],
        made: set[str],
    ) -> None:
        """Record in the run state what a strip's derivation decided: the
        ids it retires, the vector-only copies (``absent``, see
        :meth:`_read_shared`), the entities it derived and the items whose
        stored summary the commit must not extend."""
        m = self._m
        m._merge._retire(suffix, derived.id_remap, derived.split_from, derived.dropped)
        # Not an absent id this strip writes (a part split off again
        # under the id an earlier, interrupted strip re-keyed away).
        written = {e.id for e in new_entities} | {r.id for r in new_relationships}
        vector_only = absent[0] - written
        if vector_only:
            m._vector_only.setdefault(suffix, set()).update(vector_only)
        # Edges alike: an edge an interrupted strip re-keyed away is kept
        # when this strip writes it again under that id.
        vector_only |= absent[1] - written
        if vector_only:
            m._dropped.setdefault(suffix, set()).update(vector_only)
        m._derived_entities.setdefault(suffix, {}).update(
            {e.id: e for e in new_entities}
        )
        # Not extended by the commit (see CrossRunMerge): what this strip
        # summarized, and every item it wrote unsummarized — what it left
        # for the commit to summarize, and a spelling folding into it. Their
        # stored summary (if any) was made of lines the strip removed, so
        # extending it would summarize a summary, or keep removed lines,
        # where a full build summarizes the lines once.
        m._stripped_ids.setdefault(suffix, set()).update(
            made
            | {e.id for e in new_entities if not _is_summary(e)}
            | {r.id for r in new_relationships if not _is_summary(r)}
        )

    def _strip_note_ids(
        self,
        suffix: str,
        derived: Derived,
        relationships: Sequence[Relationship] = (),
    ) -> tuple[tuple[set[str], set[str], set[str]] | None, set[str]]:
        """The ids a strip notes before its first write (``None``: none),
        and the stored claims about a split's part and origin
        (``relationships``: the stored copies of the edges it derives)."""
        m = self._m
        entity_ids_now = {e.id for e in derived.entities}
        edge_ids_now = {r.id for r in derived.relationships}
        claim_ids_now = {c.id for c in derived.claims}
        merged = {
            i
            for pair in derived.id_remap.items()
            if pair[1] in entity_ids_now
            for i in pair
        }
        # The old ids of the edges re-keyed into a derived edge, and of
        # those dropped: no claim.
        retired_edges = {
            old for old, new in derived.id_remap.items() if new in edge_ids_now
        } | {i for i in derived.dropped if i not in claim_ids_now}
        # And the kept edges it re-pointed (at a split's part, say): another
        # stored edge there may be the same edge under another id, which
        # the strip folds once it writes this one (CrossRunMerge._finish).
        # A retry reaches the edge through the entities it left no more, so
        # without the note the duplicate stayed. And the edges under a new
        # id, as the claims (_StripNote): no copy of one cites a removed
        # unit, and a strip stopped or killed after its graph copy left no
        # vector copy, so a retry rewrote it from the graph copy, its
        # description cut at the graph's limit.
        retired_edges |= self._unfindable_edges(suffix, derived, relationships)
        # The stored claims about a split's part and origin: a retry
        # cannot tell them (the split kept the stored id).
        split_claims = (
            set(
                m._claims_about(
                    suffix, [*derived.split_from, *derived.split_from.values()]
                )
            )
            if derived.split_from
            else set()
        )
        if merged or retired_edges or split_claims:
            # Their old ids need not cite the removed text units (another
            # spelling merged in, or an edge re-pointed at it): once the
            # items are written, a retry finds them through this note
            # only, and strips them again. Noted: the entities on both
            # sides of a merge, the edges re-keyed or dropped and the
            # claims about a split's entities. Not a split's part and
            # origin themselves: a split keeps the stored id, and
            # stripping them again on a retry gave a claim the wrong
            # subject.
            return (merged, retired_edges, split_claims), split_claims
        return None, split_claims

    def _unfindable_edges(
        self,
        suffix: str,
        derived: Derived,
        relationships: Sequence[Relationship],
    ) -> set[str]:
        """The ids of the edges ``derived`` writes that a retry would not
        find by the removed text units: the stored ones it keeps under their
        id with other endpoints, and those the graph does not hold
        (``relationships``: the stored copies read; the edges the derivation
        reached beyond them are read here)."""
        stored = {r.id: r for r in relationships}
        reached = [r.id for r in derived.relationships if r.id not in stored]
        if reached:
            stored.update(
                (r.id, r)
                for r in self._m.graph_indexer.read_relationships(
                    reached, suffix=suffix
                )
            )
        return {
            r.id
            for r in derived.relationships
            if r.id not in stored
            or (r.source_id, r.target_id)
            != (stored[r.id].source_id, stored[r.id].target_id)
        }

    def _claims_to_strip(
        self,
        suffix: str,
        derived: Derived,
        entities: Sequence[Entity],
        claim_ids: Sequence[str],
        split_claims: set[str],
    ) -> list[str]:
        """``claim_ids``, the ``split_claims`` and the stored claims that
        follow the entities a strip's derivation re-keyed, split or renamed
        (``entities``: the stored copies)."""
        m = self._m
        # The stored claims about a re-keyed, split or renamed entity
        # follow it.
        renamed = m._merge._note_renamed(suffix, derived.entities, entities)
        # And those about the other stored spellings of their names: a
        # claim naming one may resolve to the changed entity now.
        changed = {*derived.id_remap.values(), *renamed, *derived.split_from}
        family_keys = sorted(
            {compact_name_key(e.name) for e in derived.entities if e.id in changed}
        )
        family = (
            m.graph_indexer.find_entity_ids_by_name_key(family_keys, suffix)
            if family_keys
            else None
        ) or []
        return sorted(
            {
                *claim_ids,
                *split_claims,
                *m._claims_about(
                    suffix,
                    [
                        *derived.id_remap,
                        *derived.split_from.values(),
                        *_other_spellings_of_parts(
                            derived.split_from,
                            {e.id: e for e in derived.entities},
                        ),
                        *renamed,
                        *family,
                    ],
                ),
            }
        )

    def _write_stripped(
        self,
        suffix: str,
        entities: Sequence[Entity],
        relationships: Sequence[Relationship],
        new_entities: list[Entity],
        new_relationships: list[Relationship],
        results: dict[str, IndexingStats],
    ) -> None:
        """Write a strip's derived items into ``results`` (``entities`` and
        ``relationships``: the stored copies read back), phase by phase in
        the order each depends on the one before; a phase with a rejected
        write stops the strip (see the module's *all or retry*)."""
        m = self._m
        # The vector store re-embeds only what changed: compared with the
        # vector copies themselves (name, type, confidence or endpoints, and
        # description), so a strip interrupted between the stores is repaired
        # whichever store it reached.
        # Annotated as they are written (_write_artifacts), so a copy whose
        # annotations the schema changed is written in full.
        new_entities = m._annotated_entities(new_entities)
        new_relationships = m._annotated_relationships(new_relationships)
        graph_copy = {e.id for e in entities} | {r.id for r in relationships}
        vector_copy = {
            **m.vector_indexer.read_vector_copies(
                "entities", [e.id for e in new_entities], suffix=suffix
            ),
            **m.vector_indexer.read_vector_copies(
                "relationships", [r.id for r in new_relationships], suffix=suffix
            ),
        }

        def unchanged(item: Entity | Relationship) -> bool:
            stored = vector_copy.get(item.id)
            return (
                stored is not None
                and not _vector_stale(item, stored)
                and (stored.get("description") or None) == (item.description or None)
            )

        # The new vertices (a re-keyed entity's, a split's part) come first:
        # the edges are re-pointed at them, and the stored copies they are
        # derived from (a re-keyed entity's old vertex, a split's origin)
        # still hold their lines until they are written.
        # Then the stored entities that gain a unit their graph copy does
        # not cite (another spelling's statement merged in, a split's origin
        # may), before the edges re-pointed at them: written after them, a
        # run killed in between left an edge citing a kept unit on a vertex
        # citing only removed ones, which the next removal deleted with the
        # edge.
        stored_units = {e.id: set(e.text_unit_ids or []) for e in entities}
        new_vertices = [e for e in new_entities if e.id not in graph_copy]
        gaining = [
            e
            for e in new_entities
            if e.id in graph_copy
            and set(e.text_unit_ids or []) - stored_units.get(e.id, set())
        ]
        written_first = {e.id for e in (*new_vertices, *gaining)}
        # The graph copies of the kept entities are what a retry finds the
        # strip by: written last.
        kept = [e for e in new_entities if e.id not in written_first]
        # The graph copies of the edges come before their vector copies: a
        # run killed in between leaves no vector edge without a graph edge,
        # and a vector copy behind is rewritten (``unchanged``, ``_finish``).
        # A stored edge rewritten for its rank alone has it set in place,
        # as a commit does (read before the vector write takes them).
        relationship_tasks: list[IndexingTask] = []
        if m.config.indexing.opensearch.build_relationship_vector_index:
            relationship_tasks += [
                IndexingTask(
                    m.vector_indexer.update_relationship_lineage,
                    [[r for r in new_relationships if unchanged(r)]],
                    "vector_relationship_lineage",
                ),
                IndexingTask(
                    m._upsert_vector_relationships,
                    [[r for r in new_relationships if not unchanged(r)]],
                    "vector_relationships",
                ),
            ]
        phases = [
            [
                IndexingTask(
                    m.graph_indexer.upsert_entities,
                    [new_vertices],
                    "graph_new_entities",
                )
            ],
            [
                IndexingTask(
                    m.graph_indexer.upsert_entities,
                    [gaining],
                    "graph_gaining_entities",
                )
            ],
            [
                IndexingTask(
                    partial(m._upsert_graph_relationships, set(m._rank_only)),
                    [new_relationships],
                    "graph_relationships",
                )
            ],
            relationship_tasks,
            [
                IndexingTask(
                    m.vector_indexer.update_entity_lineage,
                    [[e for e in new_entities if unchanged(e)]],
                    "vector_entity_lineage",
                ),
                IndexingTask(
                    m.vector_indexer.upsert_entities,
                    [[e for e in new_entities if not unchanged(e)]],
                    "vector_entities",
                ),
            ],
            [IndexingTask(m.graph_indexer.upsert_entities, [kept], "graph_entities")],
        ]
        for phase in phases:
            check_write_fence()
            done = m._run_indexing_phase(phase)
            results.update(done)
            if _rejected(done):
                return

    def _remove_text_units_from_claims(
        self,
        suffix: str,
        claim_ids: list[str],
        text_unit_ids: list[str],
        note: Callable[[set[str]], None] | None = None,
    ) -> dict[str, IndexingStats]:
        m = self._m
        try:
            # The write-back groups by the claim's own index tag, so tag what
            # was read with the index it came from.
            claims = m._read_stored_claims(claim_ids, suffix)
            # The claims citing none of them were read because their subject
            # or object was re-keyed: they are derived again too.
            stripped = {c.id: c for c in remove_claim_text_units(claims, text_unit_ids)}
            claims = [stripped.get(c.id, c) for c in claims]
            if not claims:
                return {}
            derived = m._merge._derive_claims(suffix, claims, removed=text_unit_ids)
            if note is not None:
                # Before the first claim write; raises when the note fails.
                note({d.id for d in derived} - {c.id for c in claims})
        except Exception as e:  # noqa: BLE001 - reported as failed items
            logger.error("Reading shared claims in '%s' failed: %s", suffix, e)
            stats = IndexingStats(total_items=len(claim_ids))
            stats.add_error(str(e), count=len(claim_ids), ids=claim_ids)
            return {f"vector_read_claims_{suffix}": stats}
        before = {c.id: c for c in claims}
        # The claims under new ids first: the stored claims they are derived
        # from (a split's) hold their lines until those are written.
        lineage_only = {
            d.id
            for d in derived
            if d.id in before and _claim_text(d) == _claim_text(before[d.id])
        }
        return {
            f"vector_claims_{suffix}": write_claims_new_ids_first(
                m.vector_indexer, derived, before, lineage_only
            )
        }

    def collect_extraction_records(self) -> dict[str, IndexingStats]:
        """See :meth:`IndexingManager.collect_extraction_records`."""
        m = self._m
        results: dict[str, IndexingStats] = {}
        vector = m.vector_indexer
        prefix = m.config.indexing.opensearch.text_units_index_prefix
        # The records of a kind this run does not produce (claims with
        # claims off) stay, as a reset keeps them: the units it extracted
        # again name none, and a run that builds the kind again reuses them.
        unbuilt = tuple(unbuilt_record_prefixes(m.config, artifacts_to_build(m.config)))
        for suffix, keys in sorted(m._removed_record_keys.items()):
            try:
                naming: set[str] = set()
                for kind in RECORD_KINDS:
                    naming.update(
                        vector.find_documents(
                            prefix,
                            f"attributes.{RECORD_KEYS_ATTRIBUTE}.{kind}",
                            sorted(keys),
                            suffix,
                        )
                        or {}
                    )
                still_named = {
                    key
                    for attributes in vector.read_text_unit_attributes(
                        sorted(naming), suffix=suffix
                    ).values()
                    for key in attribute_record_keys(attributes).values()
                }
                unused = sorted(
                    key
                    for key in keys - still_named
                    if not (unbuilt and key.startswith(unbuilt))
                )
                if unused:
                    logger.info(
                        "Deleting %d extraction records no text unit of '%s' "
                        "names any more",
                        len(unused),
                        suffix,
                    )
                    results[f"extraction_records_{suffix}"] = (
                        vector.delete_extraction_records(unused, suffix=suffix)
                    )
            except NamespaceLeaseError:
                raise
            except Exception as e:  # noqa: BLE001 - garbage only costs space
                logger.warning(
                    "Collecting the unused extraction records of '%s' failed: %s",
                    suffix,
                    e,
                )
        m._removed_record_keys.clear()
        return results

    def delete_documents(
        self,
        ids_by_suffix: dict[str, list[str]],
        vector_only: Mapping[str, Collection[str]] | None = None,
    ) -> dict[str, IndexingStats]:
        """See :meth:`IndexingManager.delete_documents`."""
        m = self._m
        results: dict[str, IndexingStats] = {}
        for suffix, ids in ids_by_suffix.items():
            if not ids:
                continue
            check_write_fence()
            # Orphan-edge cleanup: a relationship pointing AT one of these
            # deleted entities is owned (in lineage) by a SURVIVING document, so
            # it is not in `ids`. Neptune drops the incident edge on vertex drop,
            # but its OpenSearch relationship document would survive as a
            # dangling edge. Collect those incident edge ids BEFORE the Neptune
            # drop, then fold them into the relationship-index deletion below.
            try:
                incident_rel_ids = m.graph_indexer.find_incident_relationship_ids(
                    ids, suffix=suffix
                )
                # And the edges at a vector-only entity (see _strip_shared):
                # their graph copies never were either.
                incident_rel_ids = sorted(
                    {
                        *incident_rel_ids,
                        *m._vector_incident_relationship_ids(
                            sorted(set((vector_only or {}).get(suffix, ())) & set(ids)),
                            suffix,
                        ),
                    }
                )
            except Exception as e:  # noqa: BLE001 - reported as failed items
                # Deleting without them would orphan their vector documents
                # for good; fail the removal so the registry rows are kept.
                logger.error("Reading incident edges in '%s' failed: %s", suffix, e)
                stats = IndexingStats(total_items=len(ids))
                stats.add_error(str(e), count=len(ids))
                results[f"graph_read_{suffix}"] = stats
                continue
            # The vector copies first: a retry finds what to delete (and the
            # re-keyed items it retires) through the graph, so a run that
            # died between the stores must leave the graph copy, not an
            # orphaned vector copy nothing finds again.
            results.update(
                m.vector_indexer.delete_document_artifacts(
                    ids, suffix, extra_relationship_ids=incident_rel_ids
                )
            )
            results[f"graph_delete_{suffix}"] = m.graph_indexer.delete_by_id(
                ids, suffix=suffix
            )
        return results


class _Citing(NamedTuple):
    """What cites the text units a removal removes
    (:meth:`TextUnitRemoval._find_citing`), each by id with the text units
    it cites."""

    removed: set[str]
    # The graph's entities (``None``: the graph store cannot tell).
    found: dict[str, list[str]] | None
    # The graph's and the vector store's entities.
    entities: dict[str, list[str]]
    relationships: dict[str, list[str]]
    communities: dict[str, list[str]]
    claims: dict[str, list[str]]
    reports: dict[str, list[str]]
    # The other text units the communities and reports cite still stored.
    live: set[str]
    # The note of an earlier strip, and the claims it names or is about.
    note: dict[str, Any] | None
    noted_claims: list[str]

    def exclusive(self, units: Iterable[str]) -> bool:
        """Whether ``units`` are all removed."""
        return set(units) <= self.removed

    def gone(self, units: Iterable[str]) -> bool:
        """Whether none of ``units`` but the removed ones is stored."""
        return not (set(units) - self.removed) & self.live


class _StripNote:
    """The write-ahead note of one strip
    (:meth:`TextUnitRemoval._note_pending_strip`): written once, before the
    strip's first write, with the ``ids`` the strip notes (``None``: none)
    and the items of the ``earlier_note`` it replaces.
    Called with the claims the strip derived under new ids (their subject or
    object split, re-keyed or renamed): they cite kept text units and are
    written before the entities, so a strip killed in between leaves them to
    a retry that may not derive them again. Raises when the write fails."""

    def __init__(
        self,
        removal: "TextUnitRemoval",
        suffix: str,
        earlier_note: Mapping[str, Any] | None,
    ) -> None:
        self._removal = removal
        self._suffix = suffix
        self._earlier = earlier_note
        self.ids: tuple[set[str], set[str], set[str]] | None = None
        # Whether the note was written (by this strip).
        self.noted = False

    def __call__(self, new_claims: set[str]) -> None:
        if self.noted or (self.ids is None and not new_claims):
            return
        entities_, edges_, claims_ = self.ids or (set(), set(), set())
        earlier = self._earlier
        self._removal._note_pending_strip(
            self._suffix,
            sorted(entities_ | _note_ids(earlier, "entity_ids")),
            sorted(edges_ | _note_ids(earlier, "relationship_ids")),
            sorted(claims_ | new_claims | _note_ids(earlier, "claim_ids")),
        )
        self.noted = True


def _rejected(results: Mapping[str, IndexingStats]) -> bool:
    """Whether a store rejected a write (or a read failed) of ``results``."""
    return any(stats.failed_items for stats in results.values() if stats)
