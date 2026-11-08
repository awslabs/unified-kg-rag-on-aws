# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cross-run merge: a delta, or a strip, meets the stored graph it
touches, and the items it writes are derived from their text units' records."""

from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from unified_kg_rag.adapters.ingestion.description_summarizer import (
    extend_summaries,
    needs_summary,
    reuse_summaries,
)
from unified_kg_rag.application.storage.stored_items import (
    _PENDING_COMMIT,
    _STRIPPED_SUMMARIES,
    _IncidentEdges,
    _is_summary,
    _note_ids,
    _other_spellings,
    _reads_back,
    _vector_stale,
    _with_old_ids,
    unit_records,
)
from unified_kg_rag.domain.ingestion.base_resolver import FuzzyMatcher, compact_name_key
from unified_kg_rag.domain.ingestion.derivation import Derived, derive
from unified_kg_rag.domain.ingestion.graph_rank import with_graph_ranks
from unified_kg_rag.domain.ingestion.merge import (
    merge_claims,
    merge_entities,
    merge_relationships,
    relationship_id_remap,
    relationship_key,
    remove_text_units,
)
from unified_kg_rag.domain.ingestion.relationship_endpoints import (
    sync_relationship_endpoint_names,
)
from unified_kg_rag.domain.ingestion.seed_graph import is_document_backed
from unified_kg_rag.domain.models import (
    Claim,
    Constants,
    Entity,
    Relationship,
    TextUnit,
    TextUnitExtraction,
)
from unified_kg_rag.ports.indexer import BaseIndexer
from unified_kg_rag.shared import get_logger

if TYPE_CHECKING:
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

logger = get_logger(__name__)

# Rounds ``IndexingManager.derive`` reads more stored items in at most. Each
# round that goes on adds an item, so the loop ends by itself; the cap
# bounds the reads of a cascade over a large connected graph.
_DERIVE_ROUNDS = 32


class CrossRunMergeResult(BaseModel):
    """A delta merged with the stored graph state it touches."""

    entities: list[Entity] | None = None
    relationships: list[Relationship] | None = None
    # Per index suffix: delta entity/relationship id -> the stored id it merged
    # into, so lineage can record the ids actually written.
    id_remap_by_suffix: dict[str, dict[str, str]] = Field(default_factory=dict)


def _single_additions(
    added: Sequence[tuple[str, Entity | Relationship]],
    stored: Sequence[Entity | Relationship],
    excluded: Collection[str],
    derived: Sequence[Entity | Relationship],
) -> dict[str, str | None]:
    """``{final id: added description}`` of the ``added`` items (paired with
    the id each ends under) whose final id one delta item and at most one
    ``stored`` copy end under, which is not ``excluded``, and whose
    ``derived`` item cites every text unit the delta item cites (a
    derivation that moved a unit's statement to another item, an edge of a
    split entity, say, leaves the delta description stating it) and no
    other than those and its stored copy's (a derivation that folded another
    stored spelling into it adds that spelling's lines, which neither
    description states): the items :func:`extend_summaries` may extend."""
    reached = Counter(final for final, _ in added)
    held = Counter(item.id for item in stored)
    stored_units: dict[str, set[str]] = {}
    for item in stored:
        stored_units.setdefault(item.id, set()).update(item.text_unit_ids or [])
    cited = {item.id: set(item.text_unit_ids or []) for item in derived}
    return {
        final: item.description
        for final, item in added
        if final not in excluded
        and reached[final] == 1
        and held[final] <= 1
        and set(item.text_unit_ids or []) <= cited.get(final, set())
        and cited.get(final, set())
        <= set(item.text_unit_ids or []) | stored_units.get(final, set())
    }


class CrossRunMerge:
    """The cross-run merge of an incremental commit and the derivation of
    stored items, for :class:`IndexingManager` (whose stores and run state it
    uses): what a delta, a strip or a claim merge writes."""

    def __init__(self, manager: "IndexingManager") -> None:
        self._m = manager

    def merge_with_existing_graph(
        self,
        entities: list[Entity] | None,
        relationships: list[Relationship] | None,
        text_units: Sequence[TextUnit] = (),
    ) -> CrossRunMergeResult:
        """See :meth:`IndexingManager.merge_with_existing_graph`."""
        m = self._m
        entity_groups = m._group_by_suffix(entities or [])
        relationship_groups = m._group_by_suffix(relationships or [])
        merged_entities: list[Entity] = []
        merged_relationships: list[Relationship] = []
        id_remap_by_suffix: dict[str, dict[str, str]] = {}
        # A suffix this run touches without a delta (it only removed there)
        # still finishes a killed commit's note, and deletes it.
        delta_suffixes = entity_groups.keys() | relationship_groups.keys()
        noted_only = {
            suffix
            for suffix in sorted(
                {
                    m.config.processing.document_parsing.index_value
                    or Constants.DEFAULT_SUFFIX.value,
                    *m._removed_record_keys,
                }
                - delta_suffixes
            )
            if _PENDING_COMMIT
            in m.vector_indexer.read_extraction_records(
                [_PENDING_COMMIT], suffix=suffix
            )
        }
        for suffix in sorted(delta_suffixes | noted_only):
            suffix_entities, suffix_relationships, id_remap = self._merge_suffix(
                suffix,
                entity_groups.get(suffix, []),
                relationship_groups.get(suffix, []),
                [u for u in text_units if BaseIndexer.get_suffix(u) == suffix],
            )
            merged_entities.extend(suffix_entities)
            merged_relationships.extend(suffix_relationships)
            if id_remap:
                id_remap_by_suffix[suffix] = id_remap
        return CrossRunMergeResult(
            entities=merged_entities if entities is not None else None,
            relationships=(merged_relationships if relationships is not None else None),
            id_remap_by_suffix=id_remap_by_suffix,
        )

    def merge_with_stored_claims(
        self, claims: list[Claim] | None, text_units: Sequence[TextUnit] = ()
    ) -> list[Claim] | None:
        """See :meth:`IndexingManager.merge_with_stored_claims`."""
        m = self._m
        if not (
            claims
            or m._rekeyed
            or m._split_from
            or m._renamed
            or m._noted_commit_entities
        ):
            return claims
        groups = m._group_by_suffix(claims or [])
        merged: list[Claim] = []
        for suffix in sorted(
            groups.keys()
            | m._rekeyed.keys()
            | m._split_from.keys()
            | m._renamed.keys()
            | m._noted_commit_entities.keys()
        ):
            group = groups.get(suffix, [])
            stored = m._read_stored_claims(sorted({c.id for c in group}), suffix)
            group = merge_claims(stored, group) if stored else group
            units = [u for u in text_units if BaseIndexer.get_suffix(u) == suffix]
            # The stored claims about an entity the graph merge re-keyed
            # follow it (see merge_with_existing_graph), and those citing a
            # delta text unit are derived with the delta's: a resumed
            # removal may have derived one the killed run wrote under an id
            # this delta does not write.
            citing = (
                m._vector_citing(
                    m.config.indexing.opensearch.claims_index_prefix,
                    [u.id for u in units],
                    suffix,
                )
                if units
                else {}
            )
            about = sorted(
                (
                    set(m._claims_about(suffix, self._rederived_entity_ids(suffix)))
                    | set(citing)
                )
                - {c.id for c in group}
            )
            group += m._read_stored_claims(about, suffix)
            if not group:
                continue
            if not stored and not about and suffix in m._fresh:
                # Nothing stored: the delta claims are the full build's.
                merged.extend(group)
                continue
            derived = self._derive_claims(suffix, group, units)
            # A stored claim citing a unit a derived claim gained (this run's
            # removal derived it from a kept unit, before the delta's
            # entities were in the graph) may state what the derived claim
            # does under another id: it is derived with them. Under a killed
            # commit's note, the removal may have derived it from a cited
            # unit too (against the commit's unfinished graph): every unit
            # the derived claims cite is checked.
            gained = sorted(
                {u for c in derived for u in c.text_unit_ids or []}
                - (
                    set()
                    if m._noted_commit_entities.get(suffix)
                    else {u for c in group for u in c.text_unit_ids or []}
                )
            )
            overlapping = (
                set(
                    m._vector_citing(
                        m.config.indexing.opensearch.claims_index_prefix,
                        gained,
                        suffix,
                    )
                )
                - {c.id for c in [*group, *derived]}
                if gained
                else set()
            )
            if overlapping:
                group += m._read_stored_claims(sorted(overlapping), suffix)
                derived = self._derive_claims(suffix, group, units)
            merged.extend(derived)
        return merged if claims is not None or merged else claims

    def _merge_suffix(
        self,
        suffix: str,
        entities: list[Entity],
        relationships: list[Relationship],
        text_units: Sequence[TextUnit] = (),
    ) -> tuple[list[Entity], list[Relationship], dict[str, str]]:
        """Cross-run merge of one suffix's delta (see merge_with_existing_graph).

        Returns the entities and relationships to write plus ``{delta id: kept
        id}`` for every delta entity or relationship that merged into a stored
        one under a different id.
        """
        m = self._m
        merged_entities, existing, entity_id_remap, noted = self._merge_entities(
            suffix, entities
        )
        merged_relationships, existing_rels, relationship_remap = (
            self._merge_relationships(
                suffix, relationships, merged_entities, entity_id_remap
            )
        )
        # The vector copies of the noted entities and their edges the graph
        # does not hold: a commit killed after its vector writes and before
        # its graph writes left them, and nothing else finds them.
        vector_orphans: set[str] = set()
        if noted:
            (
                existing,
                existing_rels,
                merged_entities,
                merged_relationships,
                vector_orphans,
            ) = self._with_noted_edges(
                suffix,
                noted,
                existing,
                existing_rels,
                merged_entities,
                merged_relationships,
            )
        if merged_entities is not entities and merged_relationships:
            # Merged with stored entities: an edge re-pointed at one, or at one
            # whose stored name differs from the delta's, must name it as the
            # entity does.
            merged_relationships = [r.model_copy() for r in merged_relationships]
            sync_relationship_endpoint_names(merged_relationships, merged_entities)
        if existing or existing_rels:
            derived = m.derive(
                suffix, merged_entities, merged_relationships, (), text_units
            )
        else:
            # Nothing stored (a first build with the registry on): the delta
            # is what a full build over its records gives, summaries included.
            m._fresh.add(suffix)
            derived = Derived(
                entities=list(merged_entities),
                relationships=list(merged_relationships),
            )
        rekey = derived.id_remap
        self._note_decided(suffix, derived, existing)
        self._extend_stored_summaries(
            suffix,
            derived,
            entities,
            relationships,
            existing,
            existing_rels,
            entity_id_remap,
            relationship_remap,
        )
        merged_entities, merged_relationships = self._finish(
            suffix,
            derived.entities,
            derived.relationships,
            # The delta's summaries (graph resolution made them) are reused
            # as the stored ones are: by the description they were made from.
            stored_entities=[*existing, *entities],
            stored_relationships=[*existing_rels, *relationships],
            rekey=rekey,
            # The noted entities' stored edges are all among the edges given
            # (read with them): their stored links do not count either.
            rewired=set(derived.rewired) | noted,
            dropped=derived.dropped,
            # A commit killed between its vector and graph edge writes left
            # the vector copies of a split part's edges ahead of the graph's.
            split_from=derived.split_from,
        )
        # A delta id the merge folded into a stored item of another id has no
        # graph copy (the stored one of that id would have been merged), but
        # a run that died between the stores may have left its vector copy:
        # it goes with the retired ids, unless the derivation writes it (an
        # entity split off under it).
        written = {e.id for e in merged_entities} | {r.id for r in merged_relationships}
        folded = [
            delta_id
            for delta_id, kept in {**entity_id_remap, **relationship_remap}.items()
            if delta_id != kept and delta_id not in written
        ]
        self._retire(
            suffix,
            rekey,
            derived.split_from,
            [i for i in derived.dropped if i not in written]
            + folded
            + sorted(vector_orphans - written),
        )
        m._derived_entities.setdefault(suffix, {}).update(
            {e.id: e for e in merged_entities}
        )
        # A delta id that merged into a stored one follows that one's re-key;
        # the stored ids map to their new ids for what else references them.
        return (
            merged_entities,
            merged_relationships,
            {
                **rekey,
                **{
                    delta_id: rekey.get(kept, kept)
                    for delta_id, kept in {
                        **entity_id_remap,
                        **relationship_remap,
                    }.items()
                },
            },
        )

    def _merge_entities(
        self, suffix: str, entities: list[Entity]
    ) -> tuple[list[Entity], list[Entity], dict[str, str], set[str]]:
        """The delta ``entities`` merged with the stored entities they
        match (by id or name), the stored ones read, ``{delta id: stored
        id}`` and the entities a commit note names (read with them)."""
        m = self._m
        merged_entities = entities
        existing: list[Entity] = []
        entity_id_remap: dict[str, str] = {}
        # What an earlier commit of this suffix split, re-keyed or rewired
        # and may not have written in full (see note_pending_commit): the
        # stored entities and their edges are derived again with the delta,
        # and the claims about them follow.
        noted = self._read_commit_note(suffix)
        if entities:
            # The full build also merges names that differ only by legal form
            # or spacing (and by similarity with fuzzy_entity_resolution);
            # their stored entities have other ids than the delta's, so they
            # are found through the stored names and read back as well.
            name_matcher, matched_ids = self._match_stored_names(suffix, entities)
            existing = m._with_full_descriptions(
                m.graph_indexer.read_entities(
                    sorted({e.id for e in entities} | matched_ids | noted),
                    suffix=suffix,
                ),
                m.vector_indexer.read_entity_descriptions,
                suffix,
            )
            if existing:
                merged_entities, entity_id_remap = merge_entities(
                    existing, entities, fuzzy_matcher=name_matcher
                )
        elif noted:
            existing = m._with_full_descriptions(
                m.graph_indexer.read_entities(sorted(noted), suffix=suffix),
                m.vector_indexer.read_entity_descriptions,
                suffix,
            )
            merged_entities = list(existing)
        return merged_entities, existing, entity_id_remap, noted

    def _merge_relationships(
        self,
        suffix: str,
        relationships: list[Relationship],
        merged_entities: list[Entity],
        entity_id_remap: dict[str, str],
    ) -> tuple[list[Relationship], list[Relationship], dict[str, str]]:
        """The delta ``relationships`` merged with the stored edges they
        match (by id, or by endpoints once ``entity_id_remap`` re-points
        them), the stored ones read and ``{delta id: stored id}``."""
        m = self._m
        merged_relationships = relationships
        existing_rels: list[Relationship] = []
        relationship_remap: dict[str, str] = {}
        if relationships:
            rel_ids = {r.id for r in relationships}
            if merged_entities:
                # A delta edge can match a stored edge of other id between the
                # same entities (extracted with other spellings, or with an
                # endpoint that merged into a stored entity): the full build
                # merges them, so the entities' stored edges are read too.
                rel_ids |= set(
                    m.graph_indexer.find_incident_relationship_ids(
                        sorted({e.id for e in merged_entities}), suffix=suffix
                    )
                )
            delta_ids = {r.id for r in relationships}
            pairs = {
                (
                    entity_id_remap.get(r.source_id, r.source_id),
                    entity_id_remap.get(r.target_id, r.target_id),
                )
                for r in relationships
            }
            existing_rels = m._with_full_descriptions(
                [
                    r
                    for r in m.graph_indexer.read_relationships(
                        sorted(rel_ids), suffix=suffix
                    )
                    if r.id in delta_ids or (r.source_id, r.target_id) in pairs
                ],
                m.vector_indexer.read_relationship_descriptions,
                suffix,
            )
            # Even when no existing relationships are read back, delta endpoints
            # must follow any entity-id collapse from the entity merge above —
            # otherwise a remapped entity leaves its relationships pointing at a
            # now-nonexistent id. Pass the remap through unconditionally.
            if existing_rels or entity_id_remap:
                merged_relationships = merge_relationships(
                    existing_rels,
                    relationships,
                    entity_id_remap=entity_id_remap or None,
                )
                relationship_remap = relationship_id_remap(
                    relationships, merged_relationships, entity_id_remap
                )
                # The incident edges were read only to find matches: write
                # back just the edges a delta edge became or merged into.
                written = {relationship_remap.get(r.id, r.id) for r in relationships}
                merged_relationships = [
                    r for r in merged_relationships if r.id in written
                ]
        return merged_relationships, existing_rels, relationship_remap

    def _with_noted_edges(
        self,
        suffix: str,
        noted: set[str],
        existing: list[Entity],
        existing_rels: list[Relationship],
        merged_entities: list[Entity],
        merged_relationships: list[Relationship],
    ) -> tuple[
        list[Entity], list[Relationship], list[Entity], list[Relationship], set[str]
    ]:
        """The stored and merged items with the stored edges of the
        ``noted`` entities and their other endpoints (``existing``,
        ``existing_rels``, ``merged_entities``, ``merged_relationships``,
        in that order), and the vector copies of the noted entities and
        their edges the graph does not hold."""
        m = self._m
        vector_orphans: set[str] = set()
        # The noted entities' stored edges: a killed commit may not have
        # re-pointed them.
        held = {r.id for r in merged_relationships or []}
        graph_incident = set(
            m.graph_indexer.find_incident_relationship_ids(sorted(noted), suffix=suffix)
        )
        if _reads_back(m.graph_indexer):
            in_graph = {
                e.id
                for e in m.graph_indexer.read_entities(sorted(noted), suffix=suffix)
            }
            vector_entities = set(
                m.vector_indexer.read_vector_copies(
                    "entities", sorted(noted - in_graph), suffix=suffix
                )
                if noted - in_graph
                else {}
            )
            if vector_entities:
                m._vector_only.setdefault(suffix, set()).update(vector_entities)
            vector_orphans = vector_entities | (
                set(m._vector_incident_relationship_ids(sorted(noted), suffix))
                - graph_incident
                - held
            )
        noted_edges = m.graph_indexer.read_relationships(
            sorted(graph_incident - held),
            suffix=suffix,
        )
        if noted_edges:
            noted_edges = m._with_full_descriptions(
                noted_edges,
                m.vector_indexer.read_relationship_descriptions,
                suffix,
            )
            existing_rels = [*existing_rels, *noted_edges]
            # Their other endpoints too: a re-pointed edge changes their
            # rank. (Stored edges join stored ids, which the entity merge
            # keeps: no delta id remaps one.)
            known = {e.id for e in merged_entities or []}
            ends = {i for r in noted_edges for i in (r.source_id, r.target_id)} - known
            if ends:
                others = m._with_full_descriptions(
                    m.graph_indexer.read_entities(sorted(ends), suffix=suffix),
                    m.vector_indexer.read_entity_descriptions,
                    suffix,
                )
                existing = [*existing, *others]
                merged_entities = [*(merged_entities or []), *others]
            merged_relationships = [*(merged_relationships or []), *noted_edges]
        return (
            existing,
            existing_rels,
            merged_entities,
            merged_relationships,
            vector_orphans,
        )

    def _note_decided(
        self, suffix: str, derived: Derived, existing: Sequence[Entity]
    ) -> None:
        """Record the stored entities a commit's derivation renamed,
        re-keyed, split or rewired, for its write-ahead note (see
        :meth:`IndexingManager.note_pending_commit`)."""
        m = self._m
        rekey = derived.id_remap
        renamed = self._note_renamed(suffix, derived.entities, existing)
        derived_ids = {e.id for e in derived.entities}
        decided = {
            # A rename too: a run killed between the stores leaves the
            # vector copies of its edges and claims ahead of the graph's.
            *renamed,
            *(i for pair in rekey.items() if pair[1] in derived_ids for i in pair),
            *(i for pair in derived.split_from.items() for i in pair),
            *derived.rewired,
        }
        if decided:
            m._commit_entities.setdefault(suffix, set()).update(decided)

    def _extend_stored_summaries(
        self,
        suffix: str,
        derived: Derived,
        entities: Sequence[Entity],
        relationships: Sequence[Relationship],
        existing: Sequence[Entity],
        existing_rels: Sequence[Relationship],
        entity_id_remap: Mapping[str, str],
        relationship_remap: Mapping[str, str],
    ) -> None:
        """Reuse and extend the summaries of the derived items from the
        delta's and the stored ones (``existing``, ``existing_rels``)."""
        m = self._m
        rekey = derived.id_remap

        def final_id(delta_id: str, merged_into: Mapping[str, str]) -> str:
            """The id a delta item ends under: the stored one it merged into
            (another spelling, say), as the derivation re-keyed it."""
            kept = merged_into.get(delta_id, delta_id)
            return rekey.get(kept, kept)

        # A stored summary that only grew is summarized with what the delta
        # added (see extend_summaries), and so is a stored description the
        # delta added a summary to. Not one this run's removal summarized:
        # that summary was made from what the strip left, and extending it
        # would summarize a summary where a full build summarizes the lines.
        reuse_summaries(derived.entities, entities)
        reuse_summaries(derived.relationships, relationships)
        stripped = m._stripped_ids.get(suffix, set())
        # Extending holds only when the item's stored summary covers exactly
        # what its stored copy stated. Not for an entity a split parted (the
        # whole's summary holds the part's lines, the part's is the whole's),
        # nor for an id more than one delta item or stored copy ends under
        # (folded spellings: one summary would be extended, the others'
        # lines lost), nor for one whose derived item cites a unit neither
        # its stored copy nor the delta item does (a stored spelling the
        # derivation folded into it: its lines are in neither). Nor for an
        # edge with an endpoint a split parted (the delta edge's lines can
        # belong to the edge the split moved them to).
        # Those are summarized from their derived lines, as a full build does.
        parted = {i for pair in derived.split_from.items() for i in pair}
        stored_entities = [
            e.model_copy(update={"id": rekey.get(e.id, e.id)})
            for e in existing
            if e.id not in stripped
        ]
        stored_relationships = [
            r.model_copy(update={"id": rekey.get(r.id, r.id)})
            for r in existing_rels
            if r.id not in stripped
        ]
        extend_summaries(
            derived.entities,
            stored_entities,
            _single_additions(
                [(final_id(e.id, entity_id_remap), e) for e in entities],
                stored_entities,
                parted,
                derived.entities,
            ),
            summarized={
                final_id(e.id, entity_id_remap) for e in entities if _is_summary(e)
            },
        )
        extend_summaries(
            derived.relationships,
            stored_relationships,
            _single_additions(
                [(final_id(r.id, relationship_remap), r) for r in relationships],
                stored_relationships,
                {
                    final_id(r.id, relationship_remap)
                    for r in relationships
                    if {r.source_id, r.target_id} & parted
                }
                | {
                    r.id
                    for r in derived.relationships
                    if {r.source_id, r.target_id} & parted
                },
                derived.relationships,
            ),
            summarized={
                final_id(r.id, relationship_remap)
                for r in relationships
                if _is_summary(r)
            },
        )

    def derive(
        self,
        suffix: str,
        entities: Sequence[Entity],
        relationships: Sequence[Relationship],
        claims: Sequence[Claim],
        text_units: Sequence[TextUnit] = (),
        rekeyed: Mapping[str, str] | None = None,
        split_from: Mapping[str, str] | None = None,
    ) -> Derived:
        """See :meth:`IndexingManager.derive`."""
        m = self._m
        items = (list(entities), list(relationships), list(claims))
        blocked: set[str] = set()
        # The records read so far: a record does not change during the
        # derivation, so each round reads only the units it adds.
        records: dict[str, TextUnitExtraction] = {}
        claim_records: dict[str, list[Claim]] = {}
        read_units: set[str] = set()
        # Nor do the stored edges (read once per call site, _IncidentEdges).
        incident = _IncidentEdges(
            lambda ids: m.graph_indexer.find_incident_relationship_ids(
                ids, suffix=suffix
            )
        )

        # A new id another stored item holds: that item is a copy an
        # interrupted re-key left, or another spelling a full build merges
        # with this one, so it is derived with them (and merged when it is
        # the same item). One still held after that keeps its holder: the
        # item keeps its id rather than overwrite another. The stored edges
        # of an entity that was re-keyed or split are derived with it
        # (``Derived.rewired``): they may become one with a derived edge, or
        # belong to a part it split into. Until nothing more is held or
        # parted: a re-key can cascade (another spelling joins, its edges
        # move, their other ends join), and a round less leaves a moved
        # entity ranked without the edges not yet read. The last of the
        # _DERIVE_ROUNDS rounds returns whatever it reached.
        attempt = 0
        while True:
            unit_ids = sorted(
                {
                    u
                    for group in items
                    for item in group
                    for u in item.text_unit_ids or []
                }
            )
            if new_units := [u for u in unit_ids if u not in read_units]:
                read, read_claims = self._unit_records(suffix, new_units, text_units)
                records.update(read)
                claim_records.update(read_claims)
                read_units.update(new_units)
            stored_edges = (
                None
                if claims or not items[0]
                else incident(
                    "given",
                    _with_old_ids([e.id for e in items[0]], rekeyed or {}),
                )
            )
            derived = derive(
                m.config,
                *items,
                records,
                claim_records,
                rekeyed=rekeyed,
                blocked=blocked,
                split_from=split_from,
                stored_relationship_ids=stored_edges,
            )
            given = {item.id for group in items for item in group}
            sibling_ids, held = self._held_by_others(
                suffix, derived, given, blocked, claims
            )
            spellings, moved, parted = self._parted_edges(
                suffix, derived, items, claims, given, incident
            )
            rewired = sorted(moved)
            done = not any(held) and not parted
            if not done and attempt >= _DERIVE_ROUNDS - 1:
                # Out of rounds: an entity some of whose stored edges were
                # not read keeps its stored links in its rank (rewired, they
                # would not count).
                unread = {i for r in parted for i in (r.source_id, r.target_id)}
                rewired = [i for i in rewired if i not in unread]
                logger.warning(
                    "Derivation of %d items still reached more stored items "
                    "after %d rounds; %d entities keep their stored links",
                    len(items[0]) + len(items[1]) + len(items[2]),
                    _DERIVE_ROUNDS,
                    len(moved & unread),
                )
                done = True
            if done:
                m.underived_items += len(set(derived.underived))
                return derived.model_copy(
                    update={
                        "rewired": rewired,
                        "split_from": {**spellings, **derived.split_from},
                    }
                )
            items = self._next_round_items(
                suffix, items, held, parted, derived, sibling_ids, blocked, attempt
            )
            attempt += 1

    def _held_by_others(
        self,
        suffix: str,
        derived: Derived,
        given: set[str],
        blocked: set[str],
        claims: Sequence[Claim],
    ) -> tuple[set[str], tuple[list[Entity], list[Relationship], list[Claim]]]:
        """The stored entities of a derived name's key the derivation was
        not given, and the stored items another item of a round of
        :meth:`derive` holds the new id of."""
        m = self._m
        # Stored entities of a derived name's key (another spelling a
        # full build merges with it, that cites none of these units), and
        # stored claims of a derived claim's id, or of an id an earlier
        # reading of its dates gave it (``Derived.earlier_claim_ids``),
        # are held too.
        siblings = (
            m.graph_indexer.find_entity_ids_by_name_key(
                sorted({compact_name_key(e.name) for e in derived.entities}),
                suffix,
            )
            if derived.entities and not claims
            else None
        )
        sibling_ids = set(siblings or ()) - given - blocked
        targets = sorted(
            (
                set(derived.id_remap.values())
                | set(derived.split_from)
                | sibling_ids
                | {c.id for c in derived.claims}
                | set(derived.earlier_claim_ids)
            )
            - given
            - blocked
        )
        held = self._read_held(suffix, targets) if targets else ([], [], [])
        return sibling_ids, held

    def _parted_edges(
        self,
        suffix: str,
        derived: Derived,
        items: tuple[list[Entity], list[Relationship], list[Claim]],
        claims: Sequence[Claim],
        given: set[str],
        incident: _IncidentEdges,
    ) -> tuple[dict[str, str], set[str], list[Relationship]]:
        """Of a round of :meth:`derive`: the leftover spellings
        (``{part: whole}``), the stored entities whose edges moved, and
        their stored edges the derivation was not given."""
        m = self._m
        # Another spelling of a derived entity: a run that split them and
        # died before their edges and claims were written leaves those on
        # the other spelling, and only its entity copies record the split
        # (the two need not share a text unit). Such a leftover cites a
        # unit of the spelling it belongs to: only a spelling holding one
        # is rederived (all its edges, and its claims follow through
        # ``split_from``). A spelling a full build keeps apart holds none,
        # and is left as stored.
        spellings = (
            {
                part: whole
                for part, whole in _other_spellings(derived.entities, items[0]).items()
                if whole not in derived.id_remap
            }
            if not claims
            else {}
        )
        moved = set(derived.rewired)
        # And the stored endpoints a given edge was re-pointed away from
        # (onto a part an earlier, interrupted strip split off): their
        # stored links still count that edge's old copy.
        now = {r.id: {r.source_id, r.target_id} for r in derived.relationships}
        for rel in items[1]:
            if ends := now.get(derived.id_remap.get(rel.id, rel.id)):
                moved |= {rel.source_id, rel.target_id} - ends
        candidates = sorted({*moved, *spellings.values()})
        parted: list[Relationship] = []
        if candidates and not claims:
            incident_ids = incident("moved", candidates)
            if incident_ids - given:
                parted = m.graph_indexer.read_relationships(
                    sorted(incident_ids - given), suffix=suffix
                )
        if spellings:
            spellings = self._leftover_spellings(
                suffix, derived.entities, spellings, parted
            )
            moved |= set(spellings.values())
            parted = [r for r in parted if {r.source_id, r.target_id} & moved]
        return spellings, moved, parted

    def _next_round_items(
        self,
        suffix: str,
        items: tuple[list[Entity], list[Relationship], list[Claim]],
        held: tuple[list[Entity], list[Relationship], list[Claim]],
        parted: list[Relationship],
        derived: Derived,
        sibling_ids: set[str],
        blocked: set[str],
        attempt: int,
    ) -> tuple[list[Entity], list[Relationship], list[Claim]]:
        """The items the next round of :meth:`derive` derives: ``items``
        with the ``held`` items that join them (the others are added to
        ``blocked``), the ``parted`` edges and their other endpoints."""
        m = self._m
        if attempt == 0:
            items = (
                [*items[0], *held[0]],
                [*items[1], *held[1]],
                [*items[2], *held[2]],
            )
        else:
            # Another spelling of a name that joined since (an endpoint
            # of a rewired edge, say) is derived with it, as on the first
            # attempt: derived without its other statements, a spelling
            # the full build keeps apart (another type) merges in. Only
            # the other held items keep their holders.
            # So is the copy an interrupted re-key left at a new id this
            # attempt maps to (found only once a late sibling joined).
            joining = sibling_ids | set(derived.id_remap.values())
            items[0].extend(e for e in held[0] if e.id in joining)
            # And a claim stored under an earlier reading of its dates:
            # it is the derived claim (found once its statements were
            # read, a unit another held item cites).
            earlier = set(derived.earlier_claim_ids)
            items[2].extend(c for c in held[2] if c.id in earlier)
            joining |= earlier
            # Edges too: a re-key killed after writing an edge's copy at
            # its new id, before retiring the old one, left that copy.
            items[1].extend(r for r in held[1] if r.id in joining)
            blocked.update(
                item.id for group in held for item in group if item.id not in joining
            )
        # Not twice: after a split killed before its edges were written,
        # a leftover edge is both held (merged in above) and parted.
        known_edges = {r.id for r in items[1]}
        items[1].extend(r for r in parted if r.id not in known_edges)
        # Their other endpoints too: a re-pointed edge changes their rank.
        ends = {i for r in parted for i in (r.source_id, r.target_id)}
        known = {e.id for e in items[0]}
        if ends - known:
            items[0].extend(
                m.graph_indexer.read_entities(sorted(ends - known), suffix=suffix)
            )
        return items

    def _leftover_spellings(
        self,
        suffix: str,
        derived: Sequence[Entity],
        spellings: Mapping[str, str],
        edges: Sequence[Relationship],
    ) -> dict[str, str]:
        """The ``{part: whole}`` of ``spellings`` whose whole holds an edge
        among ``edges``, or a stored claim, that cites a unit of a derived
        spelling of its name (the part's or another's: a whole may hold a
        third spelling's statement): what a split a killed run left
        half-written leaves behind."""
        m = self._m
        units_of = {e.id: set(e.text_unit_ids or []) for e in derived}
        key_of = {e.id: compact_name_key(e.name) for e in derived}
        parts_by_key: dict[str, list[str]] = defaultdict(list)
        for part in spellings:
            parts_by_key[key_of.get(part, "")].append(part)

        def stated(part: str, whole: str) -> set[str]:
            # The units of every part of its name but the whole itself (a
            # spelling a full build keeps apart is a part too: its own
            # edges and claims cite its own units).
            return set().union(
                *(
                    units_of.get(other, set())
                    for other in parts_by_key[key_of.get(part, "")]
                    if other != whole
                )
            )

        left: dict[str, str] = {}
        for part, whole in spellings.items():
            units = stated(part, whole)
            if any(
                whole in (r.source_id, r.target_id)
                and units.intersection(r.text_unit_ids or [])
                for r in edges
            ):
                left[part] = whole
        pending = {p: w for p, w in spellings.items() if p not in left}
        if pending:
            prefix = m.config.indexing.opensearch.claims_index_prefix
            for part, whole in pending.items():
                part_units = sorted(stated(part, whole))
                citing = (
                    m.vector_indexer.find_documents(
                        prefix, "text_unit_ids", part_units, suffix
                    )
                    or {}
                    if part_units
                    else {}
                )
                if citing and set(citing) & set(m._claims_about(suffix, [whole])):
                    left[part] = whole
        return left

    def _read_held(
        self, suffix: str, ids: list[str]
    ) -> tuple[list[Entity], list[Relationship], list[Claim]]:
        """The stored entities, relationships and claims of ``ids``."""
        m = self._m
        return (
            m.graph_indexer.read_entities(ids, suffix=suffix),
            m.graph_indexer.read_relationships(ids, suffix=suffix),
            m._read_stored_claims(ids, suffix),
        )

    def _unit_records(
        self,
        suffix: str,
        unit_ids: Sequence[str],
        text_units: Sequence[TextUnit],
    ) -> tuple[dict[str, TextUnitExtraction], dict[str, list[Claim]]]:
        """The graph and claim records of ``unit_ids`` (those found)."""
        m = self._m
        return unit_records(m.vector_indexer, suffix, unit_ids, text_units)

    def _finish(
        self,
        suffix: str,
        entities: list[Entity],
        relationships: list[Relationship],
        *,
        stored_entities: Sequence[Entity] = (),
        stored_relationships: Sequence[Relationship] = (),
        rekey: Mapping[str, str] | None = None,
        rewired: Collection[str] = (),
        dropped: Collection[str] = (),
        split_from: Mapping[str, str] | None = None,
        unsummarized: Collection[str] = (),
        made: set[str] | None = None,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Summarize, rank and complete derived items for writing.

        A description over the summarization threshold is summarized again
        only when it differs from the one the stored summary was made from,
        and not for the ``unsummarized`` ids (the commit summarizes them, see
        :meth:`expect_delta`); the ids of the items it summarizes are added
        to ``made``.
        Ranks are recomputed over the stored graph. The stored edges of the
        entities whose names or ranks now differ (an entity renamed, or its
        degree or a neighbour's changed) are returned with the items, with
        the entities' names and their recomputed rank; ``rekey`` (stored id
        -> derived id, see ``Derived.id_remap``) re-points the stored edges
        of a re-keyed entity at its new id. The stored edges of the
        ``rewired`` entities (``Derived.rewired``) are all among
        ``relationships``: their stored links do not count. The stored edges
        at them and at a split's parts (``split_from``, see
        ``Derived.split_from``; the entities they split from are among
        ``rewired``) are compared with their vector copies too.
        """
        m = self._m
        rekey = rekey or {}
        self._summarize(
            suffix,
            entities,
            relationships,
            stored_entities,
            stored_relationships,
            unsummarized,
            made,
        )
        dropped_links = self._dropped_links(
            suffix, relationships, stored_relationships, rekey, dropped
        )
        entities, relationships = self._rank_with_stored_graph(
            suffix, entities, relationships, rekey, rewired, dropped_links
        )
        # The edges the derivation dropped go with the retired ids: they are
        # not rewritten as other stored edges of the touched entities.
        written = {r.id for r in relationships} | set(rekey) | set(dropped)
        extra_ids = sorted(
            set(
                m.graph_indexer.find_incident_relationship_ids(
                    _with_old_ids([e.id for e in entities], rekey), suffix=suffix
                )
            )
            - written
        )
        if not extra_ids:
            return entities, relationships
        extra = self._stored_edges_to_rewrite(
            suffix,
            entities,
            relationships,
            extra_ids,
            stored_entities=stored_entities,
            rekey=rekey,
            rewired=rewired,
            split_from=split_from,
            dropped_links=dropped_links,
        )
        if not extra:
            return entities, relationships
        changed = {i for r in extra for i in (r.source_id, r.target_id)}
        logger.info(
            "Rewriting the name or rank of %d stored relationships of %d "
            "entities in '%s'",
            len(extra),
            len(changed & {e.id for e in entities}),
            suffix,
        )
        return entities, [*relationships, *extra]

    def _summarize(
        self,
        suffix: str,
        entities: list[Entity],
        relationships: list[Relationship],
        stored_entities: Sequence[Entity],
        stored_relationships: Sequence[Relationship],
        unsummarized: Collection[str],
        made: set[str] | None,
    ) -> None:
        """Summarize the derived items in place, as :meth:`_finish` says."""
        m = self._m
        # A summary (reused, or already held) is not summarized again, even
        # when it is over the threshold itself (summarized in place).
        reused = reuse_summaries(
            entities, [*stored_entities, *m._offered_summaries.get(suffix, ())]
        )
        later = set(unsummarized)
        due = [e for e in entities if e.id not in reused and e.id not in later]
        derived_descriptions = {e.id: e.description for e in due}
        if any(needs_summary(m.config, i.description) for i in due):
            m.description_summarizer.summarize_entities(due)
        reused = reuse_summaries(relationships, stored_relationships)
        due_rels = [
            r for r in relationships if r.id not in reused and r.id not in later
        ]
        derived_descriptions.update({r.id: r.description for r in due_rels})
        if any(needs_summary(m.config, i.description) for i in due_rels):
            m.description_summarizer.summarize_relationships(due_rels)
        if made is not None:
            summarized: list[Entity | Relationship] = [*due, *due_rels]
            made.update(
                i.id for i in summarized if i.description != derived_descriptions[i.id]
            )

    def _dropped_links(
        self,
        suffix: str,
        relationships: Sequence[Relationship],
        stored_relationships: Sequence[Relationship],
        rekey: Mapping[str, str],
        dropped: Collection[str],
    ) -> set[frozenset[str]]:
        """Stored id pairs whose stored link does not count toward the
        ranks: the ``dropped`` edges' and the demoted ones
        (:meth:`_demoted_links`)."""
        # A dropped edge is still stored until the retired ids are deleted
        # (a resumed strip reads it back): its link does not count either.
        dropped_ids = set(dropped)
        dropped_links = {
            frozenset((r.source_id, r.target_id))
            for r in stored_relationships
            if r.id in dropped_ids
        }
        dropped_links |= self._demoted_links(
            suffix, relationships, stored_relationships, rekey, dropped_ids
        )
        return dropped_links

    def _stored_edges_to_rewrite(
        self,
        suffix: str,
        entities: list[Entity],
        relationships: list[Relationship],
        extra_ids: list[str],
        *,
        stored_entities: Sequence[Entity],
        rekey: Mapping[str, str],
        rewired: Collection[str],
        split_from: Mapping[str, str] | None,
        dropped_links: Collection[frozenset[str]],
    ) -> list[Relationship]:
        """The stored edges ``extra_ids`` of the touched ``entities`` whose
        names or ranks now differ, or whose vector copies are stale, as they
        are written (see :meth:`_finish`); ``[]`` when none."""
        m = self._m
        # Their stored copies are compared with the names and ranks the
        # touched entities now give them (not the entities with their stored
        # copies): a run that wrote the entities but died before their edges
        # leaves edges a retry still finds stale.
        stored_extra = m.graph_indexer.read_relationships(extra_ids, suffix=suffix)
        # Annotated as they will be written, so a vector copy is compared
        # with what the write would store.
        extra = m._annotated_relationships(
            r.model_copy(
                deep=True,
                update={
                    "source_id": rekey.get(r.source_id, r.source_id),
                    "target_id": rekey.get(r.target_id, r.target_id),
                },
            )
            for r in stored_extra
        )
        # A stored edge of a written edge's key (endpoints and type, as the
        # merge matches them) under another id, citing no unit the written
        # one does not, is that edge under an earlier id (the derivation
        # re-pointed the written one onto these endpoints): a full build has
        # one edge, so it goes with the retired ids.
        written_by_key = {
            relationship_key(r.source_id, r.target_id, r): r for r in relationships
        }
        duplicates = {
            r.id
            for r in extra
            if (w := written_by_key.get(relationship_key(r.source_id, r.target_id, r)))
            is not None
            and set(r.text_unit_ids or []) <= set(w.text_unit_ids or [])
        }
        # An edge between two entities the re-key merged is a self-loop a
        # full build does not draw: it goes with the retired ids.
        loops = {r.id for r in extra if r.source_id == r.target_id} | duplicates
        if loops:
            m._dropped.setdefault(suffix, set()).update(loops)
            extra = [r for r in extra if r.id not in loops]
            stored_extra = [r for r in stored_extra if r.id not in loops]
        sync_relationship_endpoint_names(extra, entities)
        endpoints = sorted({i for r in extra for i in (r.source_id, r.target_id)})
        _, ranked = with_graph_ranks(
            [],
            [*relationships, *extra],
            self._stored_neighbors(suffix, endpoints, rekey, rewired, dropped_links),
            counts=is_document_backed,
        )
        ranks = {r.id: r.rank for r in ranked}
        for rel in extra:
            rel.rank = ranks[rel.id]

        def shown(r: Relationship) -> tuple[Any, ...]:
            return r.source_id, r.target_id, r.source_name, r.target_name, r.rank

        before = {r.id: shown(r) for r in stored_extra}
        stale = self._stale_vector_copies(
            suffix, extra, entities, stored_entities, rewired, split_from
        )
        extra = [r for r in extra if before[r.id] != shown(r) or r.id in stale]
        if not extra:
            return []
        # Only the rank changed (a neighbour's degree): the vector copy is
        # updated in place, not embedded again (_upsert_vector_relationships).
        m._rank_only.update(
            r.id
            for r in extra
            if before[r.id][:4] == shown(r)[:4] and r.id not in stale
        )
        extra = m._with_full_descriptions(
            extra, m.vector_indexer.read_relationship_descriptions, suffix
        )
        return extra

    def _stale_vector_copies(
        self,
        suffix: str,
        extra: Sequence[Relationship],
        entities: Sequence[Entity],
        stored_entities: Sequence[Entity],
        rewired: Collection[str],
        split_from: Mapping[str, str] | None,
    ) -> set[str]:
        """The ids of the stored edges ``extra`` whose vector copies are
        stale (or missing, where an edge was re-pointed)."""
        m = self._m
        # And with their vector copies, where an endpoint's name or id
        # changed: a commit killed between its vector and graph edge writes
        # (or a strip before the graph edges came first) left the vector
        # copies ahead, and a derivation back to the graph copy's names must
        # still rewrite them.
        stored_names: dict[str | None, str | None] = {}
        for e in stored_entities:  # the stored copy, before a delta's of its id
            stored_names.setdefault(e.id, e.name)
        moved = {e.id for e in entities if stored_names.get(e.id) != e.name}
        # And where the derivation re-pointed edges (rewired, and the parts
        # split off; the entities they split from are rewired): a strip
        # writes the graph edges before their vector copies, so a retry after
        # a kill in between finds the graph copies current and the vector
        # copies behind, or missing (an edge under a new id, written when
        # its endpoints are re-pointed): a missing one is written.
        repointed = {*rewired, *(split_from or {})}
        moved |= repointed
        vectors = m.config.indexing.opensearch.build_relationship_vector_index
        vector_copy = (
            m.vector_indexer.read_vector_copies(
                "relationships",
                [r.id for r in extra if {r.source_id, r.target_id} & moved],
                suffix=suffix,
            )
            if moved and vectors
            else {}
        )
        stale = {
            r.id
            for r in extra
            if (
                _vector_stale(r, vector_copy[r.id])
                if r.id in vector_copy
                else vectors and bool({r.source_id, r.target_id} & repointed)
            )
        }
        return stale

    def _rank_with_stored_graph(
        self,
        suffix: str,
        entities: list[Entity],
        relationships: list[Relationship],
        rekey: Mapping[str, str],
        rewired: Collection[str] = (),
        dropped_links: Collection[frozenset[str]] = (),
    ) -> tuple[list[Entity], list[Relationship]]:
        """Recompute the ranks (degrees) of the items the delta writes over the
        stored graph plus the delta's relationships.

        A stored entity the delta does not write keeps its rank until a full
        rebuild, also when a removed document dropped one of its edges.
        """
        ids = sorted(
            {e.id for e in entities}
            | {i for r in relationships for i in (r.source_id, r.target_id)}
        )
        return with_graph_ranks(
            entities,
            relationships,
            self._stored_neighbors(suffix, ids, rekey, rewired, dropped_links),
            counts=is_document_backed,
        )

    def _demoted_links(
        self,
        suffix: str,
        relationships: Sequence[Relationship],
        stored_relationships: Sequence[Relationship],
        rekey: Mapping[str, str],
        dropped: Collection[str],
    ) -> set[frozenset[str]]:
        """Stored id pairs whose stored link no longer counts toward ranks.

        A stored edge some document backed that ``relationships`` write as
        seed-only (the units a strip removed were its document evidence) is
        still stored as document-backed until it is written, so the stored
        read counts its link. The link is left out unless another stored
        document-backed edge, not rewritten here, joins the pair. Nothing is
        read unless a written edge is seed-only.
        """
        m = self._m
        seed_only_ids = {r.id for r in relationships if not is_document_backed(r)}
        if not seed_only_ids:
            return set()
        demoted = [
            r
            for r in stored_relationships
            if rekey.get(r.id, r.id) in seed_only_ids and is_document_backed(r)
        ]
        if not demoted:
            return set()
        links = {frozenset((r.source_id, r.target_id)) for r in demoted}
        written = {r.id for r in relationships}

        def left(item_id: str) -> bool:
            # A stored edge neither rewritten (the written copy decides) nor
            # dropped.
            return rekey.get(item_id, item_id) not in written and item_id not in dropped

        ends = sorted({i for link in links for i in link})
        others = [
            i
            for i in m.graph_indexer.find_incident_relationship_ids(ends, suffix=suffix)
            if left(i)
        ]
        backed = {
            frozenset((r.source_id, r.target_id))
            for r in (
                m.graph_indexer.read_relationships(sorted(others), suffix=suffix)
                if others
                else []
            )
            if left(r.id) and is_document_backed(r)
        }
        return links - backed

    def _stored_neighbors(
        self,
        suffix: str,
        ids: Sequence[str],
        rekey: Mapping[str, str],
        rewired: Collection[str] = (),
        dropped_links: Collection[frozenset[str]] = (),
    ) -> dict[str, list[str]]:
        """``read_neighbor_ids`` of ``ids`` over the document-backed edges
        (as ranks count them), some of which may be re-keyed
        (``rekey``: stored id -> new id), stated in the new ids, without the
        links of the ``rewired`` stored entities (the caller holds all their
        edges) or the ``dropped_links`` (stored id pairs of dropped edges)."""
        m = self._m
        stored = m.graph_indexer.read_neighbor_ids(
            _with_old_ids(ids, rekey), suffix=suffix, document_backed=True
        )
        merged: dict[str, set[str]] = defaultdict(set)
        for i, neighbors in stored.items():
            if i in rewired:
                continue
            merged[rekey.get(i, i)].update(
                rekey.get(n, n)
                for n in neighbors
                if n not in rewired and frozenset((i, n)) not in dropped_links
            )
        return {i: sorted(neighbors - {i}) for i, neighbors in merged.items()}

    def _retire(
        self,
        suffix: str,
        rekey: Mapping[str, str],
        split_from: Mapping[str, str] | None = None,
        dropped: Iterable[str] = (),
    ) -> None:
        """Mark the stored ids ``rekey`` replaced, and the ``dropped`` edges,
        for deletion once the items are written under their new ids
        (:meth:`delete_retired_ids`), and note the entities split off stored
        ones (their claims follow)."""
        m = self._m
        if rekey:
            m._rekeyed.setdefault(suffix, {}).update(rekey)
        if split_from:
            m._split_from.setdefault(suffix, {}).update(split_from)
        if dropped:
            m._dropped.setdefault(suffix, set()).update(dropped)

    def _rederived_entity_ids(self, suffix: str) -> list[str]:
        """The stored entity ids whose claims this run derives again: the
        re-keyed ones, those an entity was split off and the parts split off
        them (a stored claim may name a part already), those renamed, and the
        other spellings this run derived of a split-off, re-keyed or renamed
        entity's name (a claim naming it may sit on one of them)."""
        m = self._m
        split_from = m._split_from.get(suffix, {})
        derived = m._derived_entities.get(suffix, {})
        changed = {
            *split_from,
            *m._rekeyed.get(suffix, {}).values(),
            *m._renamed.get(suffix, set()),
        }
        keys = {compact_name_key(derived[i].name) for i in changed if i in derived}
        return sorted(
            {
                *m._rekeyed.get(suffix, {}),
                *split_from.values(),
                *split_from,
                *(i for i, e in derived.items() if compact_name_key(e.name) in keys),
                *m._renamed.get(suffix, set()),
                *m._noted_commit_entities.get(suffix, set()),
            }
        )

    def _note_renamed(
        self, suffix: str, derived: Sequence[Entity], stored: Sequence[Entity]
    ) -> list[str]:
        """Record (and return) the stored entities ``derived`` renamed under
        their own id."""
        m = self._m
        names = {e.id: e.name for e in stored}
        renamed = [e.id for e in derived if e.id in names and names[e.id] != e.name]
        if renamed:
            m._renamed.setdefault(suffix, set()).update(renamed)
        return renamed

    def _read_commit_note(self, suffix: str) -> set[str]:
        """The entities an earlier commit's note of ``suffix`` names (see
        :meth:`note_pending_commit`), recorded so their claims follow. A
        failed read raises (see merge_with_existing_graph)."""
        m = self._m
        notes = m.vector_indexer.read_extraction_records(
            [_PENDING_COMMIT, _STRIPPED_SUMMARIES], suffix=suffix
        )
        note = notes.get(_PENDING_COMMIT)
        # The items an earlier run's removal stripped and whose commit did
        # not run: their stored summaries are not extended either.
        m.read_stripped_summaries(suffix, notes)
        ids = _note_ids(note, "entity_ids")
        if note is not None:
            m._commit_note_suffixes.add(suffix)
        if ids:
            m._noted_commit_entities.setdefault(suffix, set()).update(ids)
        return ids

    def _match_stored_names(
        self, suffix: str, entities: list[Entity]
    ) -> tuple[FuzzyMatcher | None, set[str]]:
        """Match delta entity names against the stored names of ``suffix``.

        Returns a matcher over the stored names (built with the full build's
        settings) and the ids of the stored entities some delta name matches.
        Returns ``(None, set())`` when the adapter projects no names, which
        leaves the merge to ids and exact keys.

        Without ``fuzzy_entity_resolution`` names match by their
        ``compact_name_key`` only, so just the stored entities with a delta
        name's key are read (``GraphIndexer.find_entity_ids_by_name_key``);
        fuzzy matching, or a graph store that cannot look keys up, reads
        every stored name.
        """
        m = self._m
        processing = m.config.processing
        graph = m.graph_indexer
        keyed = (
            None
            if processing.fuzzy_entity_resolution
            else graph.find_entity_ids_by_name_key(
                sorted({compact_name_key(e.name) for e in entities}), suffix
            )
        )
        stored_names = (
            graph.read_entity_names(suffix)
            if keyed is None
            else [(e.id, e.name) for e in graph.read_entities(keyed, suffix=suffix)]
        )
        id_by_name: dict[str, str] = {}
        for eid, name in stored_names:
            id_by_name.setdefault(name, eid)
        if not id_by_name:
            return None, set()

        matcher = FuzzyMatcher(
            candidates=list(id_by_name.keys()),
            resolution_method=processing.resolution_method,
            similarity_threshold=processing.similarity_threshold,
            fuzzy=processing.fuzzy_entity_resolution,
        )
        old_ids = {
            id_by_name[name]
            for entity in entities
            for name, _score in matcher.find_all_matches(entity.name)
            if name in id_by_name
        }
        return matcher, old_ids

    def _derive_claims(
        self,
        suffix: str,
        claims: list[Claim],
        text_units: Sequence[TextUnit] = (),
        removed: Sequence[str] = (),
    ) -> list[Claim]:
        """``claims`` with their content derived from their text units'
        records, resolved against their subject and object entities: as this
        run derived them, else as stored (stripped of ``removed`` text
        units). A claim whose derived id differs is retired (see
        :meth:`delete_retired_ids`)."""
        m = self._m
        rekeyed = m._rekeyed.get(suffix, {})
        split_from = m._split_from.get(suffix, {})
        derived_entities = m._derived_entities.get(suffix, {})
        named = {i for c in claims for i in (c.subject_id, c.object_id) if i}
        entity_ids = {rekeyed.get(i, i) for i in named} | {
            part for part, origin in split_from.items() if origin in named
        }
        entities = m.graph_indexer.read_entities(
            sorted(entity_ids - set(derived_entities)), suffix=suffix
        )
        # The claims' stored subject and object names too: an endpoint a
        # removal deleted (its spelling was stated only in removed text
        # units) is read back as nothing, and the claim's statement now
        # resolves to another spelling of its name.
        stated_keys = {
            compact_name_key(name)
            for c in claims
            for name in (c.subject_name, c.object_name)
            if name
        }
        if removed:
            # And the stored entities of the other spellings of their names:
            # a retried strip re-derives the claims it noted about a split's
            # entities without the split (it landed), and the stated
            # spelling resolves to the part only when the part is among
            # them. Only a strip reads them (a commit's merge reads its
            # names' spellings already, see merge_with_existing_graph).
            keys = (
                {compact_name_key(e.name) for e in entities}
                | {
                    compact_name_key(derived_entities[i].name)
                    for i in entity_ids
                    if i in derived_entities
                }
                | stated_keys
            )
            family = set(
                m.graph_indexer.find_entity_ids_by_name_key(sorted(keys), suffix) or []
            ) - (entity_ids | set(derived_entities))
            if family:
                entities += m.graph_indexer.read_entities(sorted(family), suffix=suffix)
            stripped = {e.id: e for e in remove_text_units(entities, [], removed)[0]}
            entities = [stripped.get(e.id, e) for e in entities]
        entities += [
            derived_entities[i] for i in sorted(entity_ids) if i in derived_entities
        ]
        # The other spellings of their names this run derived (not stored
        # yet): their text units' records decide, with the claims' own,
        # which spelling a claim's subject and object resolve to.
        keys = {compact_name_key(e.name) for e in entities} | stated_keys
        given = {e.id for e in entities}
        entities += [
            e
            for i, e in sorted(derived_entities.items())
            if i not in given and compact_name_key(e.name) in keys
        ]
        derived = m.derive(
            suffix,
            entities,
            [],
            claims,
            text_units,
            rekeyed=rekeyed,
            split_from=split_from,
        )
        # The stored claims derived with them as stored under an earlier
        # reading of their dates are retired too.
        claim_ids = {c.id for c in claims} | set(derived.earlier_claim_ids)
        self._retire(
            suffix,
            {old: new for old, new in derived.id_remap.items() if old in claim_ids},
            dropped=[i for i in derived.dropped if i in claim_ids],
        )
        return derived.claims
