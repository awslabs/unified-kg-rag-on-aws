# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers over the items the stores hold, shared by the indexing
manager and its collaborators (pure: no store is read here unless one is
passed in)."""

from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any, NamedTuple, TypeVar

from unified_kg_rag.adapters.ingestion.description_summarizer import (
    SUMMARIZED_FROM_ATTRIBUTE,
)
from unified_kg_rag.domain.ingestion.base_processor import SEED_PROPERTIES_ATTRIBUTE
from unified_kg_rag.domain.ingestion.contributions import (
    claims_from_payload,
    record_from_payload,
    record_keys,
)
from unified_kg_rag.domain.ingestion.derivation import derive
from unified_kg_rag.domain.ingestion.merge import remove_text_units
from unified_kg_rag.domain.ingestion.names import compact_name_key
from unified_kg_rag.domain.ingestion.seed_graph import annotate_sources
from unified_kg_rag.domain.models import (
    Claim,
    Config,
    Entity,
    Relationship,
    TextUnit,
    TextUnitExtraction,
)
from unified_kg_rag.ports.indexer import GraphIndexer, IndexingStats, VectorIndexer
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)

# A graph item: an entity or a relationship.
_Described = TypeVar("_Described", Entity, Relationship)


# Extraction-store key of a suffix's strip note: the stored items a
# removal's strip merges or re-keys, kept until their old ids are deleted
# (record keys are ``fingerprint:hash``: no clash).
_PENDING_STRIP = "pending-strip"


# Extraction-store key of a suffix's commit note: the stored entities an
# incremental commit splits, re-keys or rewires and the old ids it retires,
# written before its first write and deleted once the old ids are (see
# note_pending_commit).
_PENDING_COMMIT = "pending-commit"


# Extraction-store key of a suffix's write note: the items whose write a
# commit's stores rejected, written again by the next commit of the suffix
# (see pending_writes).
_PENDING_WRITES = "pending-writes"


# Extraction-store key of a suffix's summary note: the items a removal's
# strip summarized or wrote unsummarized, whose stored summary the run's
# commit must not extend; written before the strip's writes and deleted
# once a commit's writes succeed (see note_stripped_summaries).
_STRIPPED_SUMMARIES = "stripped-summaries"


def _note_ids(note: Mapping[str, Any] | None, field: str) -> set[str]:
    """The ids a strip or commit note (``None``: none stored) lists under
    ``field``."""
    return {i for i in (note or {}).get(field) or [] if isinstance(i, str)}


def _other_spellings_of_parts(
    split_from: Mapping[str, str], derived: Mapping[str, Entity]
) -> list[str]:
    """The ids of the ``derived`` entities (by id) sharing a
    ``compact_name_key`` with a part of ``split_from``, the parts excepted:
    a stored claim naming a part's spelling may be on one of them."""
    keys = {compact_name_key(derived[p].name) for p in split_from if p in derived}
    return sorted(
        i
        for i, e in derived.items()
        if i not in split_from and compact_name_key(e.name) in keys
    )


def _is_summary(item: Entity | Relationship) -> bool:
    """Whether ``item``'s description is a summary (it carries the mark of
    the description it was made from)."""
    return SUMMARIZED_FROM_ATTRIBUTE in (item.attributes or {})


def _reads_back(graph: GraphIndexer) -> bool:
    """Whether ``graph`` reads stored entities back (the port's default
    reads nothing, so every id would look absent). Asked of the bound
    method, not the type: the pipeline hands stages a borrowed view of an
    injected indexer."""
    method = getattr(graph, "read_entities", None)
    return getattr(method, "__func__", method) is not GraphIndexer.read_entities


def _vector_content(item: Entity | Relationship) -> dict[str, Any]:
    """What the vector store holds of an item besides its description,
    lineage and rank, by field (see ``read_vector_copies``); a schema
    annotation or seed-graph field is absent from a copy written without
    one (``None``), so an item without one compares equal to a copy written
    before the field existed.

    An entity's ``seed_properties`` are compared too: a lineage update
    leaves the stored attributes as they are, so a copy whose seed
    properties a removed seed (or document) stated is written in full."""
    if isinstance(item, Entity):
        return {
            "name": item.name or "",
            "type": item.type or "",
            "confidence": item.confidence if item.confidence is not None else 1.0,
            "in_schema": item.in_schema,
            "type_iri": item.type_iri,
            "iri": item.iri,
            "source_kinds": item.source_kinds or None,
            SEED_PROPERTIES_ATTRIBUTE: (item.attributes or {}).get(
                SEED_PROPERTIES_ATTRIBUTE
            )
            or None,
        }
    return {
        "source_id": item.source_id,
        "target_id": item.target_id,
        "source_name": item.source_name or "",
        "target_name": item.target_name or "",
        "in_schema": item.in_schema,
        "predicate_iri": item.predicate_iri,
        "source_kinds": item.source_kinds or None,
    }


def _annotations(item: Entity | Relationship) -> dict[str, Any]:
    """The annotations of ``item`` in :func:`_vector_content`: the schema's
    and its ``source_kinds``, which follow its lineage."""
    names = ("in_schema", "type_iri", "predicate_iri", "source_kinds")
    return {k: v for k, v in _vector_content(item).items() if k in names}


def _with_sources(item: _Described) -> _Described:
    """``item`` with its ``source_kinds`` (``annotate_sources``: ``None``
    unless it cites a seed graph's unit); itself when they already hold."""
    kinds = annotate_sources(item)
    if item.source_kinds == kinds:
        return item
    return item.model_copy(update={"source_kinds": kinds})


def _vector_stale(item: Entity | Relationship, stored: Mapping[str, Any]) -> bool:
    """Whether the stored vector copy of ``item`` differs from it in
    :func:`_vector_content`."""
    return any(stored.get(k) != v for k, v in _vector_content(item).items())


def _vector_rank_stale(item: Relationship, stored: Mapping[str, Any]) -> bool:
    """Whether the stored vector copy of ``item`` holds another rank (as
    the adapter writes it: ``1.0`` for none). Not in :func:`_vector_content`:
    a rank is set in place with the lineage, so one that differs alone is
    no reason to embed the item again."""
    return stored.get("rank") != (item.rank if item.rank is not None else 1.0)


def _claim_text(claim: Claim) -> tuple[Any, ...]:
    """What a claim states beyond its lineage: its text and the names of the
    entities it is about (a renamed entity renames them)."""
    return (
        claim.type,
        claim.status,
        claim.description,
        claim.source_text,
        claim.subject_name,
        claim.object_name,
    )


def _keeps(derived: Claim, stored: Claim) -> bool:
    """Whether ``derived`` holds every text unit and line of the ``stored``
    copy it overwrites (none moved to another id)."""

    def lines(text: str | None) -> set[str]:
        return set((text or "").split("\n")) - {""}

    return (
        set(stored.text_unit_ids or []) <= set(derived.text_unit_ids or [])
        and lines(stored.description) <= lines(derived.description)
        and lines(stored.source_text) <= lines(derived.source_text)
    )


def write_claims_new_ids_first(
    vector: VectorIndexer,
    claims: Sequence[Claim],
    stored: Mapping[str, Claim],
    lineage_only: Collection[str] = (),
) -> IndexingStats:
    """Write derived ``claims``: those under new ids first, then those
    overwriting a ``stored`` copy (by id; the ``lineage_only`` ones have
    their lineage set in place, no embedding). When a first write fails, an
    overwrite that drops a text unit or line of its stored copy is held
    back.

    A stored claim the derivation overwrites with another statement (its
    id now another's) holds its own lines until the claim they moved to,
    under a new id, is written: a commit or a strip that died (or whose
    write was rejected) after the overwrite and before the new claim lost
    them, and no later run derived them again. The stored claims held back
    count as failed, by id (a commit notes them, a strip is retried). One
    that keeps every unit and line of its stored copy (a claim restated by
    another document) loses nothing and is written: holding it back too
    failed a commit whose one new claim was rejected, as every restated
    claim counted failed."""
    stats = IndexingStats()
    new = [c for c in claims if c.id not in stored]
    rest = [c for c in claims if c.id in stored]
    if new:
        stats.merge(vector.upsert_claims(new))
    if stats.failed_items:
        held = sorted(c.id for c in rest if not _keeps(c, stored[c.id]))
        if held:
            held_stats = IndexingStats(total_items=len(held))
            held_stats.add_error(
                "not written: a claim under a new id failed", len(held), held
            )
            stats.merge(held_stats)
        rest = [c for c in rest if c.id not in set(held)]
    in_place = [c for c in rest if c.id in lineage_only]
    whole = [c for c in rest if c.id not in lineage_only]
    if in_place:
        stats.merge(vector.update_claim_lineage(in_place))
    if whole:
        stats.merge(vector.upsert_claims(whole))
    stats.total_items = len(claims)
    return stats


def _other_spellings(
    derived: Sequence[Entity], given: Sequence[Entity]
) -> dict[str, str]:
    """``{derived id: given id}`` of each given entity of another id whose
    name has a derived entity's ``compact_name_key``."""
    by_key: dict[str, list[Entity]] = defaultdict(list)
    for entity in derived:
        by_key[compact_name_key(entity.name)].append(entity)
    pairs: dict[str, str] = {}
    for other in given:
        for entity in by_key.get(compact_name_key(other.name), ()):
            if entity.id != other.id:
                pairs.setdefault(entity.id, other.id)
    return pairs


def _with_old_ids(ids: Iterable[str], rekey: Mapping[str, str]) -> list[str]:
    """``ids`` and the stored ids ``rekey`` replaced by them (an item merged
    from several stored copies has its edges under each)."""
    wanted = set(ids)
    return sorted(wanted | {old for old, new in rekey.items() if new in wanted})


class _IncidentEdges:
    """The stored edges incident to entity ids, by call site, for one
    derivation: the stored edges do not change during it, so a query over a
    superset of the ids its site asked last reads only the ids it adds. A
    query over other ids reads them all again (the edges found for ids no
    longer asked must not be returned)."""

    def __init__(self, read: Callable[[list[str]], Iterable[str]]) -> None:
        self._read = read
        self._by_site: dict[str, tuple[set[str], set[str]]] = {}

    def __call__(self, site: str, ids: Collection[str]) -> set[str]:
        asked, found = self._by_site.get(site, (set(), set()))
        wanted = set(ids)
        if not asked <= wanted:
            asked, found = set(), set()
        if new := sorted(wanted - asked):
            found = found | set(self._read(new))
        self._by_site[site] = (wanted, found)
        return found


def unit_records(
    vector: VectorIndexer,
    suffix: str,
    unit_ids: Sequence[str],
    text_units: Sequence[TextUnit] = (),
) -> tuple[dict[str, TextUnitExtraction], dict[str, list[Claim]]]:
    """The graph and claim records of ``unit_ids`` (those found): of the
    ``text_units`` given, the others' record keys read from ``vector``."""
    wanted = set(unit_ids)
    units = {u.id: u for u in text_units if u.id in wanted}
    missing = [i for i in unit_ids if i not in units]
    if missing:
        for unit_id, attributes in vector.read_text_unit_attributes(
            missing, suffix=suffix
        ).items():
            units[unit_id] = TextUnit.model_validate(
                {"id": unit_id, "text": "", "attributes": attributes}
            )
    keys = {u.id: record_keys(u) for u in units.values()}
    payloads = vector.read_extraction_records(
        sorted({key for k in keys.values() for key in k.values()}), suffix=suffix
    )
    records: dict[str, TextUnitExtraction] = {}
    claim_records: dict[str, list[Claim]] = {}
    for unit_id, unit_keys in keys.items():
        unit = units[unit_id]
        try:
            if (payload := payloads.get(unit_keys.get("graph", ""))) is not None:
                records[unit_id] = record_from_payload(payload, unit)
            if (payload := payloads.get(unit_keys.get("claims", ""))) is not None:
                claim_records[unit_id] = claims_from_payload(payload, unit)
        except (ValueError, TypeError) as e:
            logger.warning(
                "Ignoring an unreadable extraction record of text unit '%s': %s",
                unit.short_id or unit_id,
                type(e).__name__,
            )
    return records, claim_records


def stripped_descriptions(
    config: Config,
    vector: VectorIndexer,
    suffix: str,
    entities: Sequence[Entity],
    removed: Collection[str],
) -> dict[str, str]:
    """The descriptions of the stored ``entities`` once the text units
    ``removed`` are stripped from them, by id, as a removal's strip derives
    them before the cross-run merge appends a delta's lines
    (``remove_text_units``, then ``derive`` over the records of the units
    left). An entity that cites no removed unit keeps its description, as
    does one a record of whose units is missing (the strip keeps it as
    stored). Raises when a read fails."""
    descriptions = {e.id: e.description or "" for e in entities}
    stripped, _ = remove_text_units(list(entities), [], removed)
    if not stripped:
        return descriptions
    records, _ = unit_records(
        vector,
        suffix,
        sorted({u for e in stripped for u in e.text_unit_ids or []}),
    )
    derived = derive(config, stripped, [], [], records)
    by_id = {e.id: e for e in derived.entities}
    for entity in stripped:
        if entity.id in derived.underived:
            continue
        match = by_id.get(derived.id_remap.get(entity.id, entity.id))
        if match is not None:
            descriptions[entity.id] = match.description or ""
    return descriptions


class IndexingTask(NamedTuple):
    fn: Callable[..., Any]
    args: list[Any]
    key: str
    # Run even without items (a full build replacing a type with nothing).
    run_empty: bool = False
