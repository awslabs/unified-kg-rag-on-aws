# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item writes a commit's stores rejected, written again by the next commit.

The documents of a commit whose entity, relationship, claim, community or
report write failed are indexed all the same: what failed is the store's
write of an item, which the next commit can make from what the stores and
the extraction records hold, not their extraction. So the rejected items
are noted, per index suffix, in the extraction store (``pending-writes``,
the items as the commit wrote them), and the next commit of the suffix
writes them again with its own delta (:meth:`PendingWrites.read`), merged
with what is stored like any delta item. Re-extracting their documents
instead re-extracted every document a hub entity cites, on every run its
write kept failing.

A text unit's failed write is its document's: it is not noted, and the
document is recorded FAILED (see ``IncrementalIndexer.record``).
"""

from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from unified_kg_rag.application.storage.stored_items import _PENDING_WRITES
from unified_kg_rag.domain.ingestion.merge import (
    merge_claims,
    merge_entities,
    merge_relationships,
    remove_claim_text_units,
    remove_text_units,
)
from unified_kg_rag.domain.models import (
    Claim,
    Community,
    CommunityReport,
    Entity,
    Relationship,
    TextUnit,
)
from unified_kg_rag.ports.indexer import BaseIndexer, IndexingStats
from unified_kg_rag.shared import NamespaceLeaseError, get_logger

if TYPE_CHECKING:
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

logger = get_logger(__name__)

# The artifact kinds a note holds, by the model of their items.
_KINDS: dict[str, type[BaseModel]] = {
    "entities": Entity,
    "relationships": Relationship,
    "claims": Claim,
    "communities": Community,
    "community_reports": CommunityReport,
}

# The index results (``IndexingManager`` result keys) whose failed items a
# note covers, by the kind they write. Any other result (text units, a
# full build's sweep or summary nodes) is not noted.
RESULT_KINDS: dict[str, str] = {
    "vector_entities": "entities",
    "graph_entities": "entities",
    "vector_relationships": "relationships",
    "graph_relationships": "relationships",
    "vector_claims": "claims",
    "graph_communities": "communities",
    "vector_community_reports": "community_reports",
}

_Item = TypeVar("_Item", bound=BaseModel)


def _kind_of(item: Any) -> str | None:
    for kind, model in _KINDS.items():
        if type(item) is model:
            return kind
    return None


def _dump(item: BaseModel) -> dict[str, Any]:
    """``item`` as noted: without its embeddings (the write embeds again)."""
    return item.model_dump(
        mode="json", exclude={f for f in type(item).model_fields if "embedding" in f}
    )


class NotedWrites(BaseModel):
    """The items of the notes a commit read, by kind (see
    :meth:`PendingWrites.read`)."""

    entities: list[Entity] = []
    relationships: list[Relationship] = []
    claims: list[Claim] = []
    communities: list[Community] = []
    community_reports: list[CommunityReport] = []

    def __bool__(self) -> bool:
        return any(getattr(self, kind) for kind in _KINDS)


def _join(
    delta: list[_Item] | None,
    noted: Sequence[_Item],
    merge: Callable[[_Item, _Item], _Item],
) -> list[_Item] | None:
    """``delta`` with the ``noted`` items: a delta item of a noted item's id
    is ``merge(noted, delta item)``, the others are added."""
    if not noted:
        return delta
    by_id = {item.id: item for item in noted}  # type: ignore[attr-defined]
    joined = [
        item if item.id not in by_id else merge(by_id.pop(item.id), item)  # type: ignore[attr-defined]
        for item in delta or []
    ]
    return [*joined, *by_id.values()]


def _merged_entity(noted: Entity, delta: Entity) -> Entity:
    return merge_entities([noted], [delta])[0][0]


def _merged_relationship(noted: Relationship, delta: Relationship) -> Relationship:
    (merged,) = [r for r in merge_relationships([noted], [delta]) if r.id == noted.id]
    return merged


def _merged_claim(noted: Claim, delta: Claim) -> Claim:
    return merge_claims([noted], [delta])[0]


def _delta_wins(_noted: _Item, delta: _Item) -> _Item:
    # A community or report this run made again replaces the noted one.
    return delta


class PendingWrites:
    """The write notes of the suffixes a commit writes, for
    :class:`IndexingManager` (whose vector store holds them)."""

    def __init__(self, manager: "IndexingManager") -> None:
        self._m = manager
        # The suffixes whose note this run read: the commit's note step
        # deletes one it has nothing left to note in.
        self._read: set[str] = set()
        # The suffixes whose noted items the commit's delta holds: derived
        # from their records even when nothing else is stored.
        self._joined: set[str] = set()

    def reset(self) -> None:
        self._read.clear()
        self._joined.clear()

    def joined(self, suffix: str) -> bool:
        """Whether the commit's delta holds noted items of ``suffix``: their
        content is derived from their records (a noted item's lineage may
        have lost the units of a document removed since)."""
        return suffix in self._joined

    def read(
        self, suffixes: Iterable[str], text_units: Sequence[TextUnit] = ()
    ) -> NotedWrites:
        """The noted items of ``suffixes``. Each keeps only the text units
        the vector store still holds or the delta writes (``text_units``):
        a document removed or changed since the note was written takes its
        units out of them, as its removal did of the stored copies. An item
        left citing none is dropped (its removal deleted any stored copy).
        A failed read raises, before the commit writes anything."""
        noted = NotedWrites()
        delta_units = {u.id for u in text_units}
        for suffix in sorted(set(suffixes)):
            note = self._m.vector_indexer.read_extraction_records(
                [_PENDING_WRITES], suffix=suffix
            ).get(_PENDING_WRITES)
            if note is None:
                continue
            self._read.add(suffix)
            items = self._parse(note, suffix)
            cited = sorted(
                {
                    u
                    for group in items.values()
                    for item in group
                    for u in getattr(item, "text_unit_ids", None) or []
                }
                - delta_units
            )
            held = self._m.vector_indexer.read_text_unit_ids(cited, suffix=suffix)
            gone = set() if held is None else set(cited) - set(held)
            for kind, group in self._without_units(items, gone).items():
                getattr(noted, kind).extend(group)
            logger.info(
                "Writing again %d items of '%s' whose write an earlier commit's "
                "stores rejected",
                sum(len(group) for group in items.values()),
                suffix,
            )
        return noted

    @staticmethod
    def _parse(note: Mapping[str, Any], suffix: str) -> dict[str, list[Any]]:
        """The items of a stored ``note``, by kind; one that does not parse
        (written by another version) is skipped with a warning."""
        items: dict[str, list[Any]] = {}
        for kind, model in _KINDS.items():
            parsed = []
            for value in note.get(kind) or []:
                try:
                    parsed.append(model.model_validate(value))
                except ValidationError as e:
                    logger.warning(
                        "Skipping a noted %s of '%s' that does not parse: %s",
                        kind,
                        suffix,
                        e.errors()[:1],
                    )
            items[kind] = parsed
        return items

    @staticmethod
    def _without_units(
        items: dict[str, list[Any]], gone: Collection[str]
    ) -> dict[str, list[Any]]:
        """``items`` without the text units ``gone`` (weights recomputed as a
        removal does), an item that cited only those dropped."""
        if not gone:
            return items
        entities, relationships = remove_text_units(
            items["entities"], items["relationships"], gone
        )
        stripped: dict[str, dict[str, Any]] = {
            "entities": {e.id: e for e in entities},
            "relationships": {r.id: r for r in relationships},
            "claims": {c.id: c for c in remove_claim_text_units(items["claims"], gone)},
        }
        for kind in ("communities", "community_reports"):
            stripped[kind] = {
                item.id: item.model_copy(
                    update={
                        "text_unit_ids": [
                            u for u in item.text_unit_ids or [] if u not in gone
                        ]
                    }
                )
                for item in items[kind]
                if set(item.text_unit_ids or []) & set(gone)
            }
        kept: dict[str, list[Any]] = {}
        for kind, group in items.items():
            updated = [stripped[kind].get(item.id, item) for item in group]
            kept[kind] = [
                item
                for item in updated
                if item.text_unit_ids or item.id not in stripped[kind]
            ]
        return kept

    def join(
        self,
        noted: NotedWrites,
        entities: list[Entity] | None,
        relationships: list[Relationship] | None,
        claims: list[Claim] | None,
        communities: list[Community] | None,
        community_reports: list[CommunityReport] | None,
    ) -> tuple[
        list[Entity] | None,
        list[Relationship] | None,
        list[Claim] | None,
        list[Community] | None,
        list[CommunityReport] | None,
    ]:
        """The commit's delta with the ``noted`` items (see :func:`_join`).
        The noted entities and relationships are summarized from their
        records, not by extending their stored summary: the noted
        description is a summary of the lines a full build summarizes.

        The endpoints of a noted relationship are written with it, as a
        delta's are, so the ranks of their stored edges follow its degree:
        one neither the delta nor the note holds is read from the graph."""
        known = {e.id for e in [*(entities or []), *noted.entities]}
        ends: dict[str, set[str]] = defaultdict(set)
        for rel in noted.relationships:
            for end in (rel.source_id, rel.target_id):
                if end not in known:
                    ends[BaseIndexer.get_suffix(rel)].add(end)
        for suffix, ids in sorted(ends.items()):
            noted.entities.extend(
                self._m._with_full_descriptions(
                    self._m.graph_indexer.read_entities(sorted(ids), suffix=suffix),
                    self._m.vector_indexer.read_entity_descriptions,
                    suffix,
                )
            )
        for item in [*noted.entities, *noted.relationships]:
            suffix = BaseIndexer.get_suffix(item)
            self._m._stripped_ids.setdefault(suffix, set()).add(item.id)
        for group in (
            noted.entities,
            noted.relationships,
            noted.claims,
            noted.communities,
            noted.community_reports,
        ):
            self._joined.update(BaseIndexer.get_suffix(item) for item in group)
        return (
            _join(entities, noted.entities, _merged_entity),
            _join(relationships, noted.relationships, _merged_relationship),
            _join(claims, noted.claims, _merged_claim),
            _join(communities, noted.communities, _delta_wins),
            _join(community_reports, noted.community_reports, _delta_wins),
        )

    def note(
        self, results: Mapping[str, IndexingStats], written: Iterable[Any]
    ) -> set[str]:
        """Note, per suffix, the ``written`` items whose write failed (the
        ids in ``results``; every written item of the kind for a failure
        that names none), and delete the notes this run read with nothing
        left to note. Returns the result keys whose failures are noted: a
        failed id no written item of its kind has is not, and a note that
        cannot be written covers nothing (the caller then records the
        documents owning the items FAILED, as before notes)."""
        by_kind: dict[str, list[Any]] = defaultdict(list)
        for item in written:
            if (kind := _kind_of(item)) is not None:
                by_kind[kind].append(item)
        failed: dict[str, dict[str, dict[str, Any]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        covered: set[str] = set()
        for key, stats in results.items():
            kind = RESULT_KINDS.get(key)
            if kind is None or not stats or not stats.failed_items:
                continue
            items = by_kind.get(kind, [])
            if not stats.unattributed_failures:
                ids = set(stats.failed_ids)
                items = [item for item in items if item.id in ids]
                if ids - {item.id for item in items}:
                    continue
            if not items:
                continue
            for item in items:
                failed[BaseIndexer.get_suffix(item)][kind][item.id] = item
            covered.add(key)
        try:
            for suffix in sorted(failed):
                note = {
                    kind: [_dump(item) for item in group.values()]
                    for kind, group in failed[suffix].items()
                }
                self._m._write_note(_PENDING_WRITES, suffix, note)
                logger.warning(
                    "%d item writes of '%s' failed; noted for the next run, "
                    "which writes them again (their documents stay indexed)",
                    sum(len(group) for group in failed[suffix].values()),
                    suffix,
                )
        except NamespaceLeaseError:
            raise
        except Exception as e:  # noqa: BLE001 - their documents are retried
            logger.error("Noting the failed item writes failed: %s", e)
            return set()
        for suffix in sorted(self._read - set(failed)):
            self._m._delete_note(_PENDING_WRITES, suffix)
        self._read.clear()
        return covered
