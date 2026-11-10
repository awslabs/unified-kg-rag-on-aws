# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Collection, Iterator, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from unified_kg_rag.domain.models import (
    Claim,
    Community,
    CommunityReport,
    Config,
    Constants,
    Entity,
    IndexArtifact,
    IndexBuildRecord,
    Relationship,
    SummaryNode,
    TextUnit,
)
from unified_kg_rag.domain.retrieval.store_names import (
    graph_label,
    index_suffix,
    namespace_suffix,
    store_name,
)
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)

# The kinds of item ``VectorIndexer.scan_items`` pages through, one index each.
StoredItemKind = Literal[
    "text_units",
    "entities",
    "relationships",
    "claims",
    "community_reports",
    "summary_nodes",
]
STORED_ITEM_KINDS: tuple[StoredItemKind, ...] = (
    "text_units",
    "entities",
    "relationships",
    "claims",
    "community_reports",
    "summary_nodes",
)


class IndexingStats(BaseModel):
    total_items: int = Field(default=0)
    successful_items: int = Field(default=0)
    failed_items: int = Field(default=0)
    errors: list[str] = Field(default_factory=list)
    # Ids of the items whose write failed, where the failure site knows them,
    # kept with multiplicity: a failure site records exactly one id per failed
    # item, so an artifact whose write failed twice (in two batches) appears
    # twice. An incremental commit records
    # the documents owning these artifacts FAILED so the next run rewrites
    # them; see unattributed_failures for the failures no id was reported for.
    failed_ids: list[str] = Field(default_factory=list)
    processing_time: float = Field(default=0.0)

    @property
    def success_rate(self) -> float:
        return self.successful_items / self.total_items if self.total_items > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.failed_items / self.total_items if self.total_items > 0 else 0.0

    @property
    def unattributed_failures(self) -> int:
        """Failures reported without an artifact id.

        Derived from the counters rather than stored, so stats built directly
        (``IndexingStats(failed_items=N)``) still count as unattributed. Exact
        because add_error records at most one id per failed item.
        """
        return max(0, self.failed_items - len(self.failed_ids))

    def add_success(self, count: int = 1) -> None:
        self.successful_items += count

    def add_error(
        self, error_message: str, count: int = 1, ids: list[str] | None = None
    ) -> None:
        """Record ``count`` failed items, ``ids`` naming the ones known.

        ``ids`` holds one entry per failed item (an id may repeat when the same
        artifact failed more than once) and must not outnumber ``count``.
        """
        if ids and len(ids) > count:
            raise ValueError(
                f"add_error got {len(ids)} ids for {count} failed items; "
                "record one id per failed item"
            )
        self.failed_items += count
        if ids:
            self.failed_ids.extend(ids)
        if error_message not in self.errors:
            self.errors.append(error_message)

    def merge(self, other: "IndexingStats") -> None:
        self.total_items += other.total_items
        self.successful_items += other.successful_items
        self.failed_items += other.failed_items
        self.failed_ids.extend(other.failed_ids)
        self.processing_time += other.processing_time

        unique_errors = set(self.errors)
        unique_errors.update(other.errors)
        self.errors = list(unique_errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_items": self.total_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "processing_time": self.processing_time,
            "error_count": len(self.errors),
            "sample_errors": self.errors[:5],
        }


# The item kinds that carry schema annotations, as the stores name them.
AnnotatedKind = Literal["entities", "relationships"]
# (in_schema, iri) of one item; ``None`` leaves it unset (absent).
SchemaAnnotation = tuple[bool | None, str | None]


class StoredAnnotation(BaseModel):
    """The schema annotations a store holds for one item, with the type
    they follow (``domain.ingestion.schema_annotation``)."""

    model_config = ConfigDict(frozen=True)

    type: str | None = Field(
        default=None,
        description="The item's stored type; None when the store does not "
        "hold it (vector copies of relationships)",
    )
    in_schema: bool | None = Field(default=None, description="Stored in_schema")
    iri: str | None = Field(
        default=None, description="Stored type_iri (entity) or predicate_iri"
    )


class BaseIndexer(ABC):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.stats = IndexingStats()

    @abstractmethod
    def clear(self, suffixes: list[str]) -> None:
        """Delete the stored items of ``suffixes``.

        Raises the backend's typed store error when the delete fails, so the
        caller sees why (denied, unreachable, misconfigured).
        """

    def initialize(self) -> None:
        """Prepare the backend for writes; raises its typed store error.

        Default no-op: a backend that creates what it writes on the first
        write (OpenSearch indices, Neptune vertices) has nothing to prepare.
        """
        return None

    def close(self) -> None:
        """Release any backing clients/connections. Default no-op.

        Adapters holding network clients (Neptune/OpenSearch) override this;
        in-memory implementations need no teardown.
        """
        return None

    @classmethod
    def get_suffix(cls, item: Any) -> str:
        if not hasattr(item, "attributes") or not isinstance(item.attributes, dict):
            return namespace_suffix(None)
        return namespace_suffix(
            index_suffix(item.attributes.get(Constants.INDEX.value))
        )

    def _get_name(
        self, base: str, suffix: str | None, add_timestamp: bool = False
    ) -> str:
        return store_name(
            base,
            suffix,
            self.config.indexing.additional_suffix,
            timestamp=add_timestamp,
        )

    def _get_label(self, prefix: str, suffix: str | None) -> str:
        """Graph vertex label for ``prefix`` under ``suffix``."""
        return graph_label(prefix, suffix, self.config.indexing.additional_suffix)

    def _last_of_each_id(self, items: list[Any]) -> list[Any]:
        """``items`` with one per id and suffix, the last given (as an
        upsert keeps the last write)."""
        return list({(self.get_suffix(item), item.id): item for item in items}.values())

    def _group_items_by_suffix(
        self, items: list[Any], suffixes: Collection[str] = ()
    ) -> dict[str, list[Any]]:
        """``items`` by suffix; each of ``suffixes`` is present, if empty."""
        grouped: dict[str, list[Any]] = defaultdict(list)
        for suffix in suffixes:
            grouped[suffix] = []
        for item in items:
            grouped[self.get_suffix(item)].append(item)

        if len(grouped) > 1:
            logger.info(
                "Grouped %s items into %s different suffixes", len(items), len(grouped)
            )

        return dict(grouped)


class GraphIndexer(BaseIndexer):
    """Write-side port for the knowledge-graph backend (full + delta).

    This ABC is the single write-side contract for graph stores: the full-run
    ``index_*`` methods plus the incremental ``upsert_*``/``delete_by_id`` surface
    used by :class:`~unified_kg_rag.application.ingestion.incremental.IncrementalIndexer`.
    Adapters (e.g. NeptuneIndexer) implement it; nothing depends on a concrete
    backend.

    A full-run ``index_*`` replaces the stored items of every suffix its items
    carry and of every one in ``suffixes``: a suffix the rebuild has none of
    is emptied rather than left serving the previous run's items.
    """

    @abstractmethod
    def index_entities(
        self, entities: list[Entity], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        pass

    def index_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        """Write the edges; replacing ``index_entities`` already dropped the
        suffix's old ones with their vertices.

        Delegates to :meth:`upsert_relationships`: each edge is replaced by id,
        so a full run and a delta share one write path.
        """
        return self.upsert_relationships(relationships)

    @abstractmethod
    def index_communities(
        self, communities: list[Community], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        pass

    @abstractmethod
    def upsert_entities(self, entities: list[Entity]) -> IndexingStats:
        """Idempotently merge entities into the live graph (delta semantics)."""

    @abstractmethod
    def upsert_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        """Idempotently merge relationships into the live graph (delta)."""

    def update_relationship_ranks(
        self, relationships: list[Relationship]
    ) -> IndexingStats:
        """Set the ``rank`` of stored edges by id, in place, and nothing else.

        An incremental commit rewrites the stored edges of an entity whose
        degree changed for their rank alone: most of a delta's edge writes
        once a hub gains neighbours. An adapter that can update a property
        in place overrides this (batched, no drop and re-add); an id whose
        edge the suffix does not store is a failed item. Default rewrites
        each edge (:meth:`upsert_relationships`).
        """
        return self.upsert_relationships(relationships)

    def upsert_communities(self, communities: list[Community]) -> IndexingStats:
        """Idempotently merge communities into the live graph (delta semantics).

        Default delegates to :meth:`index_communities`; adapters whose
        ``index_communities`` clears the community label (and would therefore wipe
        out-of-delta communities on an incremental run) MUST override this with a
        non-clearing upsert.
        """
        return self.index_communities(communities)

    def read_entities(self, ids: list[str], suffix: str | None = None) -> list[Entity]:
        """Read existing entities by id for cross-run merge (read-merge-write).

        Scoped to ``suffix``'s graph (``None`` is the default suffix), since
        entity ids are suffix-independent. Must return every field the write
        path stores (attributes, ``text_unit_ids``, ``community_ids``, rank,
        confidence): the merged entity is written back, and its attributes
        decide the suffix it is written under.

        Default returns ``[]`` (no read-back) so an adapter that cannot or does
        not support reads simply falls back to overwrite-on-upsert. Adapters that
        can read existing state (Neptune; the test fakes) override this, and
        raise when a read fails: ``[]`` means nothing is stored.
        """
        return []

    def read_relationships(
        self, ids: list[str], suffix: str | None = None
    ) -> list[Relationship]:
        """Read existing relationships by id for cross-run merge. See
        :meth:`read_entities`.

        Scoped to ``suffix``'s graph (``None`` is the default suffix), since
        relationship ids are suffix-independent. Must return every field the
        write path stores (type, endpoint names, ``text_unit_ids``, attributes):
        the merge keys on ``(source, target, type)`` and the result is written
        back."""
        return []

    def read_entity_names(self, suffix: str | None = None) -> list[tuple[str, str]]:
        """Project ``(id, name)`` for all existing entities (for fuzzy merge).

        Used by cross-run *fuzzy* merge to find near-duplicate existing entities
        whose ids differ from a delta entity's (exact-id read-back can only
        surface entities that already normalize-equal a delta name). Default
        returns ``[]`` (no projection) so adapters that cannot enumerate simply
        skip fuzzy merge and fall back to exact-name matching. ``suffix`` scopes
        the projection to one tenant/version's label."""
        return []

    def begin_rebuild(self, suffixes: Collection[str]) -> bool:
        """Start a full build that replaces ``suffixes``' graph in place,
        so the graph is never empty meanwhile: until the sweep, queries read
        the items the build already rewrote alongside the previous ones it
        has not reached (and the ones it will drop).

        Returns whether the backend can: then ``index_entities`` and
        ``index_communities`` no longer clear the labels first, and
        :meth:`end_rebuild` drops what the build did not write. The default
        cannot (False): the full-build writes clear each label first.
        """
        return False

    def end_rebuild(self, suffixes: Collection[str], sweep: bool = True) -> None:
        """End a rebuild :meth:`begin_rebuild` started: drop the vertices and
        edges of ``suffixes`` it did not write, unless ``sweep`` is False (a
        write failed: the previous copies stay for a retry). Default: no-op."""
        return None

    def find_ids_citing(
        self,
        text_unit_ids: list[str],
        suffix: str | None = None,
        *,
        communities: bool = False,
    ) -> dict[str, list[str]] | None:
        """``{vertex id: its text_unit_ids}`` of the entities (or, with
        ``communities``, the community vertices) of ``suffix`` citing any of
        ``text_unit_ids``.

        An incremental run finds what a removed document contributed by its
        text units (every artifact cites the units it was stated in). Default
        returns ``None``: the backend cannot tell (nothing is removed through
        it); adapters that override it raise when the read fails.
        """
        return None

    def find_relationship_ids_citing(
        self, text_unit_ids: list[str], suffix: str | None = None
    ) -> dict[str, list[str]] | None:
        """``{edge id: its text_unit_ids}`` of the relationship edges of
        ``suffix`` citing any of ``text_unit_ids``.

        A removal finds the edges a removed document stated by its text
        units, as it finds the entities (:meth:`find_ids_citing`), rather
        than only among the edges of the entities citing them: an edge a
        killed run re-keyed and did not get to delete still cites them after
        its endpoints were stripped. Default returns ``None`` (the removal
        then reads the edges incident to the entities citing the units);
        adapters that override it raise when the read fails.
        """
        return None

    def find_entity_ids_by_name_key(
        self, name_keys: list[str], suffix: str | None = None
    ) -> list[str] | None:
        """Ids of the entities of ``suffix`` whose ``compact_name_key`` is one
        of ``name_keys`` (stored with the entity).

        The cross-run merge links names by that key (``FuzzyMatcher`` without
        fuzzy resolution), so the stored entities a delta can merge into are
        found by its names' keys instead of reading every name. Default
        returns ``None`` (the merge then reads every name,
        :meth:`read_entity_names`).
        """
        return None

    def read_neighbor_ids(
        self,
        entity_ids: list[str],
        suffix: str | None = None,
        *,
        document_backed: bool = False,
    ) -> dict[str, list[str]]:
        """Return ``{entity id: ids of the entities it is related to}``.

        Scoped to ``suffix``'s graph. Used to recompute the rank (degree) of
        the entities an incremental merge writes, over the document-backed
        edges only (``document_backed``): an edge whose stored
        ``source_kinds`` is ``SEED_ONLY_SOURCE_KINDS`` (every one of its text
        units a seed unit) is then not followed; one without the property
        is. Default returns ``{}`` (the rank then counts the delta's
        relationships only); adapters that can read the graph override it
        and raise when the read fails.
        """
        return {}

    def count_neighbors(
        self,
        entity_ids: list[str],
        suffix: str | None = None,
        at_most: int | None = None,
    ) -> dict[str, int]:
        """Return ``{entity id: how many entities it is related to}`` (the
        length of its :meth:`read_neighbor_ids` list) for the stored ones,
        each count capped at ``at_most`` (a store stops counting there).

        The incremental community walk tells hubs by it without reading
        their neighbours. Default counts :meth:`read_neighbor_ids`; adapters
        override it with a count the store computes, and raise when the read
        fails.
        """
        return {
            i: len(n) if at_most is None else min(len(n), at_most)
            for i, n in self.read_neighbor_ids(entity_ids, suffix).items()
        }

    def read_schema_annotations(
        self, kind: AnnotatedKind, suffix: str | None = None
    ) -> dict[str, StoredAnnotation] | None:
        """``{id: annotations}`` of every stored item of ``kind`` in
        ``suffix``'s graph (``None``: the default suffix), with its type
        (an entity's ``type``, a relationship's edge label).

        A schema edit that changes only the annotations is applied by
        reading them back and rewriting the ones that differ
        (``IndexingManager.reannotate``), with no extraction. Default
        returns ``None``: the backend cannot enumerate its items, and only a
        reset rewrites them. Adapters that override it (and
        :meth:`write_schema_annotations`, see ``rewrites_schema_annotations``)
        read in bounded requests, and raise when a read fails.
        """
        return None

    def holds_any_item(self, suffix: str | None = None) -> bool | None:
        """Whether ``suffix``'s graph holds any entity or relationship (an
        item :meth:`read_schema_annotations` lists).

        An incremental run whose registry holds none of its namespace's
        documents asks it before it records, without reading them, that the
        stores hold only what it writes: a lost or replaced registry is no
        proof the stores are empty. Default: from
        :meth:`read_schema_annotations` (``None`` when it cannot list);
        adapters that override it read one item and raise when a read fails.
        """
        return _holds_any_item(self, suffix)

    def holds_any_relationship(self, suffix: str | None = None) -> bool | None:
        """Whether ``suffix``'s graph holds a relationship (an edge between
        two of its entities).

        A build record rewritten from the stores asks it to tell an empty
        relationship vector index apart: beside a stored edge, the index
        was never built. Default: from :meth:`read_schema_annotations`
        (``None`` when it cannot list); adapters that override it read one
        edge and raise when the read fails.
        """
        listed = self.read_schema_annotations("relationships", suffix)
        return None if listed is None else bool(listed)

    def write_schema_annotations(
        self,
        kind: AnnotatedKind,
        annotations: Mapping[str, SchemaAnnotation],
        suffix: str | None = None,
    ) -> IndexingStats:
        """Set ``in_schema`` and the IRI (``type_iri`` of an entity,
        ``predicate_iri`` of a relationship) of the stored items of ``kind``
        in ``suffix`` by id, nothing else; a ``None`` value removes it. An
        id not stored is skipped. Default writes nothing (a backend that
        cannot :meth:`read_schema_annotations`)."""
        return IndexingStats()

    @abstractmethod
    def delete_by_id(self, ids: list[str], suffix: str | None = None) -> IndexingStats:
        """Delete vertices/edges by id (for removed/changed documents).

        Scoped to ``suffix``'s labels (``None``: the default suffix, as for
        every method), so a content-hash id shared across suffixes does not
        delete another tenant's data."""

    def find_incident_relationship_ids(
        self, entity_ids: list[str], suffix: str | None = None
    ) -> list[str]:
        """Return ids of ``suffix``'s edges incident to any of ``entity_ids``.

        Used to find relationship documents orphaned when an entity vertex is
        dropped (the edge is owned in lineage by a surviving document, so it is
        not in the exclusive id-set). Default returns ``[]`` for backends that
        cannot answer this; adapters that can (Neptune) override it and raise
        when the read fails.
        """
        return []


class VectorIndexer(BaseIndexer):
    """Write-side port for the vector/lexical backend (full + delta).

    Single write-side contract for vector stores: full-run ``index_*`` plus the
    incremental ``upsert_*``/``delete_by_id`` surface. Adapters (e.g.
    OpenSearchIndexer) implement it. A full-run ``index_*`` replaces its
    suffixes' items, those in ``suffixes`` included (see :class:`GraphIndexer`).
    """

    @abstractmethod
    def index_text_units(
        self, text_units: list[TextUnit], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        pass

    @abstractmethod
    def index_entities(
        self, entities: list[Entity], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        pass

    @abstractmethod
    def index_community_reports(
        self, reports: list[CommunityReport], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        pass

    @abstractmethod
    def index_relationships(
        self, relationships: list[Relationship], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        """Embed and index relationship descriptions (LightRAG global)."""

    @abstractmethod
    def index_claims(
        self, claims: list[Claim], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        """Embed and index claim (covariate) descriptions as searchable vectors."""

    @abstractmethod
    def upsert_community_reports(self, reports: list[CommunityReport]) -> IndexingStats:
        """Upsert community reports by id into the live index (delta).

        Distinct from index_community_reports (full alias-swap): the delta path
        must upsert so it does not wipe unchanged documents' reports.
        """

    @abstractmethod
    def upsert_claims(self, claims: list[Claim]) -> IndexingStats:
        """Upsert claim vectors by id into the live index (delta)."""

    @abstractmethod
    def upsert_text_units(self, text_units: list[TextUnit]) -> IndexingStats:
        """Upsert text units by id into the live index (delta)."""

    @abstractmethod
    def upsert_entities(self, entities: list[Entity]) -> IndexingStats:
        """Upsert entities by id into the live index (delta)."""

    @abstractmethod
    def upsert_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        """Upsert relationship vectors by id into the live index (delta)."""

    def check_index_settings(self, suffix: str | None = None) -> None:
        """Raise before an incremental write when the live indices of
        ``suffix`` were built for other settings.

        An incremental run writes only its new and changed documents, so the
        live indices must hold vectors of the model it embeds with, map their
        text fields for its languages, and have been built by a run that
        recorded their documents in the doc-status registry. Raises
        :class:`~unified_kg_rag.shared.ConfigurationError` on a mismatch.
        Indices that pass may be brought up to what this release writes, in
        one additive mapping update before any write (OpenSearch adds the
        fields they lack and turns date detection off). Default does nothing
        (a backend that records no index settings).
        """
        return None

    def set_doc_status_tracking(self, tracked: bool) -> None:
        """Record whether the doc-status registry tracks the documents of the
        indices a full build creates from now on (``check_index_settings``
        refuses incremental writes into an untracked one). Default does
        nothing."""
        return None

    def set_embedding_reuse(self, reuse: bool) -> None:
        """Whether writes from now on may reuse the stored vector of a text
        they embed unchanged (``False``: every text is embedded again, as
        ``--force-rebuild`` asks). Default does nothing."""
        return None

    def set_replaced_vector_keeping(self, keep: bool) -> None:
        """Whether upserts from now on keep the stored vectors they replace,
        for a later write of the same text in this run (``False`` before a
        run's last write: none follows to take them). Default does
        nothing."""
        return None

    def record_build(
        self, record: IndexBuildRecord, suffix: str | None
    ) -> IndexBuildRecord | None:
        """Record on ``suffix``'s indices which ingestion run last wrote them.

        Read back by ``run-eval`` (``VectorRetriever`` side) so a run
        manifest names the index it was scored against. Returns the record
        as written (``IndexBuildRecord.over`` the stored one, which carries
        its community counters forward), None when an index did not take it
        (the caller writes it again). Best-effort: a failure is logged, not
        raised. Default does nothing (a backend that
        keeps no index metadata).
        """
        return None

    def read_build_record(self, suffix: str | None) -> dict[str, Any] | None:
        """The build record ``record_build`` last wrote on ``suffix``'s
        indices (an ``IndexBuildRecord`` dump), read before a run to tell
        what the namespace holds. ``None`` when there is none or it cannot
        be read (logged). Default: ``None`` (no index metadata)."""
        return None

    def describe_indices(self, suffix: str | None) -> list[dict[str, Any]] | None:
        """One entry per index of ``suffix``, in ``STORED_ITEM_KINDS`` order:
        ``kind``, ``alias``, ``index`` (the live index, ``None`` without
        one), ``documents`` (how many it holds) and ``meta`` (its ``_meta``:
        the embedding model and dimension, the build record). ``None`` (the
        default) when the backend keeps no index metadata. Raises when a
        read fails."""
        return None

    def scan_items(
        self,
        kind: StoredItemKind,
        suffix: str | None,
        *,
        page_size: int | None = None,
    ) -> Iterator[list[dict[str, Any]]] | None:
        """The stored items of ``kind`` of ``suffix`` (their stored fields
        and ``id``, no vector), in id order, one page at a time
        (``page_size`` at most, the index's window otherwise): a page is
        read only once the one before it is consumed, so a caller streams
        the namespace. Yields nothing without a live index; ``None`` (the
        default) when the backend cannot page its items. Raises when a read
        fails."""
        return None

    # --- the summary tree (docs/design.md#65-raptor-summary-tree-raptor-tree-opt-in) -----------------------------
    # Every method has a "cannot" default, so a backend without the tree
    # keeps working: the summary_tree stage then skips with a warning.

    def scan_text_units(
        self,
        suffix: str | None,
        *,
        fields: list[str],
        page_size: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """Every stored text unit of ``suffix`` (a dict of ``id`` and the
        stored ``fields``, a vector field included when named, as a list or
        a float32 array), in id order, read a page at a time (``page_size`` documents at most);
        the seed graphs' units are left out, as every text search leaves
        them out. ``[]`` without a live index; ``None`` (the default) when
        the backend cannot page its text units. Raises when a read fails."""
        return None

    def scan_text_unit_pages(
        self,
        suffix: str | None,
        *,
        fields: list[str],
        page_size: int | None = None,
    ) -> Iterator[list[dict[str, Any]]] | None:
        """:meth:`scan_text_units`, one page at a time: a page is read only
        once the one before it is consumed, so a caller that keeps a little
        of each unit holds one page of their text. Yields nothing without a
        live index; ``None`` when the backend cannot page its text units.
        Raises when a read fails.

        The default serves :meth:`scan_text_units` (``None`` when it
        returns ``None``) in pages of ``page_size``: a store that implements
        only that one still builds the summary tree, holding every unit's
        fields at once."""
        units = self.scan_text_units(suffix, fields=fields, page_size=page_size)
        if units is None:
            return None
        size = max(page_size or len(units), 1)
        return iter(
            [units[start : start + size] for start in range(0, len(units), size)]
        )

    def read_text_unit_fields(
        self, ids: list[str], suffix: str | None, *, fields: list[str]
    ) -> list[dict[str, Any]] | None:
        """The stored text units of ``ids`` in ``suffix`` (a dict of ``id``
        and the stored ``fields``, a vector field as a float32 array), as
        :meth:`scan_text_units` gives them, in no set order; an id not
        stored is absent. ``None`` (the default) when the backend cannot
        read them by id: a caller scans instead. Raises when a read fails."""
        return None

    def read_summary_nodes(
        self, suffix: str | None, *, fields: list[str]
    ) -> list[dict[str, Any]] | None:
        """Every node of ``suffix``'s live summary tree (``id`` and the
        stored ``fields``), in id order; ``[]`` without one, ``None`` (the
        default) when the backend keeps no summary tree. Raises when a read
        fails."""
        return None

    def read_summary_tree_record(self, suffix: str | None) -> dict[str, Any] | None:
        """The ``SummaryTreeRecord`` dump written with ``suffix``'s live
        summary tree, ``None`` without one (the default)."""
        return None

    def embed_summary_texts(
        self, texts: list[str], suffix: str | None
    ) -> list[list[float] | None] | None:
        """Vectors of ``texts`` with this store's embedding model (``None``
        for a text that failed), taking the stored vector of a text the live
        summary tree of ``suffix`` holds unchanged. ``None`` (the default):
        the backend cannot embed for the tree."""
        return None

    def write_summary_tree(
        self,
        nodes: list[SummaryNode],
        vectors: Mapping[str, Collection[float]],
        suffix: str | None,
        record: dict[str, Any],
    ) -> IndexingStats:
        """Replace ``suffix``'s summary tree with ``nodes`` (each with its
        vector in ``vectors``, by id) and ``record``, at once: queries read
        the old tree until the new one is complete. A failed write keeps the
        old tree. The default writes nothing and reports every node failed."""
        stats = IndexingStats(total_items=len(nodes))
        stats.add_error(
            f"{type(self).__name__} keeps no summary tree", max(1, len(nodes))
        )
        return stats

    def delete_summary_nodes_citing(
        self, text_unit_ids: list[str], suffix: str
    ) -> IndexingStats:
        """Delete every summary node of ``suffix`` citing one of
        ``text_unit_ids`` and every node above it: a summary paraphrases
        its children, so it goes with any of them (a removal). Default
        deletes nothing; adapters that override it report a failed read as
        failed items."""
        return IndexingStats()

    def delete_summary_nodes_outside(
        self, text_unit_ids: Collection[str], suffix: str
    ) -> IndexingStats:
        """Delete every summary node of ``suffix`` citing a text unit not in
        ``text_unit_ids`` (all a full build wrote), and every node above it.
        Default deletes nothing; adapters that override it report a failed
        read as failed items."""
        return IndexingStats()

    def delete_summary_tree(self, suffix: str) -> IndexingStats:
        """Delete ``suffix``'s summary tree whole, the indices of builds that
        never swapped in included (a full build of a run without
        ``raptor``). Default deletes nothing; adapters that override it
        report a failure as a failed item."""
        return IndexingStats()

    def add_build_artifacts(
        self, artifacts: Collection[IndexArtifact], suffix: str | None
    ) -> bool:
        """Name ``artifacts`` in the build record of ``suffix``'s indices
        besides those it names (the summary tree, built after the record);
        a record naming none is left as it is. Returns whether a record
        changed. Default changes nothing."""
        return False

    def drop_build_artifacts(
        self, artifacts: Collection[IndexArtifact], suffix: str | None
    ) -> bool:
        """Stop naming ``artifacts`` in the build record of ``suffix``'s
        indices (the summary tree, when its build failed). Returns whether a
        record changed. Default changes nothing."""
        return False

    def prune_text_unit_communities(
        self, community_ids: list[str], suffix: str | None = None
    ) -> IndexingStats:
        """Remove ``community_ids`` (deleted communities) from the
        ``community_ids`` of ``suffix``'s stored text units. Default does
        nothing; adapters that override it raise when the read fails."""
        return IndexingStats()

    def read_text_unit_ids(
        self, ids: list[str], suffix: str | None = None
    ) -> list[str] | None:
        """The ids among ``ids`` that ``suffix``'s text-unit index holds.

        An incremental run probes it to tell the vector store still holds the
        registry's documents. Default returns ``None``: the backend cannot
        tell, and the probe is skipped. Adapters that override it return
        ``[]`` without a live index and raise when a read fails.
        """
        return None

    def read_claims(self, ids: list[str], suffix: str | None = None) -> list[Claim]:
        """Read stored claims by id for the cross-run claim merge.

        Scoped to ``suffix``'s index (``None``: the default suffix). Must
        return the fields the merge folds in (description, source text,
        ``text_unit_ids``, status). Default returns ``[]`` (no read-back:
        the upsert overwrites); adapters that override it raise when a read
        fails, since ``[]`` means nothing is stored.
        """
        return []

    def read_entity_descriptions(
        self, ids: list[str], suffix: str | None = None
    ) -> dict[str, str]:
        """Stored descriptions of entities by id, in full.

        Scoped to ``suffix``'s index (``None``: the default suffix): an id
        stored only under another suffix is absent from the result.

        The cross-run merge reads entities back from the graph store, which
        may cap property length (``indexing.neptune.property_max_length``);
        the merged description is written to this store too, so a capped one
        is taken from here. Default returns ``{}`` (keep the graph's);
        adapters that override it raise when a read fails.
        """
        return {}

    def read_relationship_descriptions(
        self, ids: list[str], suffix: str | None = None
    ) -> dict[str, str]:
        """Stored descriptions of relationships by id, in full (see
        :meth:`read_entity_descriptions`)."""
        return {}

    def read_vector_copies(
        self,
        kind: Literal["entities", "relationships"],
        ids: list[str],
        suffix: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """What the stored vector copies of ``kind`` hold besides lineage,
        by id (scoped to ``suffix``; an id not stored is absent): ``name``,
        ``type``, ``confidence``, ``description``, the schema annotations
        ``in_schema`` and ``type_iri`` and the seed-graph fields ``iri``,
        ``source_kinds`` and ``seed_properties`` (its
        ``attributes.seed_properties``) of an entity, ``source_id``,
        ``target_id``, ``source_name``, ``target_name``, ``description``,
        ``rank``, ``in_schema``, ``predicate_iri`` and ``source_kinds`` of
        a relationship (a field written ``None`` is absent). Incremental
        writes compare them with the derived items, so a copy an
        interrupted run left behind or ahead of the graph copy is
        rewritten. Default ``{}`` (every item is
        written in full); adapters that override it raise when a read fails.
        """
        return {}

    def read_schema_annotations(
        self, kind: AnnotatedKind, suffix: str | None = None
    ) -> dict[str, StoredAnnotation] | None:
        """``{id: annotations}`` of every stored vector copy of ``kind`` in
        ``suffix``'s live index (see :meth:`GraphIndexer.read_schema_annotations`):
        an entity's with its ``type``, a relationship's without (a copy
        written before the index stored it has none, so the edge's is
        used). ``{}`` without a live index. Default ``None``:
        the backend cannot enumerate its documents."""
        return None

    def holds_any_item(self, suffix: str | None = None) -> bool | None:
        """Whether ``suffix``'s live indices hold any entity or relationship
        (see :meth:`GraphIndexer.holds_any_item`). Default: from
        :meth:`read_schema_annotations`."""
        return _holds_any_item(self, suffix)

    def holds_any_text_unit(self, suffix: str | None = None) -> bool | None:
        """Whether ``suffix``'s live text-unit index holds a text unit: a
        corpus none of whose units stated an item holds text units alone,
        which :meth:`holds_any_item` does not see. Default ``None``: the
        backend cannot tell. Adapters that override it read one document
        and raise when the read fails."""
        return None

    def write_schema_annotations(
        self,
        kind: AnnotatedKind,
        annotations: Mapping[str, SchemaAnnotation],
        suffix: str | None = None,
    ) -> IndexingStats:
        """Set the schema annotations of stored vector copies by id, in
        place (no re-embedding); see
        :meth:`GraphIndexer.write_schema_annotations`. A ``None`` value is
        written as null, which no filter matches, as an absent one. Default
        writes nothing."""
        return IndexingStats()

    def read_text_unit_attributes(
        self, ids: list[str], suffix: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """The stored ``attributes`` of text units by id (scoped to
        ``suffix``; a unit not stored is absent). They name the units'
        extraction records (``contributions.record_keys``). Default returns
        ``{}``; adapters that override it raise when the read fails."""
        return {}

    def read_extraction_records(
        self, keys: list[str], suffix: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """Stored text-unit extraction records by key (the extraction store).

        A key names a text's record under one extraction fingerprint
        (``domain.ingestion.contributions.record_key``); the value is the
        record's payload as written. Scoped to ``suffix``. A missing key is
        absent from the result. Default returns ``{}``: the backend keeps no
        records, and every text unit is extracted. Adapters that override it
        raise when the read fails.
        """
        return {}

    def stored_extraction_record_keys(
        self, keys: list[str], suffix: str | None = None
    ) -> set[str]:
        """The keys among ``keys`` that ``suffix``'s extraction store holds,
        without reading their records (a check that they are still there).
        Default: the keys :meth:`read_extraction_records` returns. Adapters
        that override it raise when the read fails.
        """
        return set(self.read_extraction_records(keys, suffix=suffix))

    def find_documents(
        self,
        alias_prefix: str,
        field: str,
        values: list[str],
        suffix: str | None = None,
    ) -> dict[str, list[str]] | None:
        """``{document id: its text_unit_ids}`` of the documents of
        ``suffix``'s ``alias_prefix`` index whose keyword ``field`` (dotted
        for a nested one) holds any of ``values``.

        Finds the claims and community reports citing removed text units and
        the text units naming an extraction record. Default returns ``None``:
        the backend cannot tell; adapters that override it raise when the read
        fails.
        """
        return None

    def delete_extraction_records(
        self, keys: list[str], suffix: str | None = None
    ) -> IndexingStats:
        """Delete extraction records by key (no text unit names them any
        more). Default deletes nothing."""
        return IndexingStats()

    def retain_extraction_records(
        self, keys: list[str], suffix: str | None = None
    ) -> IndexingStats:
        """Keep only the extraction records of ``suffix`` named by ``keys``
        and drop every other (a full build, once its text units are stored:
        records of other extraction settings, or of texts no stored unit
        holds any more). Raises when it fails; the records are then left as
        they were. Default keeps everything."""
        return IndexingStats()

    def extraction_record_keys(
        self, prefixes: list[str], suffix: str | None = None
    ) -> set[str] | None:
        """The keys of ``suffix``'s stored extraction records that start with
        any of ``prefixes`` (a record fingerprint and its ``:``: the records
        of one kind under one set of settings), which a full build that does
        not produce that kind keeps (``retain_extraction_records``). Default
        returns ``None``: the backend cannot tell, and such a build keeps
        every record. Adapters that override it raise when the read
        fails."""
        return None

    def prepare_extraction_records(self, suffix: str | None = None) -> None:
        """Create ``suffix``'s extraction-record store unless it exists,
        before extraction, rather than failing every record write after it.
        Raises
        :class:`~unified_kg_rag.shared.exceptions.StoreRequestRejectedError`
        (a ``ConfigurationError``) when refused. Default does nothing."""
        return None

    def check_index_creation(
        self, suffix: str | None = None, *, rebuild: bool = True
    ) -> None:
        """Raise when the backend refuses to create the indices of ``suffix``
        the run will create with this config (index settings, vector method,
        analyzers, room for them all): every one on a ``rebuild`` (a full
        build or reset), otherwise (an incremental run) only those that do
        not exist yet.

        The ingestion pipeline calls it once per run, under the namespace's
        write lease, before the first stage that may call a model: a refused
        setting then costs no model call, rather than failing the first
        index creation after extraction (a reset's, or a full build's).
        Raises
        :class:`~unified_kg_rag.shared.exceptions.StoreRequestRejectedError`
        (a ``ConfigurationError``) when refused. Default does nothing."""
        return None

    def write_extraction_records(
        self, records: dict[str, dict[str, Any]], suffix: str | None = None
    ) -> IndexingStats:
        """Store text-unit extraction records by key (an upsert; see
        :meth:`read_extraction_records`). The keys are content-derived:
        a record goes when no stored text unit names it any more (an
        incremental removal, :meth:`delete_extraction_records`) or a full
        build or reset keeps only its own units' records
        (:meth:`retain_extraction_records`). Default stores nothing."""
        return IndexingStats()

    def repoint_report_findings(
        self, id_remap: Mapping[str, str], suffix: str | None = None
    ) -> IndexingStats:
        """Point the record ids the findings of ``suffix``'s stored community
        reports cite at their new ids (``id_remap``: old id -> new id), in
        place: an incremental run re-keyed those items (their name changed)
        and deletes the old ids. One item per report rewritten. Default
        rewrites nothing (the reports keep citing the old ids, which a query
        reading the records skips); adapters that override it raise when
        the read of the citing reports fails."""
        return IndexingStats()

    def update_claim_lineage(self, claims: list[Claim]) -> IndexingStats:
        """Rewrite stored claims' ``text_unit_ids`` (text-unit removal); see
        :meth:`update_entity_lineage`."""
        return self.upsert_claims(claims)

    def update_entity_lineage(self, entities: list[Entity]) -> IndexingStats:
        """Rewrite stored entities' ``text_unit_ids`` (text-unit removal).

        Their embedded text is unchanged, so an adapter that can update
        fields in place overrides this; the default re-upserts them.
        """
        return self.upsert_entities(entities)

    def update_relationship_lineage(
        self, relationships: list[Relationship]
    ) -> IndexingStats:
        """Rewrite stored relationships' ``text_unit_ids`` and ``weight``
        (see :meth:`update_entity_lineage`)."""
        return self.upsert_relationships(relationships)

    @abstractmethod
    def delete_by_id(
        self, ids: list[str], alias_prefix: str, suffix: str
    ) -> IndexingStats:
        """Delete documents by id from the live aliased index (delta)."""

    @abstractmethod
    def delete_document_artifacts(
        self,
        ids: list[str],
        suffix: str,
        extra_relationship_ids: list[str] | None = None,
    ) -> dict[str, IndexingStats]:
        """Delete a document's artifacts from every vector index, by id.

        Fans ``ids`` out across all of this backend's artifact indices
        (text-units, entities, relationships, claims, community-reports) so the
        caller does not need to know the backend's index layout.
        ``extra_relationship_ids`` are additionally removed from the relationship
        index only (orphaned incident edges). Returns a per-index stats map.
        """


_ANNOTATION_METHODS = ("read_schema_annotations", "write_schema_annotations")


def _holds_any_item(
    store: GraphIndexer | VectorIndexer, suffix: str | None
) -> bool | None:
    """The ports' default ``holds_any_item``: a full listing of each kind."""
    kinds: tuple[AnnotatedKind, ...] = ("entities", "relationships")
    for kind in kinds:
        listed = store.read_schema_annotations(kind, suffix)
        if listed is None:
            return None
        if listed:
            return True
    return False


def rewrites_schema_annotations(store: object) -> bool:
    """Whether ``store`` can enumerate and rewrite its schema annotations:
    it implements both ``read_schema_annotations`` and
    ``write_schema_annotations`` instead of inheriting the defaults, which
    read nothing. Checked before a re-annotation pass reads anything, so a
    store that cannot does not make the pass read the other in vain.

    Compares the functions the store's attributes are bound to, so a
    wrapper that forwards its attributes (the pipeline's borrowed
    indexers) answers for the store it wraps."""
    defaults = {
        getattr(port, name)
        for port in (GraphIndexer, VectorIndexer)
        for name in _ANNOTATION_METHODS
    }
    for name in _ANNOTATION_METHODS:
        method = getattr(store, name, None)
        if method is None or getattr(method, "__func__", method) in defaults:
            return False
    return True


def keeps_build_records(store: object) -> bool:
    """Whether ``store`` keeps build records: it implements
    ``VectorIndexer.record_build`` instead of inheriting the default, which
    writes nothing (so a ``None`` from it means a write that failed). Bound
    functions are compared, as :func:`rewrites_schema_annotations` does."""
    method = getattr(store, "record_build", None)
    return method is not None and (
        getattr(method, "__func__", method) is not VectorIndexer.record_build
    )
