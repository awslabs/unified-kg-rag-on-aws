# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Collection
from typing import Any

from pydantic import BaseModel, Field

from unified_kg_rag.domain.models import (
    Claim,
    Community,
    CommunityReport,
    Config,
    Constants,
    Entity,
    IndexBuildRecord,
    Relationship,
    TextUnit,
    validate_safe_name,
)
from unified_kg_rag.shared import ConfigurationError, get_logger
from unified_kg_rag.shared.utils.store_names import graph_label, store_name

logger = get_logger(__name__)


class IndexingStats(BaseModel):
    total_items: int = Field(default=0)
    successful_items: int = Field(default=0)
    failed_items: int = Field(default=0)
    errors: list[str] = Field(default_factory=list)
    # Ids of the items whose write failed, where the failure site knows them,
    # kept with multiplicity: a failure site records exactly one id per failed
    # item, so an artifact whose write failed twice (e.g. two MemberOf edge
    # batches of one community) appears twice. An incremental commit records
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
            return Constants.DEFAULT_SUFFIX.value

        index_value = item.attributes.get(Constants.INDEX.value)
        suffix = None

        if isinstance(index_value, str) and index_value:
            suffix = index_value
        elif isinstance(index_value, (list | tuple)) and index_value:
            suffix = str(index_value[0])
        else:
            return Constants.DEFAULT_SUFFIX.value

        try:
            return validate_safe_name(suffix, "index suffix")
        except ValueError as e:
            raise ConfigurationError(str(e)) from e

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
        so the graph keeps serving its previous content meanwhile.

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
        self, entity_ids: list[str], suffix: str | None = None
    ) -> dict[str, list[str]]:
        """Return ``{entity id: ids of the entities it is related to}``.

        Scoped to ``suffix``'s graph. Used to recompute the rank (degree) of
        the entities an incremental merge writes. Default returns ``{}`` (the
        rank then counts the delta's relationships only); adapters that can
        read the graph override it and raise when the read fails.
        """
        return {}

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
        Default does nothing (a backend that records no index settings).
        """
        return None

    def set_doc_status_tracking(self, tracked: bool) -> None:
        """Record whether the doc-status registry tracks the documents of the
        indices a full build creates from now on (``check_index_settings``
        refuses incremental writes into an untracked one). Default does
        nothing."""
        return None

    def record_build(self, record: IndexBuildRecord, suffix: str | None) -> None:
        """Record on ``suffix``'s indices which ingestion run last wrote them.

        Read back by ``run-eval`` (``VectorRetriever`` side) so a run
        manifest names the index it was scored against. Best-effort: a
        failure is logged, not raised. Default does nothing (a backend that
        keeps no index metadata).
        """
        return None

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

    def write_extraction_records(
        self, records: dict[str, dict[str, Any]], suffix: str | None = None
    ) -> IndexingStats:
        """Store text-unit extraction records by key (an upsert; see
        :meth:`read_extraction_records`). The keys are content-derived, so
        the store is only ever added to: a full build, a reset or a document
        deletion leaves it as it is. Default stores nothing."""
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
