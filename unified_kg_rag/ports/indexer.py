# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from unified_kg_rag.domain.models import (
    Claim,
    Community,
    CommunityReport,
    Config,
    Constants,
    Entity,
    Relationship,
    TextUnit,
)
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)


class IndexingStats(BaseModel):
    total_items: int = Field(default=0)
    successful_items: int = Field(default=0)
    failed_items: int = Field(default=0)
    errors: list[str] = Field(default_factory=list)
    processing_time: float = Field(default=0.0)

    @property
    def success_rate(self) -> float:
        return self.successful_items / self.total_items if self.total_items > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.failed_items / self.total_items if self.total_items > 0 else 0.0

    def add_success(self, count: int = 1) -> None:
        self.successful_items += count

    def add_error(self, error_message: str, count: int = 1) -> None:
        self.failed_items += count
        if error_message not in self.errors:
            self.errors.append(error_message)

    def merge(self, other: "IndexingStats") -> None:
        self.total_items += other.total_items
        self.successful_items += other.successful_items
        self.failed_items += other.failed_items
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
    def clear(self, suffixes: list[str]) -> bool:
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def initialize(self) -> bool:
        pass

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

        cls._validate_suffix_format(suffix)
        return suffix

    @classmethod
    def _validate_suffix_format(cls, suffix: str) -> None:
        if not re.match(r"^[a-z0-9-_]+$", suffix):
            invalid_chars = set(re.findall(r"[^a-z0-9-_]", suffix))
            error_msg = (
                f"Invalid suffix format: '{suffix}'. "
                f"OpenSearch requires index names to be lowercase and only contain: "
                f"lowercase letters (a-z), numbers (0-9), hyphens (-), and underscores (_). "
                f"Found invalid characters: {invalid_chars}. "
                f"Please update your configuration to use a valid suffix."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _get_name(
        self, base: str, suffix: str | None, add_timestamp: bool = False
    ) -> str:
        final_suffix = suffix or Constants.DEFAULT_SUFFIX.value

        if self.config.indexing.additional_suffix:
            final_suffix = f"{final_suffix}-{self.config.indexing.additional_suffix}"

        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            return f"{base}-{final_suffix}-{timestamp}"

        return f"{base}-{final_suffix}"

    def _group_items_by_suffix(self, items: list[Any]) -> dict[str, list[Any]]:
        grouped = defaultdict(list)
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
    """

    @abstractmethod
    def index_entities(self, entities: list[Entity]) -> IndexingStats:
        pass

    @abstractmethod
    def index_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        pass

    @abstractmethod
    def index_communities(self, communities: list[Community]) -> IndexingStats:
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

    def read_entities(self, ids: list[str]) -> list[Entity]:
        """Read existing entities by id for cross-run merge (read-merge-write).

        Default returns ``[]`` (no read-back) so an adapter that cannot or does
        not support reads simply falls back to overwrite-on-upsert. Adapters that
        can read existing state (Neptune; the test fakes) override this.
        """
        return []

    def read_relationships(self, ids: list[str]) -> list[Relationship]:
        """Read existing relationships by id for cross-run merge. See
        :meth:`read_entities`."""
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

    @abstractmethod
    def delete_by_id(self, ids: list[str], suffix: str | None = None) -> IndexingStats:
        """Delete vertices/edges by id (for removed/changed documents).

        ``suffix`` scopes the deletion to one tenant/version's labels so a
        content-hash id shared across suffixes does not delete another tenant's
        data; ``None`` is unscoped (single-tenant)."""

    def find_incident_relationship_ids(
        self, entity_ids: list[str], suffix: str | None = None
    ) -> list[str]:
        """Return ids of edges incident to any of ``entity_ids``.

        Used to find relationship documents orphaned when an entity vertex is
        dropped (the edge is owned in lineage by a surviving document, so it is
        not in the exclusive id-set). Default returns ``[]`` for backends that
        cannot answer this; adapters that can (Neptune) override it.
        """
        return []

    @abstractmethod
    def get_entity_count(self, suffixes: list[str]) -> int:
        pass


class VectorIndexer(BaseIndexer):
    """Write-side port for the vector/lexical backend (full + delta).

    Single write-side contract for vector stores: full-run ``index_*`` plus the
    incremental ``upsert_*``/``delete_by_id`` surface. Adapters (e.g.
    OpenSearchIndexer) implement it.
    """

    @abstractmethod
    def index_text_units(self, text_units: list[TextUnit]) -> IndexingStats:
        pass

    @abstractmethod
    def index_entities(self, entities: list[Entity]) -> IndexingStats:
        pass

    @abstractmethod
    def index_community_reports(self, reports: list[CommunityReport]) -> IndexingStats:
        pass

    @abstractmethod
    def index_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        """Embed and index relationship descriptions (LightRAG global)."""

    @abstractmethod
    def index_claims(self, claims: list[Claim]) -> IndexingStats:
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

    @abstractmethod
    def get_entity_count(self, suffixes: list[str]) -> int:
        pass
