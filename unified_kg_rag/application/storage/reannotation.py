# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema re-annotation of the stored items: no model call, no embedding."""

from collections.abc import Collection
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from unified_kg_rag.ports.indexer import (
    AnnotatedKind,
    GraphIndexer,
    SchemaAnnotation,
    VectorIndexer,
    rewrites_schema_annotations,
)
from unified_kg_rag.shared import NamespaceLeaseError, get_logger
from unified_kg_rag.shared.logging import error_log_text

if TYPE_CHECKING:
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

logger = get_logger(__name__)


class SchemaReannotation(BaseModel):
    """What one annotation-only pass (``IndexingManager.reannotate``) did."""

    entities: int = Field(default=0, description="Stored entities read")
    relationships: int = Field(default=0, description="Stored relationships read")
    entities_rewritten: int = Field(
        default=0, description="Entities whose annotations a store rewrote"
    )
    relationships_rewritten: int = Field(
        default=0, description="Relationships whose annotations a store rewrote"
    )
    failed: int = Field(default=0, description="Annotation writes that failed")
    unsupported: bool = Field(
        default=False,
        description="A store cannot enumerate its items: its annotations "
        "were not rewritten",
    )
    error: str | None = Field(
        default=None,
        description="The store read or write that raised, where the pass "
        "stopped; None when none did",
    )

    @property
    def complete(self) -> bool:
        """Whether every store now holds the current annotations."""
        return not self.failed and not self.unsupported and self.error is None


class SchemaReannotator:
    """Annotation-only passes over the stored items, for :class:`IndexingManager`
    (see :meth:`IndexingManager.reannotate`)."""

    def __init__(self, manager: "IndexingManager") -> None:
        self._m = manager

    def stores_hold_items(self, suffix: str) -> bool:
        """See :meth:`IndexingManager.stores_hold_items`."""
        m = self._m
        stores: tuple[GraphIndexer | VectorIndexer, ...] = (
            m.graph_indexer,
            m.vector_indexer,
        )
        if not all(rewrites_schema_annotations(store) for store in stores):
            return False
        return any(store.holds_any_item(suffix) for store in stores)

    def reannotate(self, suffixes: Collection[str]) -> SchemaReannotation:
        """See :meth:`IndexingManager.reannotate`."""
        m = self._m
        result = SchemaReannotation()
        stores: tuple[GraphIndexer | VectorIndexer, ...] = (
            m.graph_indexer,
            m.vector_indexer,
        )
        if not all(rewrites_schema_annotations(store) for store in stores):
            result.unsupported = True
            return result
        try:
            for suffix in sorted(suffixes):
                kinds: tuple[AnnotatedKind, ...] = ("entities", "relationships")
                for kind in kinds:
                    self._reannotate_kind(kind, suffix, stores, result)
        except NamespaceLeaseError:
            raise
        except Exception as e:  # noqa: BLE001 - retried by the next run
            logger.debug("Schema re-annotation failed: %s", e)
            result.error = error_log_text(e)
        if result.complete:
            logger.info(
                "Schema annotations: checked %d stored entities and %d stored "
                "relationships in %s and rewrote in_schema and the IRIs of %d "
                "and %d of them, without extracting again",
                result.entities,
                result.relationships,
                ", ".join(f"'{s}'" for s in sorted(suffixes)),
                result.entities_rewritten,
                result.relationships_rewritten,
            )
        return result

    def _reannotate_kind(
        self,
        kind: AnnotatedKind,
        suffix: str,
        stores: tuple[GraphIndexer | VectorIndexer, ...],
        result: SchemaReannotation,
    ) -> None:
        """One kind of one suffix of :meth:`reannotate`, counted in ``result``."""
        m = self._m
        annotate = (
            m._annotator.entity_annotation
            if kind == "entities"
            else m._annotator.relationship_annotation
        )
        graph, vector = (s.read_schema_annotations(kind, suffix) for s in stores)
        types = {i: held.type for i, held in (graph or {}).items()}
        rewritten: set[str] = set()
        for store, stored in zip(stores, (graph, vector), strict=True):
            if stored is None:
                result.unsupported = True
                continue
            changes: dict[str, SchemaAnnotation] = {}
            for item_id, held in stored.items():
                if held.type is None and item_id not in types:
                    continue
                target = annotate(
                    held.type if held.type is not None else types[item_id]
                )
                if (held.in_schema, held.iri) != target:
                    changes[item_id] = target
            if changes:
                stats = store.write_schema_annotations(kind, changes, suffix)
                result.failed += stats.failed_items
                rewritten |= changes.keys()
        read = len(types) if graph is not None else len(vector or {})
        if kind == "entities":
            result.entities += read
            result.entities_rewritten += len(rewritten)
        else:
            result.relationships += read
            result.relationships_rewritten += len(rewritten)
