# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``visualization_data.json`` of an index namespace, read from its stores.

Ingestion writes it at the end of every run that indexed
(``graph.visualization.export_data``), so the file shows the namespace as
queries see it: the whole graph after an incremental run, and every
corpus of a namespace several share, where the run's own graph held only
its delta or its corpus. The entities, claims and community reports come
from the vector store's pages (``VectorIndexer.scan_items``), each entity's
communities from the graph store (``GraphIndexer.read_entities``, which
holds ``community_ids``), and the relationships from the vector store, or
the graph store when the namespace has no relationships index.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from unified_kg_rag.domain.ingestion.visualization_data import (
    VISUALIZATION_DATA_FILENAME,
    stored_visualization_data,
)
from unified_kg_rag.domain.models import Claim, Entity, Relationship
from unified_kg_rag.ports.indexer import GraphIndexer, StoredItemKind, VectorIndexer
from unified_kg_rag.shared import ConfigurationError, get_logger
from unified_kg_rag.shared.utils.files import atomic_write_text

if TYPE_CHECKING:
    from unified_kg_rag.adapters.providers import Providers

logger = get_logger(__name__)

# Ids one graph-store read takes (as the incremental merge reads them).
_READ_BATCH = 500


def _scan(
    vector: VectorIndexer, kind: StoredItemKind, suffix: str
) -> list[dict[str, Any]]:
    pages = vector.scan_items(kind, suffix)
    if pages is None:
        raise ConfigurationError(
            "The vector store cannot page its items: no visualization data "
            "can be read from it"
        )
    return [item for page in pages for item in page]


def _batches(ids: list[str]) -> Iterator[list[str]]:
    for start in range(0, len(ids), _READ_BATCH):
        yield ids[start : start + _READ_BATCH]


def _entities(vector: VectorIndexer, graph: GraphIndexer, suffix: str) -> list[Entity]:
    """The namespace's entities, with the communities the graph store
    records for each (the vector store does not keep them)."""
    stored = [Entity.model_validate(e) for e in _scan(vector, "entities", suffix)]
    held = {
        e.id: e
        for batch in _batches([e.id for e in stored])
        for e in graph.read_entities(batch, suffix)
    }
    return [
        (
            entity.model_copy(update={"community_ids": held[entity.id].community_ids})
            if entity.id in held
            else entity
        )
        for entity in stored
    ]


def _relationships(
    vector: VectorIndexer,
    graph: GraphIndexer,
    suffix: str,
    entity_ids: list[str],
) -> list[Relationship]:
    stored = _scan(vector, "relationships", suffix)
    if stored:
        return [Relationship.model_validate(r) for r in stored]
    # No relationships index (indexing.opensearch.build_relationship_vector_index
    # off): the graph store's edges.
    ids = sorted(
        {
            rel_id
            for batch in _batches(entity_ids)
            for rel_id in graph.find_incident_relationship_ids(batch, suffix)
        }
    )
    return [
        r for batch in _batches(ids) for r in graph.read_relationships(batch, suffix)
    ]


def write_visualization_data(
    providers: Providers, suffix: str, path: Path
) -> dict[str, int]:
    """Write the visualization data of the index namespace ``suffix``
    (``resolve_namespace``) to ``path``, atomically; the nodes, edges and
    communities written.

    Raises:
        ConfigurationError: The vector store cannot page its items.
        StoreConnectionError: A store cannot be reached.
    """
    vector = providers.build_vector_indexer()
    graph = providers.build_graph_indexer()
    try:
        entities = _entities(vector, graph, suffix)
        relationships = _relationships(vector, graph, suffix, [e.id for e in entities])
        claims = [Claim.model_validate(c) for c in _scan(vector, "claims", suffix)]
        reports = _scan(vector, "community_reports", suffix)
    finally:
        vector.close()
        graph.close()
    data = stored_visualization_data(
        entities,
        relationships,
        claims,
        reports,
        namespace=suffix,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(data, default=str))
    return {
        "nodes": len(data["nodes"]),
        "edges": len(data["edges"]),
        "communities": len(data["communities"]["hierarchy"]),
    }


# How the file begins (stored_visualization_data writes the namespace first).
_NAMESPACE_HEAD = re.compile(r'^\{"namespace": "([^"\\]*)"')


def export_namespace(path: Path) -> str | None:
    """The namespace an export names in its first bytes (``None`` for an
    export written before they named one)."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            head = handle.read(512)
    except OSError:
        return None
    match = _NAMESPACE_HEAD.match(head)
    return match.group(1) if match else None


def newest_export(
    directories: Iterable[Path], namespace: str | None = None
) -> Path | None:
    """The most recently written ``visualization_data.json`` under
    ``directories`` (searched recursively), of ``namespace`` when given."""
    found = [
        path
        for directory in directories
        if directory.is_dir()
        for path in directory.rglob(VISUALIZATION_DATA_FILENAME)
        if path.is_file() and (namespace is None or export_namespace(path) == namespace)
    ]
    return max(found, key=lambda path: path.stat().st_mtime, default=None)
