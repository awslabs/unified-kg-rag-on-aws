# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``visualization_data.json`` of an index namespace, read from its stores.

Ingestion writes it at the end of a full build
(``graph.visualization.export_data``), and ``run-visualization`` when no
export shows the build the stores hold (:func:`current_visualization_data`:
after an incremental run, which exports nothing), so the file shows the
namespace as queries see it: the whole graph, and every corpus of a
namespace several share, where a run's own graph held only its delta or
its corpus. The entities, claims and community reports come
from the vector store's pages (``VectorIndexer.scan_items``), each entity's
communities from the graph store (``GraphIndexer.read_entities``, which
holds ``community_ids``), and the relationships from the vector store, or
the graph store when the namespace has no relationships index.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

from unified_kg_rag.domain.ingestion.visualization_data import (
    VISUALIZATION_DATA_FILENAME,
    stored_visualization_data,
)
from unified_kg_rag.domain.models import Claim, Entity, Relationship
from unified_kg_rag.ports.indexer import GraphIndexer, StoredItemKind, VectorIndexer
from unified_kg_rag.shared import ConfigurationError, DataProcessingError, get_logger
from unified_kg_rag.shared.utils import error_name
from unified_kg_rag.shared.utils.files import atomic_write_text

if TYPE_CHECKING:
    from unified_kg_rag.adapters.providers import Providers

logger = get_logger(__name__)


class ExportWriteError(DataProcessingError, OSError):
    """The export file (or its folder) could not be written, as distinct from
    the stores it was read from failing: a caller says which of the two. Its
    message names the path and the OS error (``error_name``: errno and path,
    no data), so it is logged and shown in full (``has_own_text``)."""

    own_text = True


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
        ExportWriteError: ``path`` (or its folder) cannot be written.
    """
    with ExitStack() as stores:
        vector = providers.build_vector_indexer()
        stores.callback(vector.close)
        graph = providers.build_graph_indexer()
        stores.callback(graph.close)
        built_at = _built_at(vector, suffix)
        entities = _entities(vector, graph, suffix)
        relationships = _relationships(vector, graph, suffix, [e.id for e in entities])
        claims = [Claim.model_validate(c) for c in _scan(vector, "claims", suffix)]
        reports = _scan(vector, "community_reports", suffix)
    data = stored_visualization_data(
        entities,
        relationships,
        claims,
        reports,
        namespace=suffix,
        built_at=built_at,
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(data, default=str))
    except OSError as e:
        raise ExportWriteError(f"Cannot write '{path}': {error_name(e)}") from e
    return {
        "nodes": len(data["nodes"]),
        "edges": len(data["edges"]),
        "communities": len(data["communities"]["hierarchy"]),
    }


def _built_at(vector: VectorIndexer, suffix: str) -> str | None:
    """When the build ``suffix``'s record names finished (``None`` without
    one)."""
    built_at = (vector.read_build_record(suffix) or {}).get("built_at")
    return str(built_at) if built_at else None


def stored_export(providers: Providers, suffix: str) -> tuple[str | None, str | None]:
    """When the build the stores of ``suffix`` hold finished, as an export
    names it, and the run that exports it (``visualization_export_run``):
    ``None`` for either the build record does not name."""
    vector = providers.build_vector_indexer()
    try:
        record = vector.read_build_record(suffix) or {}
    finally:
        vector.close()
    built_at, run = record.get("built_at"), record.get("visualization_export_run")
    return (
        str(built_at) if built_at else None,
        run if isinstance(run, str) and run else None,
    )


# How the file begins (stored_visualization_data writes the namespace, then
# the build, first).
_HEAD = re.compile(
    r'^\{"namespace": "([^"\\]*)"(?:, "built_at": (?:"([^"\\]*)"|null))?'
)


# How many leading bytes hold the namespace and the build (``_HEAD``).
EXPORT_HEAD_BYTES = 512


def _head(path: Path) -> re.Match[str] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            head = handle.read(EXPORT_HEAD_BYTES)
    except OSError:
        return None
    return _HEAD.match(head)


def export_head(head: str) -> tuple[str | None, str | None]:
    """The namespace and the build time an export's first bytes ``head``
    name (``None`` for either it does not)."""
    match = _HEAD.match(head)
    return (match.group(1), match.group(2)) if match else (None, None)


def export_namespace(path: Path) -> str | None:
    """The namespace an export names in its first bytes (``None`` for an
    export written before they named one)."""
    match = _head(path)
    return match.group(1) if match else None


def export_build_time(path: Path) -> str | None:
    """When the build an export shows finished, as its first bytes name it
    (``None`` for one without, or written before exports named it)."""
    match = _head(path)
    return match.group(2) if match else None


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


def current_visualization_data(
    providers: Providers, suffix: str, directories: Iterable[Path], path: Path
) -> tuple[Path, dict[str, int] | None]:
    """An export that shows what the stores of the index namespace
    ``suffix`` hold: the newest under ``directories`` or at ``path`` that
    names the build its build record names, else one written to ``path``
    from the stores (:func:`write_visualization_data`); with the counts
    written, ``None`` for an export reused.

    Raises:
        ConfigurationError: The vector store cannot page its items.
        StoreConnectionError: A store cannot be reached.
    """
    built_at, _ = stored_export(providers, suffix)
    if built_at is not None:
        found = newest_export([*directories, path.parent], suffix)
        if found is not None and export_build_time(found) == built_at:
            return found, None
    return path, write_visualization_data(providers, suffix, path)
