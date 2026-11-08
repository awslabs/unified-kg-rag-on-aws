# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph + vector stores for the incremental property tests, and their state.

The convergence properties ran on the in-memory fakes only, whose write
semantics are a model of the stores'. :func:`property_stores` opens the
same pair on ``fake`` or ``sqlite`` (the embedded stores, files in a fresh
temporary directory per example: a Hypothesis example cannot take
``tmp_path``), and the readers return what a store holds through the port
methods both implement, so one property states its expected state once.

``sqlite`` also takes an ``additional_suffix``: two namespaces over the
same files, for the isolation variants (the fakes keep one namespace).
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import Config, Entity, Relationship


class Stores:
    """A graph and a vector indexer of one namespace, and its ``config``."""

    def __init__(self, kind: str, config: Config, graph: Any, vector: Any) -> None:
        self.kind = kind
        self.config = config
        self.graph = graph
        self.vector = vector

    # --- reads ----------------------------------------------------------------

    def entities(self) -> list[Entity]:
        """The graph store's entities, by id."""
        if self.kind == "fake":
            ids = sorted(self.graph.ids("entities"))
        else:
            ids = [entity_id for entity_id, _ in self.graph.read_entity_names()]
        return list(self.graph.read_entities(ids))

    def relationships(self) -> list[Relationship]:
        """The graph store's relationships, by id."""
        if self.kind == "fake":
            ids = sorted(self.graph.ids("relationships"))
        else:
            # An edge has two stored endpoints: every one is incident to one.
            entity_ids = [entity_id for entity_id, _ in self.graph.read_entity_names()]
            ids = sorted(self.graph.find_incident_relationship_ids(entity_ids))
        return list(self.graph.read_relationships(ids))

    def vector_entities(self) -> list[Entity]:
        """The vector store's entity copies (the fields a state compares)."""
        if self.kind == "fake":
            return list(self.vector.data.get("entities", {}).values())
        prefix = self.config.indexing.opensearch.entities_index_prefix
        return [
            Entity(
                id=hit["_id"],
                name=source.get("name") or "",
                type=source.get("type") or None,
                description=source.get("description") or None,
                rank=source.get("rank"),
                confidence=source.get("confidence"),
                text_unit_ids=list(source.get("text_unit_ids") or []),
            )
            for hit in self._hits(prefix)
            for source in (hit["_source"],)
        ]

    def vector_ids(self, kind: str) -> set[str]:
        """Ids of the vector store's ``kind`` documents (``text_units``,
        ``entities``, ``relationships``, ``claims``)."""
        if self.kind == "fake":
            return set(self.vector.ids(kind))
        prefix = getattr(self.config.indexing.opensearch, f"{kind}_index_prefix")
        return {hit["_id"] for hit in self._hits(prefix)}

    def _hits(self, prefix: str) -> list[dict[str, Any]]:
        client = self.vector.opensearch_client
        alias = self.vector._get_name(prefix, None)
        if not client.get_index_name_by_alias(alias):
            return []
        return client.search(
            index=alias,
            body={
                "size": 10_000,
                "query": {"match_all": {}},
                "_source": {"excludes": ["*_embedding"]},
            },
        )["hits"]["hits"]


@contextmanager
def property_stores(
    kind: str,
    config: Config | None = None,
    *,
    root: Path | None = None,
    additional_suffix: str | None = None,
) -> Iterator[Stores]:
    """The stores of ``kind`` under ``config`` (a copy; the default when
    ``None``), closed on exit. ``sqlite`` writes its files under ``root``,
    a fresh temporary directory when ``None``."""
    config = (config or Config()).model_copy(deep=True)
    if kind == "fake":
        yield Stores(
            kind,
            config,
            FakeGraphStore(config),
            FakeVectorStore(opensearch_config=config.indexing.opensearch),
        )
        return
    if kind != "sqlite":
        raise ValueError(kind)
    with tempfile.TemporaryDirectory(prefix="ukr-prop-") as scratch:
        files = root or Path(scratch)
        config.providers.graph.kind = "sqlite"
        config.providers.graph.sqlite_path = str(files / "graph.sqlite3")
        config.providers.vector.kind = "sqlite"
        config.providers.vector.sqlite_path = str(files / "vector.sqlite3")
        if additional_suffix is not None:
            config.indexing.additional_suffix = additional_suffix
        providers = Providers(config, embedding_factory=HashingEmbeddingFactory())
        graph = providers.build_graph_indexer()
        vector = providers.build_vector_indexer()
        try:
            vector.initialize()
            yield Stores(kind, config, graph, vector)
        finally:
            graph.close()
            vector.close()
