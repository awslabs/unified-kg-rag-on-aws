# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One graph + vector store pair per backend, for the store conformance suites.

A suite's harness fixture is parametrized over :data:`BACKENDS` over every backend the framework writes to,
so one test states a store contract once and runs it on each:

- ``fake``: ``FakeGraphStore`` / ``FakeVectorStore`` (the in-memory reference
  for write semantics; no retrievers, no index administration).
- ``sqlite``: the embedded stores (``providers.*.kind: sqlite``) in
  ``tmp_path``.
- ``container``: Gremlin Server and OpenSearch of ``docker/compose.local.yaml``,
  reached through ``docker/config.local.yaml``. Skipped unless
  ``LOCAL_STORES=1``. ``LOCAL_STORES_GREMLIN_PORT`` and
  ``LOCAL_STORES_OPENSEARCH_PORT`` point it at a stack published on other
  ports.
- ``aws``: Neptune and OpenSearch Service (``pytest.mark.aws``). The config is
  ``GRAPHRAG_TEST_CONFIG`` if set, else the defaults; ``NEPTUNE_ENDPOINT`` and
  ``OPENSEARCH_ENDPOINT`` override the endpoints. Skipped without both
  endpoints.

Every harness of a shared server writes under its own
``indexing.additional_suffix`` and a hybrid search pipeline of its own, and
clears every suffix it handed out (``StoreHarness.suffix``) on teardown, so a
run leaves nothing behind. Embeddings come from ``HashingEmbeddingFactory``: no
model is called on any backend.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import (
    Config,
    Constants,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
)
from unified_kg_rag.ports.indexer import BaseIndexer
from unified_kg_rag.shared import get_config

LOCAL_CONFIG = Path(__file__).parents[2] / "docker" / "config.local.yaml"
DEFAULT_SUFFIX = Constants.DEFAULT_SUFFIX.value

# Capabilities a backend may lack; a test needing one runs only on the
# backends that have it (``backends``).
RETRIEVERS = "retrievers"
# Aliases, index mappings and _meta, search pipelines (OpenSearch protocol).
INDEX_ADMIN = "index_admin"
# Vector documents keyed by (suffix, id), so one id lives in two suffixes.
SUFFIX_SCOPED_VECTOR_IDS = "suffix_scoped_vector_ids"

_FULL = frozenset({RETRIEVERS, INDEX_ADMIN, SUFFIX_SCOPED_VECTOR_IDS})
CAPABILITIES = {
    "fake": frozenset(),
    "sqlite": _FULL,
    "container": _FULL,
    "aws": _FULL,
}

# Vector kinds -> the ``indexing.opensearch`` prefix attribute of their index.
_VECTOR_PREFIXES = {
    "text_units": "text_units_index_prefix",
    "entities": "entities_index_prefix",
    "relationships": "relationships_index_prefix",
    "claims": "claims_index_prefix",
    "community_reports": "community_reports_index_prefix",
}
# Graph kinds -> the fake's collection.
_GRAPH_KINDS = {
    "graph.entities": "entities",
    "graph.relationships": "relationships",
    "graph.communities": "communities",
}


def local_store_config() -> Config:
    """``docker/config.local.yaml`` on the Gremlin and OpenSearch kinds.

    The ports follow ``LOCAL_STORES_GREMLIN_PORT`` and
    ``LOCAL_STORES_OPENSEARCH_PORT`` when set (a stack published on other
    ports); the returned config is a copy, safe to change.
    """
    config = get_config(str(LOCAL_CONFIG)).model_copy(deep=True)
    config.providers.graph.kind = "gremlin"
    config.providers.vector.kind = "opensearch"
    if port := os.getenv("LOCAL_STORES_GREMLIN_PORT"):
        config.aws.neptune.port = int(port)
    if port := os.getenv("LOCAL_STORES_OPENSEARCH_PORT"):
        config.aws.opensearch.port = int(port)
    return config


def _aws_config() -> Config:
    path = os.getenv("GRAPHRAG_TEST_CONFIG")
    config = (
        get_config(path).model_copy(deep=True)
        if path and Path(path).is_file()
        else Config()
    )
    if endpoint := os.getenv("NEPTUNE_ENDPOINT"):
        config.aws.neptune.endpoint = endpoint
    if endpoint := os.getenv("OPENSEARCH_ENDPOINT"):
        config.aws.opensearch.endpoint = endpoint
    if not (config.aws.neptune.endpoint and config.aws.opensearch.endpoint):
        pytest.skip("set NEPTUNE_ENDPOINT and OPENSEARCH_ENDPOINT (real AWS)")
    config.providers.graph.kind = "neptune"
    config.providers.vector.kind = "opensearch"
    return config


class StoreHarness:
    """A graph and a vector indexer (and their retrievers) on one backend.

    ``graph_retriever`` / ``vector_retriever`` build a fresh retriever (each
    query batch runs on its own event loop); they are ``None`` on a backend
    without retrievers. :meth:`derive` opens another harness on the same
    stores with a changed config, e.g. another ``additional_suffix``.
    """

    def __init__(
        self,
        kind: str,
        config: Config,
        *,
        embedding_factory: Any | None = None,
        parent: StoreHarness | None = None,
    ) -> None:
        self.kind = kind
        self.config = config
        self.capabilities = CAPABILITIES[kind]
        self.embedding_factory = embedding_factory or HashingEmbeddingFactory()
        self._root: StoreHarness = parent._root if parent else self
        if parent is None:
            # Unique per harness: compose namespaces, suffixes and pipelines.
            self.tag = f"cf{uuid.uuid4().hex[:8]}"
            self._suffixes: set[str] = {DEFAULT_SUFFIX}
            self._derived: list[StoreHarness] = []
        else:
            self.tag = parent.tag
            self._root._derived.append(self)
        self.graph_indexer: Any
        self.vector_indexer: Any
        self.graph_retriever: Callable[[], Any] | None = None
        self.vector_retriever: Callable[[], Any] | None = None
        if kind == "fake":
            self.graph_indexer = FakeGraphStore(config)
            self.vector_indexer = FakeVectorStore(config.indexing.opensearch)
            return
        self.providers = Providers(config, embedding_factory=self.embedding_factory)
        self.graph_indexer = self.providers.build_graph_indexer()
        self.vector_indexer = self.providers.build_vector_indexer()
        self.graph_retriever = lambda: self.providers.build_retriever(
            RetrieverRole.GRAPH
        )
        self.vector_retriever = lambda: self.providers.build_retriever(
            RetrieverRole.DOCUMENT
        )

    # --- setup ---------------------------------------------------------------

    def suffix(self, name: str) -> str:
        """A suffix unique to this harness (``<name>-<tag>``), cleared on
        teardown."""
        suffix = f"{name}-{self.tag}"
        self._root._suffixes.add(suffix)
        return suffix

    def derive(
        self,
        change: Callable[[Config], None] | None = None,
        *,
        embedding_factory: Any | None = None,
    ) -> StoreHarness:
        """Another harness on the same stores, its config changed by
        ``change``; torn down with this one. On ``fake`` a changed
        ``additional_suffix`` gets fresh stores (the fakes hold one
        namespace)."""
        config = self.config.model_copy(deep=True)
        if change is not None:
            change(config)
        derived = StoreHarness(
            self.kind,
            config,
            embedding_factory=embedding_factory or self.embedding_factory,
            parent=self,
        )
        if self.kind == "fake" and (
            config.indexing.additional_suffix == self.config.indexing.additional_suffix
        ):
            derived.graph_indexer = self.graph_indexer
            derived.vector_indexer = self.vector_indexer
        return derived

    def initialize(self) -> None:
        """Create the hybrid search pipeline (``OpenSearchIndexer.initialize``)."""
        assert self.vector_indexer.initialize()

    # --- reads ---------------------------------------------------------------

    def stored_ids(self, kind: str, suffix: str | None = None) -> set[str]:
        """Ids stored in ``suffix`` (default: the default suffix).

        ``kind`` is a vector index (``text_units``, ``entities``,
        ``relationships``, ``claims``, ``community_reports``) or a graph
        collection (``graph.entities``, ``graph.relationships``,
        ``graph.communities``).
        """
        scope = suffix or DEFAULT_SUFFIX
        if kind in _GRAPH_KINDS:
            return self._graph_ids(kind, scope)
        if self.kind == "fake":
            return {
                item_id
                for item_id, item in self.vector_indexer.data.get(kind, {}).items()
                if BaseIndexer.get_suffix(item) == scope
            }
        prefix = getattr(self.config.indexing.opensearch, _VECTOR_PREFIXES[kind])
        alias = self.vector_indexer._get_name(prefix, scope)
        client = self.vector_indexer.opensearch_client
        if not client.get_index_name_by_alias(alias):
            return set()
        response = client.search(
            index=alias,
            body={
                "size": 1000,
                "query": {"match_all": {}},
                "_source": {"excludes": ["*_embedding"]},
            },
        )
        return {hit["_id"] for hit in response["hits"]["hits"]}

    def _graph_ids(self, kind: str, suffix: str) -> set[str]:
        if self.kind == "fake":
            bucket = self.graph_indexer.data.get(_GRAPH_KINDS[kind], {})
            return {item_id for scope, item_id in bucket if scope == suffix}
        neptune = self.config.indexing.neptune
        entity_label = self.graph_indexer._get_label(
            neptune.entity_label_prefix, suffix
        )
        community_label = self.graph_indexer._get_label(
            neptune.community_label_prefix, suffix
        )
        if self.kind == "sqlite":
            table, label = {
                "graph.entities": ("vertices", entity_label),
                "graph.relationships": ("edges", entity_label),
                "graph.communities": ("vertices", community_label),
            }[kind]
            with self.graph_indexer.store.read("read ids") as connection:
                rows = connection.execute(
                    f"SELECT id FROM {table} WHERE label = ?", (label,)
                ).fetchall()
            return {row[0] for row in rows}
        from gremlin_python.process.graph_traversal import __

        g = self.graph_indexer.neptune_client.g
        if kind == "graph.relationships":
            traversal = (
                g.V()
                .hasLabel(entity_label)
                .outE()
                .not_(__.hasLabel("MemberOf"))
                .values("id")
            )
        else:
            label = entity_label if kind == "graph.entities" else community_label
            traversal = g.V().hasLabel(label).values("id")
        return {str(item_id) for item_id in traversal.toList()}

    def retrieve(
        self,
        role: RetrieverRole,
        *queries: SearchQuery,
        prepare: Callable[[Any], None] | None = None,
    ) -> list[list[RetrievalResult]]:
        """Run ``queries`` on a fresh retriever of ``role`` (first given to
        ``prepare``, if any), in order."""
        build = (
            self.graph_retriever
            if role is RetrieverRole.GRAPH
            else self.vector_retriever
        )
        assert build is not None

        async def run() -> list[list[RetrievalResult]]:
            retriever = build()
            if prepare is not None:
                prepare(retriever)
            try:
                return [await retriever.aretrieve(query) for query in queries]
            finally:
                await retriever.aclose()

        return asyncio.run(run())

    # --- teardown ------------------------------------------------------------

    def teardown(self) -> None:
        """Clear every suffix handed out, in every derived namespace, then
        close the stores."""
        harnesses = [self, *self._derived] if self is self._root else [self]
        suffixes = sorted(self._root._suffixes)
        shared = self.kind in ("container", "aws")
        errors: list[str] = []
        for harness in harnesses:
            if shared:
                if not harness.graph_indexer.clear(suffixes):
                    errors.append("graph clear failed")
                if not harness.vector_indexer.clear(suffixes):
                    errors.append("vector clear failed")
                harness._delete_pipeline()
            for store in (harness.graph_indexer, harness.vector_indexer):
                store.close()
        if errors:
            raise AssertionError(
                f"{self.kind} teardown left data behind: {', '.join(errors)}"
            )

    def _delete_pipeline(self) -> None:
        from opensearchpy.exceptions import NotFoundError

        name = self.config.indexing.opensearch.hybrid_search_pipeline_name
        try:
            self.vector_indexer.opensearch_client.client.search_pipeline.delete(id=name)
        except NotFoundError:
            pass


def _harness_config(kind: str, tmp_path: Path) -> Config:
    if kind == "fake":
        return Config()
    if kind == "sqlite":
        config = Config()
        config.providers.graph.kind = "sqlite"
        config.providers.graph.sqlite_path = str(tmp_path / "graph.sqlite3")
        config.providers.vector.kind = "sqlite"
        config.providers.vector.sqlite_path = str(tmp_path / "vector.sqlite3")
        return config
    if kind == "container":
        if os.getenv("LOCAL_STORES") != "1":
            pytest.skip("set LOCAL_STORES=1 with docker/compose.local.yaml running")
        config = local_store_config()
    else:
        config = _aws_config()
    tag = f"cf{uuid.uuid4().hex[:8]}"
    config.indexing.additional_suffix = tag
    config.indexing.opensearch.hybrid_search_pipeline_name = f"{tag}-pipeline"
    return config


BACKENDS = [
    pytest.param("fake", id="fake"),
    pytest.param("sqlite", id="sqlite"),
    pytest.param("container", id="container"),
    pytest.param("aws", id="aws", marks=pytest.mark.aws),
]


def backends(*capabilities: str) -> list[Any]:
    """The :data:`BACKENDS` that have every one of ``capabilities``."""
    return [
        param
        for param in BACKENDS
        if set(capabilities) <= CAPABILITIES[param.values[0]]
    ]


def open_harness(kind: str, tmp_path: Path) -> Iterator[StoreHarness]:
    """A :class:`StoreHarness` on ``kind``, torn down after the ``yield``.

    A suite parametrizes its fixture over :data:`BACKENDS`::

        @pytest.fixture(params=BACKENDS)
        def harness(request, tmp_path):
            yield from open_harness(request.param, tmp_path)
    """
    harness = StoreHarness(kind, _harness_config(kind, tmp_path))
    try:
        yield harness
    finally:
        harness.teardown()
