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
- ``container+custom-ids`` / ``aws+custom-ids`` (graph suites only,
  :data:`GRAPH_BACKENDS`): the same stores with
  ``indexing.neptune.vertex_ids: custom``.
- one more backend, a custom one, named by ``UKR_TEST_BACKEND``
  (``module:attribute`` of an :class:`ExtraBackend`): its stores come from
  the ``Providers`` it builds, so a backend author runs the suites on it
  without editing this file.
- ``aws``: Neptune and OpenSearch Service (``pytest.mark.aws``). The config is
  ``UKR_TEST_CONFIG`` if set, else the defaults; ``UKR__AWS__NEPTUNE__ENDPOINT`` and
  ``UKR__AWS__OPENSEARCH__ENDPOINT`` override the endpoints. Skipped without both
  endpoints.

Every harness of a shared server writes under its own
``indexing.additional_suffix``, and clears every suffix it handed out (``StoreHarness.suffix``) on teardown, so a
run leaves nothing behind. Embeddings come from ``HashingEmbeddingFactory``: no
model is called on any backend.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.config import get_config
from unified_kg_rag.domain.models import (
    Config,
    Constants,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
)
from unified_kg_rag.ports.indexer import BaseIndexer, VectorIndexer, _implements

LOCAL_CONFIG = Path(__file__).parents[2] / "docker" / "config.local.yaml"
# The variable that runs the local_stores tests (tests/conftest.py skips
# them without it).
LOCAL_STORES_ENV = "LOCAL_STORES"
DEFAULT_SUFFIX = Constants.DEFAULT_SUFFIX.value

# Capabilities a backend may lack; a test needing one runs only on the
# backends that have it (``backends``).
RETRIEVERS = "retrievers"
# Aliases, index mappings and _meta (OpenSearch protocol).
INDEX_ADMIN = "index_admin"
# Vector documents keyed by (suffix, id), so one id lives in two suffixes.
SUFFIX_SCOPED_VECTOR_IDS = "suffix_scoped_vector_ids"
# INDEX_ADMIN's tests also read the store's OpenSearch-protocol internals
# (``opensearch_client``, ``_get_name``), as ``OpenSearchIndexer`` has them on
# either engine: a custom backend declares it only when it is one.

# The backends this package ships; any other is an ``ExtraBackend``.
_PACKAGED = frozenset({"fake", "sqlite", "container", "aws"})

_FULL = frozenset({RETRIEVERS, INDEX_ADMIN, SUFFIX_SCOPED_VECTOR_IDS})
CAPABILITIES = {
    "fake": frozenset(),
    "sqlite": _FULL,
    "container": _FULL,
    "aws": _FULL,
}


@dataclass(frozen=True)
class ExtraBackend:
    """A custom backend the suites run on too: ``UKR_TEST_BACKEND`` names
    one (``my_package.conformance:BACKEND``).

    ``providers(config, embedding_factory)`` builds the ``Providers`` its
    stores and retrievers come from (``vector_indexer_factory``,
    ``graph_indexer_factory``, ``retriever_builders``), with
    ``embedding_factory``'s embeddings; ``configure(config, tmp_path)``
    sets what it reads before that (a path under ``tmp_path``).
    ``capabilities`` are those it has of :data:`CAPABILITIES`' (a test
    needing another skips it; ``RETRIEVERS`` builds the harness's
    retrievers)."""

    name: str
    providers: Callable[[Config, Any], Providers]
    capabilities: frozenset[str] = frozenset()
    configure: Callable[[Config, Path], None] | None = None


def extra_backend(spec: str | None) -> ExtraBackend | None:
    """The :class:`ExtraBackend` ``spec`` (``module:attribute``) names."""
    if not spec:
        return None
    module, _, attribute = spec.partition(":")
    backend = getattr(importlib.import_module(module), attribute)
    if not isinstance(backend, ExtraBackend):
        raise TypeError(f"UKR_TEST_BACKEND={spec} is not an ExtraBackend")
    if backend.name.partition("+")[0] in CAPABILITIES:
        raise ValueError(f"UKR_TEST_BACKEND={spec}: '{backend.name}' is taken")
    return backend


EXTRA_BACKEND = extra_backend(os.getenv("UKR_TEST_BACKEND"))
if EXTRA_BACKEND is not None:
    CAPABILITIES[EXTRA_BACKEND.name] = EXTRA_BACKEND.capabilities

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
    # The suites embed with HashingEmbeddingFactory, not the preset's model:
    # the dimension is the one that factory reports, not bge-m3's 1024.
    config.models.embedding.dimension = None
    if port := os.getenv("LOCAL_STORES_GREMLIN_PORT"):
        config.aws.neptune.port = int(port)
    if port := os.getenv("LOCAL_STORES_OPENSEARCH_PORT"):
        config.aws.opensearch.port = int(port)
    return config


def _aws_config() -> Config:
    path = os.getenv("UKR_TEST_CONFIG")
    config = (
        get_config(path).model_copy(deep=True)
        if path and Path(path).is_file()
        else Config()
    )
    if endpoint := os.getenv("UKR__AWS__NEPTUNE__ENDPOINT"):
        config.aws.neptune.endpoint = endpoint
    if endpoint := os.getenv("UKR__AWS__OPENSEARCH__ENDPOINT"):
        config.aws.opensearch.endpoint = endpoint
    if not (config.aws.neptune.endpoint and config.aws.opensearch.endpoint):
        pytest.skip(
            "set UKR__AWS__NEPTUNE__ENDPOINT and UKR__AWS__OPENSEARCH__ENDPOINT (real AWS)"
        )
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
        extra = EXTRA_BACKEND is not None and kind == EXTRA_BACKEND.name
        self.providers = (
            EXTRA_BACKEND.providers(config, self.embedding_factory)
            if EXTRA_BACKEND is not None and extra
            else Providers(config, embedding_factory=self.embedding_factory)
        )
        self.graph_indexer = self.providers.build_graph_indexer()
        self.vector_indexer = self.providers.build_vector_indexer()
        if extra and RETRIEVERS not in self.capabilities:
            return
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
        """Prepare the vector store for writes (the indexer's ``initialize``)."""
        self.vector_indexer.initialize()

    def implements(self, method: str) -> bool:
        """Whether the vector store answers ``method``'s contract: a
        packaged backend always does; an :class:`ExtraBackend`'s when it
        overrides the method, or leaves a default that derives its answer
        from methods it does override (``holds_any_item`` from
        ``read_schema_annotations``: a working default, not a "cannot")."""
        if self.kind in _PACKAGED:
            return True
        if _implements(self.vector_indexer, VectorIndexer, method):
            return True
        sources = _DERIVED_DEFAULTS.get(method)
        return sources is not None and all(self.implements(m) for m in sources)

    def needs(self, *methods: str) -> None:
        """Skip on an :class:`ExtraBackend` whose vector store leaves any of
        ``methods`` (the ones the test reads) at its port's "cannot"
        default: a choice the port allows (a capability the backend lacks),
        not a broken contract, so the test skips naming each method by its
        port. A method the store overrides, or a working default
        (:meth:`implements`), runs the test. The packaged backends implement
        every method a suite calls: on them nothing is skipped and the test
        states the contract."""
        left = [
            f"{_owner(method)}.{method}"
            for method in methods
            if not self.implements(method)
        ]
        if left:
            defaults = "its default" if len(left) == 1 else "their defaults"
            pytest.skip(f"{self.kind} leaves {', '.join(left)} at {defaults}")

    def needs_port(self, port: type) -> None:
        """Skip on an :class:`ExtraBackend` whose vector store overrides
        none of ``port``'s public methods (a concern port ``VectorIndexer``
        inherits), naming the port. One that overrides any runs the suite:
        each test then :meth:`needs` the methods it reads, so a method left
        at its default skips only those tests, never the port's others."""
        methods = [
            name
            for name, value in vars(port).items()
            if callable(value) and not name.startswith("_")
        ]
        if self.kind not in _PACKAGED and not any(
            _implements(self.vector_indexer, VectorIndexer, m) for m in methods
        ):
            pytest.skip(f"{self.kind} leaves {port.__name__} at its defaults")

    # --- reads ---------------------------------------------------------------

    def stored_ids(self, kind: str, suffix: str | None = None) -> set[str]:
        """Ids stored in ``suffix`` (default: the default suffix).

        ``kind`` is a vector index (``text_units``, ``entities``,
        ``relationships``, ``claims``, ``community_reports``) or a graph
        collection (``graph.entities``, ``graph.relationships``,
        ``graph.communities``).
        """
        scope = suffix or DEFAULT_SUFFIX
        if self.kind not in _PACKAGED:
            return self._listed_ids(kind, scope)
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

    def _listed_ids(self, kind: str, suffix: str) -> set[str]:
        """An :class:`ExtraBackend`'s ids, through its vector store's
        ``StoreInspection.scan_items`` (skipped where it cannot page them:
        the graph kinds, or a store left at the "cannot" default)."""
        pages = (
            None
            if kind in _GRAPH_KINDS
            else self.vector_indexer.scan_items(kind, suffix)
        )
        if pages is None:
            pytest.skip(f"{self.kind} cannot list its {kind} ids")
        return {str(item["id"]) for page in pages for item in page}

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

        g = self.graph_indexer.neptune_client.g
        if kind == "graph.relationships":
            traversal = g.V().hasLabel(entity_label).outE().values("id")
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
                vector = harness.vector_indexer
                for name, clear in (
                    ("graph", harness.graph_indexer.clear),
                    ("vector", vector.clear),
                    ("extraction records", partial(clear_extraction_records, vector)),
                ):
                    try:
                        clear(suffixes)
                    except Exception as e:  # noqa: BLE001 - reported below
                        errors.append(f"{name} clear failed ({e})")
            for store in (harness.graph_indexer, harness.vector_indexer):
                store.close()
        if errors:
            raise AssertionError(
                f"{self.kind} teardown left data behind: {', '.join(errors)}"
            )


# Port defaults that are a working derivation, not a "cannot": the method
# answers its contract whenever the store implements the ones named.
_DERIVED_DEFAULTS: dict[str, tuple[str, ...]] = {
    "holds_any_item": ("read_schema_annotations",),
    "stored_extraction_record_keys": ("read_extraction_records",),
}


def _owner(method: str) -> str:
    """The port that declares ``method``: ``VectorIndexer`` or one of the
    concern ports it inherits."""
    return next(port.__name__ for port in VectorIndexer.__mro__ if method in vars(port))


def clear_extraction_records(vector_indexer: Any, suffixes: list[str]) -> None:
    """Delete the extraction-records indices of ``suffixes`` (an
    ``OpenSearchIndexer`` on a shared server).

    ``clear`` leaves them (a reset keeps the records), so every test that
    ingested left one behind and a long-lived stack reached the cluster's
    shard limit.
    """
    client = vector_indexer.opensearch_client
    prefix = vector_indexer.opensearch_config.extraction_records_index_prefix
    names = sorted(
        {
            name
            for alias in (vector_indexer._get_name(prefix, s) for s in suffixes)
            for name in client.get_aliases_by_index(f"{alias}-*")
            if vector_indexer._index_timestamp(alias, name) is not None
        }
    )
    if names:
        client.delete_indices(names)


def _harness_config(kind: str, tmp_path: Path) -> Config:
    if kind == "fake":
        return Config()
    if EXTRA_BACKEND is not None and kind == EXTRA_BACKEND.name:
        config = Config()
        if EXTRA_BACKEND.configure is not None:
            EXTRA_BACKEND.configure(config, tmp_path)
        return config
    if kind == "sqlite":
        config = Config()
        config.providers.graph.kind = "sqlite"
        config.providers.graph.sqlite_path = str(tmp_path / "graph.sqlite3")
        config.providers.vector.kind = "sqlite"
        config.providers.vector.sqlite_path = str(tmp_path / "vector.sqlite3")
        return config
    if kind == "container":
        # A container test carries the local_stores marker, which skips it
        # unless LOCAL_STORES=1 (tests/conftest.py); one without it fails
        # here rather than skipping where CI never runs it.
        if os.getenv(LOCAL_STORES_ENV) != "1":
            pytest.fail(
                "a container store test needs the local_stores marker "
                f"(it runs with {LOCAL_STORES_ENV}=1 only)"
            )
        config = local_store_config()
    else:
        config = _aws_config()
    tag = f"cf{uuid.uuid4().hex[:8]}"
    config.indexing.additional_suffix = tag
    return config


BACKENDS = [
    pytest.param("fake", id="fake"),
    pytest.param("sqlite", id="sqlite"),
    pytest.param("container", id="container", marks=pytest.mark.local_stores),
    pytest.param("aws", id="aws", marks=pytest.mark.aws),
    *(
        [pytest.param(EXTRA_BACKEND.name, id=EXTRA_BACKEND.name)]
        if EXTRA_BACKEND is not None
        else []
    ),
]

# A Neptune-only variant of a backend: ``<kind>+<variant>``.
CUSTOM_IDS = "custom-ids"
_VARIANTS: dict[str, Callable[[Config], None]] = {
    CUSTOM_IDS: lambda config: setattr(config.indexing.neptune, "vertex_ids", "custom"),
}

# BACKENDS plus the Gremlin stores in custom vertex-id mode, for the suites
# of graph contracts.
GRAPH_BACKENDS = [
    *BACKENDS,
    pytest.param(
        f"container+{CUSTOM_IDS}",
        id=f"container-{CUSTOM_IDS}",
        marks=pytest.mark.local_stores,
    ),
    pytest.param(f"aws+{CUSTOM_IDS}", id=f"aws-{CUSTOM_IDS}", marks=pytest.mark.aws),
]


def backends(*capabilities: str, graph: bool = False) -> list[Any]:
    """The :data:`BACKENDS` (with ``graph``, :data:`GRAPH_BACKENDS`) that
    have every one of ``capabilities``."""
    return [
        param
        for param in (GRAPH_BACKENDS if graph else BACKENDS)
        if set(capabilities) <= CAPABILITIES[param.values[0].partition("+")[0]]
    ]


def open_harness(kind: str, tmp_path: Path) -> Iterator[StoreHarness]:
    """A :class:`StoreHarness` on ``kind``, torn down after the ``yield``.

    A suite parametrizes its fixture over :data:`BACKENDS`::

        @pytest.fixture(params=BACKENDS)
        def harness(request, tmp_path):
            yield from open_harness(request.param, tmp_path)
    """
    base, _, variant = kind.partition("+")
    config = _harness_config(base, tmp_path)
    if variant:
        _VARIANTS[variant](config)
    harness = StoreHarness(base, config)
    try:
        yield harness
    finally:
        harness.teardown()
