# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The production process pools (MinHash dedup, fuzzy entity matching)
start their workers from the forkserver, never by forking a process that
already runs threads: a lock another thread held stays held in a forked
child, which then hangs."""

from __future__ import annotations

import pickle
import threading
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from tests.fixtures import fork_probe
from unified_kg_rag.adapters.ingestion import loader as loader_module
from unified_kg_rag.adapters.ingestion.loader import DirectoryLoader
from unified_kg_rag.domain.ingestion import graph_resolver as resolver_module
from unified_kg_rag.domain.ingestion.base_resolver import FuzzyMatcher
from unified_kg_rag.domain.ingestion.graph_resolver import GraphResolver
from unified_kg_rag.domain.models import Config, Document, Entity, ResolutionMethod
from unified_kg_rag.shared.utils.concurrency import process_pool, shut_down_after

pytestmark = pytest.mark.unit


@pytest.fixture
def lock_held_by_another_thread() -> Iterator[None]:
    """A thread holds ``fork_probe.LOCK`` while the pools start, as an I/O
    or client-pool thread of a run may hold a lock (logging's, a connection
    pool's) at any moment."""
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with fork_probe.LOCK:
            held.set()
            release.wait()

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    held.wait()
    yield
    release.set()
    thread.join()


@pytest.mark.timeout(60)
def test_a_worker_never_inherits_a_lock_another_thread_holds(
    lock_held_by_another_thread: None,
) -> None:
    # A forked worker finds the lock held, by a thread it does not have, and
    # would wait on it forever.
    with shut_down_after(process_pool(1)) as pool:
        assert pool.submit(fork_probe.lock_is_free).result() is True


def _document(i: int, text: str) -> Document:
    return Document(
        document_id=f"d{i}",
        file_name=f"{i}.txt",
        file_path=f"/corpus/{i}.txt",
        page_content=text,
    )


def _dedup() -> list[str]:
    korean = "공급자는 구매자에게 물품을 납품하고 구매자는 대금을 지급한다. " * 20
    japanese = "東邦商事は倉庫を借りて物流を委託する。" * 20
    documents = [
        _document(0, korean),
        _document(1, korean),
        _document(2, japanese),
        _document(3, "The Vendor ships parts to the Buyer every month. " * 20),
    ]
    loader = DirectoryLoader("/")
    loader.max_workers = 2
    return [d.document_id for d in loader.deduplicate_documents(documents)]


def _fuzzy_resolve(*, use_process_pool: bool = True) -> list[str]:
    config = Config()
    config.processing.fuzzy_entity_resolution = True
    entities = [
        Entity(id="e1", name="Acme Corporation Holdings", text_unit_ids=["t1"]),
        Entity(id="e2", name="Acme Corporation Holding", text_unit_ids=["t2"]),
        Entity(id="e3", name="한빛상사 주식회사", text_unit_ids=["t3"]),
        Entity(id="e4", name="Globex Sensors", text_unit_ids=["t4"]),
    ]
    resolver = GraphResolver(config, max_workers=2, use_process_pool=use_process_pool)
    result, _ = resolver.resolve_graph(entities, [])
    return sorted(e.name for e in result["entities"])


@pytest.mark.parametrize(
    ("module", "run"),
    [(loader_module, _dedup), (resolver_module, _fuzzy_resolve)],
    ids=["dedup", "fuzzy_matching"],
)
def test_the_production_pools_start_from_the_forkserver(
    module: Any, run: Callable[[], object], monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[Any] = []

    def recording(*args: Any, **kwargs: Any) -> Any:
        pool = process_pool(*args, **kwargs)
        built.append(pool)
        return pool

    monkeypatch.setattr(module, "process_pool", recording)
    run()
    (pool,) = built
    assert pool._mp_context.get_start_method() == "forkserver"


@pytest.mark.timeout(60)
def test_dedup_in_worker_processes_with_a_lock_held(
    lock_held_by_another_thread: None,
) -> None:
    assert _dedup() == ["d0", "d2", "d3"]


@pytest.mark.timeout(60)
def test_fuzzy_matching_in_worker_processes_with_a_lock_held(
    lock_held_by_another_thread: None,
) -> None:
    names = _fuzzy_resolve()
    assert names == _fuzzy_resolve(use_process_pool=False)
    assert len(names) == 3


@pytest.mark.parametrize("method", list(ResolutionMethod))
def test_what_crosses_into_a_worker_pickles(method: ResolutionMethod) -> None:
    # The forkserver workers share nothing: the matcher the initializer
    # sends must survive a pickle round trip with its index.
    matcher = FuzzyMatcher(
        ["Acme Corporation Holdings", "한빛상사 주식회사", "東邦商事"],
        resolution_method=method,
    )
    # The test's own bytes: what multiprocessing pickles to a worker.
    copy = pickle.loads(pickle.dumps(matcher))  # nosec B301 - own round trip

    for query in ("Acme Corporation Holding", "한빛상사", "東邦商事株式会社"):
        assert copy.find_all_matches(query) == matcher.find_all_matches(query)
