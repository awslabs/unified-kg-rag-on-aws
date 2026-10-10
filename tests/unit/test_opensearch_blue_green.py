# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free tests for OpenSearchIndexer's blue/green index-write + alias swap.

``_index_item_type`` is the core full-index write orchestration — create a new
timestamped index, embed + prepare docs, bulk-index, then atomically swap the
alias onto the new index and reap stale indices; on failure it rolls back by
deleting the half-written index. A fake cluster that keeps index names and
alias names apart (as OpenSearch does) drives the real method, so the
swap/cleanup/rollback behavior is verified without AWS.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from fnmatch import fnmatchcase

import pytest

from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.domain.models import Config, Entity

pytestmark = pytest.mark.unit

_ALIAS = "graphrag-entities-default"


class _FakeCluster:
    """Indices and their aliases, with OpenSearch's name-matching semantics.

    ``get_indices_by_alias`` matches *alias* names (``GET _alias/<name>``) and
    ``get_aliases_by_index`` matches *index* names (``GET <index>/_alias``), so
    looking up timestamped index names through the alias API finds nothing,
    exactly like the real cluster.
    """

    def __init__(self, existing: dict[str, set[str]] | None = None) -> None:
        self.indices: dict[str, set[str]] = {
            name: set(aliases) for name, aliases in (existing or {}).items()
        }
        self.calls: list[tuple] = []
        self.created: list[str] = []
        self.bulk_should_fail = False

    def _match(self, pattern: str) -> list[str]:
        return [n for n in self.indices if fnmatchcase(n, pattern)]

    def create_index(self, index_name, mapping):
        self.calls.append(("create_index", index_name))
        self.created.append(index_name)
        self.indices.setdefault(index_name, set())

    def update_alias(self, alias_name, index_name, remove_pattern=None):
        self.calls.append(("update_alias", alias_name, index_name, remove_pattern))
        if remove_pattern:
            for name in self._match(remove_pattern):
                self.indices[name].discard(alias_name)
        self.indices[index_name].add(alias_name)

    def get_indices_by_alias(self, alias_name):
        self.calls.append(("get_indices_by_alias", alias_name))
        return [
            name
            for name, aliases in self.indices.items()
            if any(fnmatchcase(a, alias_name) for a in aliases)
        ]

    def get_aliases_by_index(self, index_pattern):
        self.calls.append(("get_aliases_by_index", index_pattern))
        return {name: sorted(self.indices[name]) for name in self._match(index_pattern)}

    def delete_alias(self, index_names, alias_names):
        self.calls.append(("delete_alias", tuple(index_names), tuple(alias_names)))
        targets = [
            aliases
            for name, aliases in self.indices.items()
            if index_names == "_all" or name in index_names
        ]
        # OpenSearch answers 404 when none of the named aliases exists.
        if not any(alias in aliases for aliases in targets for alias in alias_names):
            raise RuntimeError("aliases_not_found_exception")
        for aliases in targets:
            aliases.difference_update(alias_names)

    def delete_indices(self, indices):
        self.calls.append(("delete_indices", tuple(indices)))
        for pattern in indices:
            for name in self._match(pattern):
                del self.indices[name]

    def bulk_index(self, index_name, documents, **kwargs):
        self.calls.append(("bulk_index", index_name, len(documents)))
        if self.bulk_should_fail:
            raise RuntimeError("bulk failed")
        return {"errors": False, "items": []}


@pytest.fixture(autouse=True)
def _ticking_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Advance the index-name timestamp by one second per call.

    Index names carry a second-resolution timestamp, so back-to-back runs in
    a test would otherwise reuse one name.
    """
    start = datetime(2026, 1, 1, 0, 0, 0)
    ticks = iter(range(10_000))

    class _Clock:
        @staticmethod
        def now() -> datetime:
            return start + timedelta(seconds=next(ticks))

    monkeypatch.setattr("unified_kg_rag.shared.utils.store_names.datetime", _Clock)


def _indexer(config: Config, client: _FakeCluster) -> OpenSearchIndexer:
    inst = OpenSearchIndexer.__new__(OpenSearchIndexer)
    inst.config = config
    inst.opensearch_config = config.indexing.opensearch
    inst.analyzer = "standard"
    inst.target_language = config.processing.translation.target_language.value
    inst._embedding_dimension = 1024
    inst.opensearch_client = client
    # Avoid Bedrock: deterministic embeddings + flush no-op.
    inst._batch_embed = lambda texts, batch_size=50: [[0.1] for _ in texts]
    inst._flush_embedding_cache = lambda: None
    return inst


def _op_names(calls: list[tuple]) -> list[str]:
    return [c[0] for c in calls]


def _entities(suffix: str | None = None) -> list[Entity]:
    attributes = {"index": suffix} if suffix else {}
    return [Entity(id="e1", name="Alice", attributes=attributes)]


def test_full_index_creates_then_swaps_alias_then_cleans_stale(config) -> None:
    # The previous build's index carries the alias; after a successful write
    # the new index is created, the alias swapped onto it, and the old reaped.
    old = f"{_ALIAS}-20251231000000"
    client = _FakeCluster({old: {_ALIAS}})
    indexer = _indexer(config, client)

    stats = indexer.index_entities(_entities())

    ops = _op_names(client.calls)
    assert ops.index("create_index") < ops.index("bulk_index")
    assert ops.index("bulk_index") < ops.index("update_alias")
    assert ops.index("update_alias") < ops.index("delete_indices")

    new = client.created[0]
    swap = next(c for c in client.calls if c[0] == "update_alias")
    assert swap[1:3] == (_ALIAS, new)
    assert client.indices == {new: {_ALIAS}}
    assert stats.successful_items == 1


def test_repeated_full_runs_leave_exactly_one_index(config) -> None:
    # Regression: stale indices were looked up through the alias-name API with
    # an index-name pattern, matched nothing, and leaked one index per run.
    client = _FakeCluster()
    indexer = _indexer(config, client)

    for _ in range(3):
        assert indexer.index_entities(_entities()).successful_items == 1

    assert client.indices == {client.created[-1]: {_ALIAS}}


def test_cleanup_leaves_other_suffixes_untouched(config) -> None:
    # ``<alias>-*`` also matches suffix ``default-2``'s indices by name; only
    # this alias's own ``<alias>-<timestamp>`` indices may be reaped.
    sibling_live = f"{_ALIAS}-2-20251231000000"
    sibling_stale = f"{_ALIAS}-2-20251230000000"
    unrelated = f"{_ALIAS}-manual-backup"
    client = _FakeCluster(
        {
            sibling_live: {f"{_ALIAS}-2"},
            sibling_stale: set(),
            unrelated: set(),
        }
    )
    indexer = _indexer(config, client)

    indexer.index_entities(_entities())
    indexer.index_entities(_entities())

    assert client.indices == {
        sibling_live: {f"{_ALIAS}-2"},
        sibling_stale: set(),
        unrelated: set(),
        client.created[-1]: {_ALIAS},
    }


def test_cleanup_keeps_a_newer_concurrent_build(config) -> None:
    newer = f"{_ALIAS}-20991231000000"
    client = _FakeCluster({newer: set()})
    indexer = _indexer(config, client)

    indexer.index_entities(_entities())

    assert newer in client.indices


def test_first_write_has_no_stale_index_to_clean(config) -> None:
    client = _FakeCluster()
    indexer = _indexer(config, client)
    indexer.index_entities(_entities())
    assert not any(c[0] == "delete_indices" for c in client.calls)
    assert client.indices == {client.created[0]: {_ALIAS}}


def test_bulk_failure_does_not_swap_alias_or_delete_live_index(config) -> None:
    # A bulk-index failure is absorbed by _perform_indexing (returns failed
    # stats, does not raise), so the alias is NOT swapped onto an index with
    # zero successful docs, the live index is kept, and the unused new index
    # is dropped instead of leaking.
    live = f"{_ALIAS}-20251231000000"
    older = f"{_ALIAS}-20251230000000"
    client = _FakeCluster({live: {_ALIAS}, older: set()})
    client.bulk_should_fail = True
    indexer = _indexer(config, client)

    stats = indexer.index_entities(_entities())

    assert "update_alias" not in _op_names(client.calls)
    assert client.indices == {live: {_ALIAS}, older: set()}
    assert stats.successful_items == 0
    assert stats.failed_items >= 1


def test_cleanup_failure_keeps_the_new_live_index(config) -> None:
    # The swap already succeeded; a failing stale-index delete must not route
    # into the rollback that would delete the now-live index.
    old = f"{_ALIAS}-20251231000000"
    client = _FakeCluster({old: {_ALIAS}})

    def _boom_delete(indices):
        raise RuntimeError("delete failed")

    client.delete_indices = _boom_delete
    indexer = _indexer(config, client)

    stats = indexer.index_entities(_entities())

    new = client.created[0]
    assert client.indices[new] == {_ALIAS}
    assert stats.successful_items == 1
    assert stats.failed_items == 0


def test_create_index_failure_rolls_back(config) -> None:
    # An exception from the write machinery (here: create_index) triggers the
    # except branch, which deletes the half-written index so no orphan is left.
    live = f"{_ALIAS}-20251231000000"
    client = _FakeCluster({live: {_ALIAS}})

    def _boom_create(index_name, mapping):
        client.created.append(index_name)
        client.indices[index_name] = set()  # partially created before failing
        client.calls.append(("create_index", index_name))
        raise RuntimeError("create failed")

    client.create_index = _boom_create
    indexer = _indexer(config, client)

    stats = indexer.index_entities(_entities())

    assert client.indices == {live: {_ALIAS}}
    assert "update_alias" not in _op_names(client.calls)
    assert stats.failed_items >= 1


def test_clear_of_a_namespace_never_indexed_succeeds(config) -> None:
    # No alias of the namespace exists: deleting them would be a 404 that
    # failed the reset of a namespace before its first run.
    client = _FakeCluster()
    indexer = _indexer(config, client)

    assert indexer.clear(["default"])
    assert "delete_alias" not in _op_names(client.calls)


def test_clear_of_some_stores_deletes_their_indices_and_aliases(config) -> None:
    # Only the entities store was written; the other stores have no alias.
    # An alias left on an index outside the naming scheme is deleted too.
    live = f"{_ALIAS}-20260101000000"
    client = _FakeCluster({live: {_ALIAS}, "manual-index": {_ALIAS}})
    indexer = _indexer(config, client)

    assert indexer.clear(["default"])

    assert client.indices == {"manual-index": set()}
