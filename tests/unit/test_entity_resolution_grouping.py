# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entity-resolution grouping must not lose same-name distinct entities.

Regression: _group_similar_entities keyed a dict on entity.name
({name: entity}), so two distinct entities sharing a surface name collapsed to
one (last-writer-wins) BEFORE resolution ran — silently dropping the others'
type/description/text_unit_ids. Grouping must carry every entity through.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.ingestion.graph_resolver import EntityResolver
from unified_kg_rag.domain.models import Config, Entity

pytestmark = pytest.mark.unit


def _resolver() -> EntityResolver:
    # Threads (not processes) so the test runs in-process; small input.
    return EntityResolver(Config(), max_workers=2, use_process_pool=False)


def test_same_name_entities_are_not_dropped() -> None:
    resolver = _resolver()
    resolver.show_progress = False
    entities = [
        Entity(id="e1", name="Mercury", type="planet", text_unit_ids=["t1"]),
        Entity(id="e2", name="Mercury", type="element", text_unit_ids=["t2"]),
        Entity(id="e3", name="Venus", type="planet", text_unit_ids=["t3"]),
    ]
    groups = resolver._group_similar_entities(entities)
    grouped_ids = {e.id for group in groups for e in group}
    # All three survive grouping — none silently dropped by name collision.
    assert grouped_ids == {"e1", "e2", "e3"}


def test_distinct_names_each_present() -> None:
    resolver = _resolver()
    resolver.show_progress = False
    entities = [
        Entity(id="a", name="Alpha", type="x"),
        Entity(id="b", name="Beta", type="y"),
    ]
    groups = resolver._group_similar_entities(entities)
    assert {e.id for g in groups for e in g} == {"a", "b"}


def test_matcher_passed_to_pool_via_initializer_not_per_task(mocker) -> None:
    # Regression (perf): the FuzzyMatcher (tens of MB at scale) must be shipped
    # to workers ONCE via the pool initializer, not pickled per submitted task.
    # Assert the executor is constructed with initializer/initargs and that
    # submit() is called with ONLY the entity name (no matcher argument).
    import unified_kg_rag.domain.ingestion.graph_resolver as gr

    resolver = _resolver()
    resolver.show_progress = False

    captured: dict = {}

    class FakeExecutor:
        def __init__(self, max_workers=None, initializer=None, initargs=None):
            captured["initializer"] = initializer
            captured["initargs"] = initargs

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, *args):
            captured.setdefault("submit_args", []).append(args)

            class _F:
                def result(self_inner):
                    return []

            return _F()

    mocker.patch.object(gr, "ThreadPoolExecutor", FakeExecutor)
    mocker.patch.object(gr, "as_completed", lambda futs: list(futs))

    resolver._group_similar_entities(
        [Entity(id="a", name="Alpha"), Entity(id="b", name="Beta")]
    )

    # The matcher is handed to the pool initializer exactly once...
    assert captured["initializer"] is gr._init_worker_fuzzy_matcher
    assert captured["initargs"] is not None and len(captured["initargs"]) == 1
    # ...and each submit carries only the (tiny) name, never the matcher.
    assert all(
        len(args) == 1 and isinstance(args[0], str) for args in captured["submit_args"]
    )
