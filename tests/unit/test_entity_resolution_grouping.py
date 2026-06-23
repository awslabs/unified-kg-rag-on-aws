# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Entity-resolution grouping must not lose same-name distinct entities.

Regression: _group_similar_entities keyed a dict on entity.name
({name: entity}), so two distinct entities sharing a surface name collapsed to
one (last-writer-wins) BEFORE resolution ran — silently dropping the others'
type/description/text_unit_ids. Grouping must carry every entity through.
"""

from __future__ import annotations

import pytest

from aws_graphrag.domain.ingestion.graph_resolver import EntityResolver
from aws_graphrag.domain.models import Config, Entity

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
