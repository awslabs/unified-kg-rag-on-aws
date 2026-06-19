# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for lineage-based entity/relationship relevance (Tier-3 refactor).

Relevance now uses the authoritative ``text_unit_ids`` recorded during
extraction instead of token-overlap heuristics — exact and language-agnostic.
"""

from __future__ import annotations

import pytest

from aws_graphrag.ingestion.base_processor import (
    check_entity_relevance_task,
    check_relationship_relevance_task,
)
from aws_graphrag.models import Entity, Relationship

pytestmark = pytest.mark.unit


class TestEntityRelevance:
    def test_relevant_when_extracted_from_unit(self) -> None:
        entity = Entity(id="e1", name="Alice", text_unit_ids=["t1", "t2"])
        assert check_entity_relevance_task(entity, "t1")[1] is True

    def test_not_relevant_for_other_unit(self) -> None:
        entity = Entity(id="e1", name="Alice", text_unit_ids=["t1"])
        assert check_entity_relevance_task(entity, "t9")[1] is False

    def test_no_lineage_is_not_relevant(self) -> None:
        entity = Entity(id="e1", name="Alice", text_unit_ids=None)
        assert check_entity_relevance_task(entity, "t1")[1] is False

    def test_language_agnostic(self) -> None:
        # A non-Latin name that token-overlap heuristics would mishandle is
        # still correctly matched purely by lineage.
        entity = Entity(id="e1", name="앨리스", text_unit_ids=["t1"])
        assert check_entity_relevance_task(entity, "t1")[1] is True


class TestRelationshipRelevance:
    def test_relevant_when_extracted_from_unit(self) -> None:
        rel = Relationship(
            id="r1", source_id="e1", target_id="e2", text_unit_ids=["t1"]
        )
        assert check_relationship_relevance_task(rel, "t1")[1] is True

    def test_not_relevant_for_other_unit(self) -> None:
        rel = Relationship(
            id="r1", source_id="e1", target_id="e2", text_unit_ids=["t1"]
        )
        assert check_relationship_relevance_task(rel, "t9")[1] is False
