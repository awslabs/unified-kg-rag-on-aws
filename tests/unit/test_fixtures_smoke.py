# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Smoke tests confirming shared fixtures build valid domain objects."""

from __future__ import annotations

import pytest

from aws_graphrag.models import (
    Community,
    CommunityReport,
    Config,
    Entity,
    Relationship,
    TextUnit,
)

pytestmark = pytest.mark.unit


def test_config_fixture(config: Config) -> None:
    assert config.aws is not None
    assert config.search is not None


def test_entity_fixture(sample_entities: list[Entity]) -> None:
    assert len(sample_entities) == 3
    assert {e.id for e in sample_entities} == {"e1", "e2", "e3"}


def test_relationship_fixture(sample_relationships: list[Relationship]) -> None:
    assert all(r.source_id and r.target_id for r in sample_relationships)


def test_text_unit_fixture(sample_text_units: list[TextUnit]) -> None:
    assert all(t.text for t in sample_text_units)


def test_community_fixtures(
    sample_communities: list[Community],
    sample_community_reports: list[CommunityReport],
) -> None:
    assert sample_communities[0].entity_ids == ["e1", "e2", "e3"]
    assert sample_community_reports[0].community_id == "c1"
