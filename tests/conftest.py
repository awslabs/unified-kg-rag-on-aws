# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Shared pytest fixtures.

Tests run AWS-free by default: the ``aws`` marker gates the few that need real
services, and port-based fakes (``tests/fixtures/fakes``) stand in for Neptune /
OpenSearch / DynamoDB. ``moto`` provides mocked AWS APIs where an adapter must
be exercised against a boto3 surface.
"""
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
from tests.fixtures.fakes.doc_status import FakeDocStatusStore


@pytest.fixture
def config() -> Config:
    """A default ``Config`` (all nested defaults, no external services)."""
    return Config()


@pytest.fixture
def fake_doc_status() -> FakeDocStatusStore:
    """An empty in-memory DocStatusPort implementation."""
    return FakeDocStatusStore()


@pytest.fixture
def sample_entities() -> list[Entity]:
    return [
        Entity(id="e1", name="Alice", type="PERSON", description="A researcher"),
        Entity(id="e2", name="Acme Corp", type="ORG", description="A company"),
        Entity(id="e3", name="Seattle", type="GPE", description="A city"),
    ]


@pytest.fixture
def sample_relationships() -> list[Relationship]:
    return [
        Relationship(
            id="r1",
            source_id="e1",
            target_id="e2",
            source_name="Alice",
            target_name="Acme Corp",
            description="Alice works at Acme Corp",
            weight=1.0,
        ),
        Relationship(
            id="r2",
            source_id="e2",
            target_id="e3",
            source_name="Acme Corp",
            target_name="Seattle",
            description="Acme Corp is based in Seattle",
            weight=0.8,
        ),
    ]


@pytest.fixture
def sample_text_units() -> list[TextUnit]:
    return [
        TextUnit(id="t1", text="Alice works at Acme Corp.", entity_ids=["e1", "e2"]),
        TextUnit(
            id="t2", text="Acme Corp is based in Seattle.", entity_ids=["e2", "e3"]
        ),
    ]


@pytest.fixture
def sample_communities() -> list[Community]:
    return [
        Community(
            id="c1",
            name="Acme cluster",
            level="0",
            parent="",
            children=[],
            entity_ids=["e1", "e2", "e3"],
            text_unit_ids=["t1", "t2"],
        ),
    ]


@pytest.fixture
def sample_community_reports() -> list[CommunityReport]:
    return [
        CommunityReport(
            id="cr1",
            community_id="c1",
            name="Acme cluster report",
            summary="Alice, Acme Corp, and Seattle form a cluster.",
            full_content="Detailed report about the Acme cluster.",
        ),
    ]
