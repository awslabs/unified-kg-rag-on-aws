# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Store-name prefixes that could name the same store are refused.

A store is ``<prefix>-<suffix>``: equal prefixes share every store, and a
prefix ``a-b`` names prefix ``a``'s store of the suffix ``b-<suffix>``.
"""

import pytest
from pydantic import ValidationError

from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.methodology import IndexArtifact
from unified_kg_rag.domain.retrieval.index_prefixes import index_artifact_of
from unified_kg_rag.domain.retrieval.store_names import graph_label, store_name

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "opensearch",
    [
        {"entities_index_prefix": "kg", "relationships_index_prefix": "kg"},
        {"text_units_index_prefix": "kg", "entities_index_prefix": "kg-entities"},
        {"extraction_records_index_prefix": "kg", "claims_index_prefix": "kg-c"},
    ],
)
def test_colliding_index_prefixes_are_refused(opensearch: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="can name the same store"):
        Config.model_validate({"indexing": {"opensearch": opensearch}})


def test_nested_prefixes_would_share_a_store() -> None:
    # Why the rule holds: the two names below are one index.
    assert store_name("kg", "entities-acme") == store_name("kg-entities", "acme")
    assert graph_label("Kg", "x-acme") == graph_label("kg-X", "acme")


@pytest.mark.parametrize(
    "neptune",
    [
        {"entity_label_prefix": "Node", "community_label_prefix": "node"},
        {"entity_label_prefix": "Kg", "community_label_prefix": "kg-Community"},
    ],
)
def test_colliding_label_prefixes_are_refused(neptune: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="can name the same store"):
        Config.model_validate({"indexing": {"neptune": neptune}})


def test_assignment_is_checked_too() -> None:
    config = Config()
    with pytest.raises(ValidationError, match="can name the same store"):
        config.indexing.opensearch.entities_index_prefix = (
            f"{config.indexing.opensearch.text_units_index_prefix}-e"
        )


def test_distinct_prefixes_sharing_a_start_are_accepted() -> None:
    config = Config.model_validate(
        {
            "indexing": {
                "opensearch": {
                    "text_units_index_prefix": "kg-a",
                    "entities_index_prefix": "kg-ab",
                },
                "neptune": {
                    "entity_label_prefix": "Kg",
                    "community_label_prefix": "Kgc",
                },
            }
        }
    )
    assert (
        index_artifact_of(config, "kg-ab-default-20261112101010123")
        is IndexArtifact.ENTITIES
    )
    assert index_artifact_of(config, "kg-a-default") is IndexArtifact.TEXT_UNITS
    assert index_artifact_of(config, "kg-abc-default") is None
