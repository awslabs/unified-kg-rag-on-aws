# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Changing a config: an assignment validates its field, and
``Config.updated`` validates the whole config as loading does."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from unified_kg_rag.application.cli.console.jobs import ingest_config
from unified_kg_rag.domain.models import Config, Methodology
from unified_kg_rag.domain.models.config import ConfigSection, SearchConfig
from unified_kg_rag.shared import ConfigurationError, InvalidArgumentError

pytestmark = pytest.mark.unit


def _lightrag_without_communities() -> Config:
    return Config.model_validate(
        {
            "graph": {"community_detection": {"enabled": False}},
            "indexing": {"methodologies": ["lightrag"]},
        }
    )


def test_every_section_validates_an_assignment() -> None:
    config = Config()
    with pytest.raises(ValidationError):
        config.search.local_search.entity_frequency_threshold = -5
    with pytest.raises(ValidationError):
        config.search.global_search.evidence_tokens = "many"  # type: ignore[assignment]


def test_every_config_section_shares_the_base() -> None:
    def sections(model: type) -> set[type]:
        found: set[type] = set()
        for field in model.model_fields.values():
            annotation = field.annotation
            if isinstance(annotation, type) and issubclass(annotation, ConfigSection):
                found |= {annotation, *sections(annotation)}
        return found

    assert len(sections(Config)) > 40
    for name, field in Config.model_fields.items():
        if isinstance(field.annotation, type) and issubclass(
            field.annotation, BaseModel
        ):
            assert issubclass(field.annotation, ConfigSection), name


def test_the_hybrid_weights_validate_on_assignment() -> None:
    # The weights are relative (every reader divides by their sum): checked
    # on each assignment, never rewritten, so setting one at a time works.
    config = Config()
    config.search.hybrid.lexical_weight = 0.25
    config.search.hybrid.vector_weight = 0.75
    assert config.search.hybrid.lexical_weight == 0.25
    changed = config.updated(
        {"search.hybrid.lexical_weight": 0.2, "search.hybrid.vector_weight": 0.6}
    )
    assert changed.search.hybrid.lexical_weight == pytest.approx(0.2)
    with pytest.raises(ValidationError):
        config.search.hybrid.lexical_weight = -1.0
    config.search.hybrid.lexical_weight = 0.0
    with pytest.raises(ValidationError, match="cannot be zero"):
        config.search.hybrid.vector_weight = 0.0
    assert config.search.hybrid.vector_weight == 0.75


def test_updated_runs_the_checks_across_sections() -> None:
    # Regression: the console set indexing.methodologies by assignment, so a
    # selection community_detection.enabled false leaves nothing of passed.
    config = _lightrag_without_communities()
    with pytest.raises(ValidationError, match="indexing.methodologies"):
        config.updated({"indexing.methodologies": ["graphrag"]})
    assert config.indexing.methodologies == [Methodology.LIGHTRAG]


def test_updated_returns_a_changed_copy() -> None:
    config = Config()
    changed = config.updated(
        {"search.citations.enabled": False, "indexing.methodologies": ["lightrag"]}
    )
    assert changed.search.citations.enabled is False
    assert changed.indexing.methodologies == [Methodology.LIGHTRAG]
    assert config.search.citations.enabled is not False or config == changed
    assert config.indexing.methodologies != changed.indexing.methodologies


def test_updated_refuses_a_key_that_names_no_setting() -> None:
    with pytest.raises(ValueError, match="No setting 'search.no_such_key'"):
        Config().updated({"search.no_such_key": 1})


def test_the_console_refuses_a_methodology_the_config_cannot_build() -> None:
    # The console refusing the request's selection: invalid_argument.
    with pytest.raises(InvalidArgumentError, match="methodologies"):
        ingest_config(_lightrag_without_communities(), ["graphrag"])
    config, chosen = ingest_config(Config(), ["lightrag"])
    assert chosen == ["lightrag"]
    assert config.indexing.methodologies == [Methodology.LIGHTRAG]


def test_a_section_refuses_a_key_it_does_not_define() -> None:
    # The backstop under Config's walk (which names moved keys): a section
    # validated on its own refused no misspelled key.
    with pytest.raises(ValidationError, match="entity_frequency_treshold"):
        SearchConfig.model_validate({"local_search": {"entity_frequency_treshold": 5}})


def test_a_removed_section_still_loads() -> None:
    config = Config.model_validate(
        {"graph": {"analysis": {"statistics": {"calculate_density": True}}}}
    )
    assert config.graph == Config().graph


def test_the_library_log_levels_are_checked() -> None:
    with pytest.raises(ValidationError, match="library_levels"):
        Config.model_validate({"logging": {"library_levels": {"botocore": "LOUD"}}})
    config = Config.model_validate(
        {"logging": {"level": "debug", "library_levels": {"botocore": "error"}}}
    )
    assert config.logging.level == "DEBUG"
    assert config.logging.library_levels == {"botocore": "ERROR"}


def test_updated_merges_an_overlay_then_sets_the_dotted_keys() -> None:
    config = Config.model_validate({"search": {"citations": {"enabled": False}}})
    changed = config.updated(
        {"search.max_concurrency": 7},
        overlay={"search": {"local_search": {"entity_frequency_threshold": 3}}},
    )
    assert changed.search.max_concurrency == 7
    assert changed.search.local_search.entity_frequency_threshold == 3
    # Merged into the section: what the config set there stays.
    assert changed.search.citations.enabled is False
    with pytest.raises(ValidationError, match="unknown config keys"):
        config.updated(overlay={"search": {"no_such_key": 1}})


def test_the_evaluation_cli_changes_are_validated() -> None:
    # They were model_copy updates: a bogus value was kept unvalidated.
    from unified_kg_rag.application.cli.run_evaluation import with_pairwise_mode

    with pytest.raises(ConfigurationError, match="pairwise_mode"):
        with_pairwise_mode(Config(), "bogus")
    changed = with_pairwise_mode(Config(), "differential")
    assert changed.evaluation.pairwise_mode == "differential"


def test_the_root_validates_an_assignment() -> None:
    # Regression (rv26 architecture): Config itself took any value, so
    # `config.search = "x"` broke every later read of the section.
    config = Config()
    with pytest.raises(ValidationError):
        config.search = "x"  # type: ignore[assignment]
    assert isinstance(config.search, SearchConfig)
    # A whole config's checks run too, and a refused value is not kept.
    lightrag = _lightrag_without_communities()
    indexing = lightrag.indexing
    with pytest.raises(ValidationError, match="indexing.methodologies"):
        lightrag.indexing = indexing.model_copy(
            update={"methodologies": [Methodology.GRAPHRAG]}
        )
    assert lightrag.indexing is indexing
