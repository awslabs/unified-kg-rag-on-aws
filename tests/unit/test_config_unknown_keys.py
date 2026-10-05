# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unknown config keys are warned about instead of silently dropped."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared.config import ConfigLoader, warn_unknown_keys

pytestmark = pytest.mark.unit

_TEMPLATE = Path(__file__).resolve().parents[2] / "config-template.yaml"


def _unknown(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "Unknown config key" in r.message]


def test_typo_in_nested_section_is_warned(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "search": {"rerankng": {"enabled": True}},
                "logging": {"level": "INFO", "colour": True},
                "not_a_section": 1,
            }
        )
    )
    with caplog.at_level(logging.WARNING):
        ConfigLoader(path).load_config()
    assert sorted(_unknown(caplog)) == [
        "Unknown config key 'logging.colour' is ignored",
        "Unknown config key 'not_a_section' is ignored",
        "Unknown config key 'search.rerankng' is ignored",
    ]


def test_free_form_dict_fields_are_not_walked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = {
        "graph": {"visualization": {"interactive": {"any_renderer_option": 1}}},
        "logging": {"library_levels": {"some.library": "ERROR"}},
    }
    with caplog.at_level(logging.WARNING):
        warn_unknown_keys(data, Config)
    assert _unknown(caplog) == []


def test_config_template_has_no_unknown_keys(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        warn_unknown_keys(yaml.safe_load(_TEMPLATE.read_text()), Config)
    assert _unknown(caplog) == []
