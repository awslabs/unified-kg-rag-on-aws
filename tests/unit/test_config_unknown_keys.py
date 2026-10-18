# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unknown config keys fail the load instead of being silently dropped.

A misspelled key leaves its setting at the default: a guardrail identifier
under ``aws.bedrock.guardrail.identifer`` left every query unscreened, with
only a warning to show for it. ``allow_unknown_keys`` (``--allow-unknown-keys``)
turns the error back into a warning per key.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from tests.fixtures.config_template import uncommented_template
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import ConfigurationError
from unified_kg_rag.shared.config import ConfigLoader, unknown_keys

pytestmark = pytest.mark.unit

_TEMPLATE = Path(__file__).resolve().parents[2] / "config-template.yaml"


def _unknown(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "Unknown config key" in r.message]


_TYPOS = {
    "search": {"rerankng": {"enabled": True}},
    "logging": {"level": "INFO", "colour": True},
    "not_a_section": 1,
    "aws": {"bedrock": {"guardrail": {"identifer": "gr-1"}}},
}


def test_typo_in_nested_section_fails_the_load(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_TYPOS))
    with pytest.raises(ConfigurationError) as raised:
        ConfigLoader(path).load_config()
    message = str(raised.value)
    for key in (
        "aws.bedrock.guardrail.identifer",
        "logging.colour",
        "not_a_section",
        "search.rerankng",
    ):
        assert f"'{key}'" in message
    assert "--allow-unknown-keys" in message


def test_allow_unknown_keys_warns_and_loads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_TYPOS))
    with caplog.at_level(logging.WARNING):
        config = ConfigLoader(path, allow_unknown_keys=True).load_config()
    assert len(_unknown(caplog)) == 4
    assert config.aws.bedrock.guardrail.identifier is None


def test_an_unknown_environment_variable_fails_the_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UKR__AWS__NEPTUNE__ENDPOIN", "graph.example")
    with pytest.raises(ConfigurationError, match="UKR__AWS__NEPTUNE__ENDPOIN"):
        ConfigLoader(None).load_config()
    assert ConfigLoader(None, allow_unknown_keys=True).load_config()


def test_a_moved_key_names_its_replacement(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"graph": {"analysis": {"centrality": {"pagerank_alpha": 0.5}}}})
    )
    with pytest.raises(
        ConfigurationError,
        match="graph.analysis.centrality -> graph.visualization.centrality",
    ):
        ConfigLoader(path).load_config()


def test_free_form_dict_fields_are_not_walked() -> None:
    data = {"logging": {"library_levels": {"some.library": "ERROR"}}}
    assert unknown_keys(data, Config) == []


def test_a_misspelled_renderer_setting_is_an_unknown_key() -> None:
    # The renderer settings were free-form dicts: a typo such as max_node
    # left the cap at its default without a word.
    data = {"graph": {"visualization": {"interactive": {"max_node": 50}}}}
    assert unknown_keys(data, Config) == ["graph.visualization.interactive.max_node"]


@pytest.mark.parametrize("view", ["as_shipped", "uncommented"])
def test_config_template_has_no_unknown_keys(view: str) -> None:
    # The uncommented view also covers the keys the template shows commented
    # out (tests.fixtures.config_template).
    data = (
        yaml.safe_load(_TEMPLATE.read_text())
        if view == "as_shipped"
        else uncommented_template()
    )
    assert unknown_keys(data, Config) == []


@pytest.mark.parametrize("text", ["", "# only a comment\n"])
def test_empty_config_file_loads_defaults(tmp_path: Path, text: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    assert ConfigLoader(path).load_config() == Config()


def test_non_mapping_config_file_is_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        ConfigLoader(path).load_config()


def test_a_removed_key_loads_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch
) -> None:
    # indexing.cross_run_merge is gone: false used to overwrite entities other
    # documents still cite. A config that still sets it loads, says so, and
    # merges anyway.
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"indexing": {"cross_run_merge": False}}))
    monkeypatch.setenv("UKR__INDEXING__CROSS_RUN_MERGE", "false")
    with caplog.at_level(logging.WARNING):
        config = ConfigLoader(path).load_config()
    removed = [
        r.getMessage() for r in caplog.records if "removed and ignored" in r.message
    ]
    assert len(removed) == 2
    assert "'indexing.cross_run_merge'" in removed[0]
    assert "UKR__INDEXING__CROSS_RUN_MERGE" in removed[1]
    assert not hasattr(config.indexing, "cross_run_merge")
