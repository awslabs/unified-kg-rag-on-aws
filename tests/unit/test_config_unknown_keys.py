# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unknown config keys fail the load instead of being silently dropped.

A misspelled key leaves its setting at the default: a guardrail identifier
under ``aws.bedrock.guardrail.identifer`` left every query unscreened, with
only a warning to show for it. ``allow_unknown_keys`` (``--allow-unknown-keys``)
turns the error back into a warning per key.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.fixtures.config_template import uncommented_template
from unified_kg_rag.application.config import ConfigLoader, env_var, unknown_keys
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import REMOVED_KEYS
from unified_kg_rag.shared import ConfigurationError

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


def test_the_removed_ingestion_rendering_keys_load_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Ingestion no longer renders: a config written for it (rendering on, an
    # embedding layout and its UMAP parameters) loads, says each key is gone,
    # and still exports the data run-visualization renders.
    path = tmp_path / "config.yaml"
    visualization = {
        "enabled": True,
        "embedding_method": "description",
        "layout_method": "umap",
        "layout": {"umap": {"n_neighbors": 5}},
    }
    path.write_text(yaml.safe_dump({"graph": {"visualization": visualization}}))
    with caplog.at_level(logging.WARNING):
        config = ConfigLoader(path).load_config()
    removed = [
        r.getMessage() for r in caplog.records if "removed and ignored" in r.message
    ]
    assert len(removed) == 4
    assert all("run-visualization" in message for message in removed)
    assert config.graph.visualization.export_data is True
    assert not set(visualization) & set(type(config.graph.visualization).model_fields)


# Removed settings: (the removed key, a path at or under it a config sets, a
# value it could hold).
_REMOVED_SETTINGS = [
    ("indexing.opensearch.persist_embedding_cache", None, True),
    ("indexing.opensearch.embedding_cache_s3_key", None, "embedding-cache/cache.json"),
    ("search.local_search.type_quota", "text_floor", 4),
    ("cache.chunking", "chunk_size", 500),
]


def _nested(path: str, value: Any) -> dict[str, Any]:
    data: Any = value
    for part in reversed(path.split(".")):
        data = {part: data}
    return data


@pytest.mark.parametrize(("removed", "under", "value"), _REMOVED_SETTINGS)
def test_a_removed_setting_loads_with_a_warning_from_file_or_environment(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    removed: str,
    under: str | None,
    value: Any,
) -> None:
    # A config written before the key went away still loads (no unknown-key
    # error), names the key and why it is gone, and keeps the defaults.
    assert removed in REMOVED_KEYS
    path = f"{removed}.{under}" if under else removed
    unset = ConfigLoader().load_config()
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(_nested(path, value)))
    with caplog.at_level(logging.WARNING):
        assert ConfigLoader(config_file).load_config() == unset
    monkeypatch.setenv(env_var(path), json.dumps(value))
    with caplog.at_level(logging.WARNING):
        assert ConfigLoader().load_config() == unset
    warnings = [
        r.getMessage() for r in caplog.records if "removed and ignored" in r.message
    ]
    assert len(warnings) == 2
    assert f"'{removed}'" in warnings[0] and REMOVED_KEYS[removed] in warnings[0]
    assert env_var(path) in warnings[1] and REMOVED_KEYS[removed] in warnings[1]
