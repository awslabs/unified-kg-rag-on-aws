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
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.fixtures.config_template import uncommented_template
from unified_kg_rag.application.config import ConfigLoader, env_var
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import MOVED_KEYS, REMOVED_KEYS, unknown_keys
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


def test_a_section_left_with_only_removed_keys_still_names_a_typo() -> None:
    # graph.analysis has no setting left: its removed keys warn, a key that
    # never existed under it is still unknown.
    data = {
        "graph": {
            "analysis": {
                "statistics": {"calculate_diameter": True, "calculate_diamter": True},
                "other": 1,
            }
        }
    }
    assert unknown_keys(data, Config) == [
        "graph.analysis.statistics.calculate_diamter",
        "graph.analysis.other",
    ]


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
    ("cache.chunking", "chunk_size", 500),
    ("graph.visualization.embeddings", "model_id", "amazon.titan-embed-text-v2:0"),
    ("search.lightrag_search.structured_keyword_output", None, True),
    # Settings with one useful value: the behaviour is the former default.
    ("search.global_search.reduce_with_llm", None, True),
    ("search.global_search.map_reduce_model_id", None, "m"),
    ("models.roles.global_reduce", None, "fast"),
    ("search.fusion.method", None, "weighted"),
    ("search.reranking.enabled", None, False),
    ("graph.visualization.centrality.calculate_closeness", None, True),
    ("graph.visualization.centrality.calculate_eigenvector", None, True),
    ("graph.visualization.centrality.eigenvector_max_iter", None, 50),
    ("graph.visualization.centrality.eigenvector_tol", None, 0.01),
    ("graph.analysis.statistics.calculate_density", None, False),
    ("graph.analysis.statistics.calculate_clustering", None, False),
    ("graph.analysis.statistics.calculate_components", None, False),
    ("graph.analysis.statistics.calculate_average_clustering", None, True),
    ("graph.analysis.statistics.calculate_diameter", None, True),
    ("fixing", "enabled", False),
    ("evaluation.warm_up_query", None, False),
    ("graph.community_detection.auto_resolution", None, True),
    ("graph.community_detection.auto_resolution_candidates", None, [0.5, 1.0]),
    ("graph.community_detection.auto_resolution_max_nodes", None, 10),
    ("aws.dynamodb.billing_mode", None, "PROVISIONED"),
]


# The retry keys #468 folded into the retry section: each counts what its
# replacement counts, so a tuned value fails the load naming the new key
# rather than silently reverting to the default.
_MOVED_RETRY_KEYS = {
    "processing.max_attempts": "retry.max_attempts",
    "aws.bedrock.transient_retry.max_attempts": "retry.max_attempts",
    "aws.bedrock.transient_retry.base_delay_seconds": "retry.base_delay_seconds",
    "aws.bedrock.transient_retry.max_delay_seconds": "retry.max_delay_seconds",
    "aws.bedrock.transient_retry.max_total_seconds": "retry.max_total_seconds",
    "providers.llm.max_attempts": "retry.model_server_max_attempts",
    "providers.embedding.max_attempts": "retry.model_server_max_attempts",
    "indexing.neptune.max_attempts": "retry.graph_store_max_attempts",
    "indexing.neptune.retry_delay_seconds": "retry.base_delay_seconds",
    "search.store_max_attempts": "retry.query_store_max_attempts",
    "evaluation.ragas_max_attempts": "retry.judge_max_attempts",
}


@pytest.mark.parametrize(("old", "new"), sorted(_MOVED_RETRY_KEYS.items()))
def test_a_tuned_old_retry_key_fails_naming_its_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old: str, new: str
) -> None:
    assert MOVED_KEYS[old] == new
    assert old not in REMOVED_KEYS
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(_nested(old, 2)))
    with pytest.raises(ConfigurationError, match=f"{old} -> {new}"):
        ConfigLoader(config_file).load_config()
    monkeypatch.setenv(env_var(old), "2")
    with pytest.raises(ConfigurationError, match=f"moved: set {env_var(new)}"):
        ConfigLoader().load_config()


@pytest.mark.parametrize(
    ("old", "new", "note"),
    [
        # 0.2.3 passed it to a pipeline setting nothing read: model calls
        # made BatchProcessor's 5 attempts whatever it said.
        ("processing.max_retries", "retry.max_attempts", "had no effect"),
        # 0.2.3's Neptune indexer made max_retries + 1 attempts.
        ("indexing.neptune.max_retries", "retry.graph_store_max_attempts", "N+1"),
    ],
)
def test_a_0_2_3_retry_count_fails_saying_how_to_convert_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old: str, new: str, note: str
) -> None:
    assert MOVED_KEYS[old] == new
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(_nested(old, 3)))
    with pytest.raises(
        ConfigurationError, match=rf"{old} -> {new} \(.*{re.escape(note)}"
    ):
        ConfigLoader(config_file).load_config()
    monkeypatch.setenv(env_var(old), "3")
    with pytest.raises(
        ConfigurationError, match=rf"moved: set {env_var(new)} \(.*{re.escape(note)}"
    ):
        ConfigLoader().load_config()


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


def test_the_removed_rerank_switch_names_its_replacement() -> None:
    # enabled: false alone no longer turns reranking off; the warning says
    # what does.
    assert "providers.rerank.kind: none" in REMOVED_KEYS["search.reranking.enabled"]


def test_a_moved_key_under_a_removed_section_still_fails_as_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # fixing is removed, but fixing.fixing_model_id moved to
    # models.roles.fixing: ignoring it would drop the model it names.
    old, new = "fixing.fixing_model_id", "models.roles.fixing"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(_nested(old, "m")))
    with pytest.raises(ConfigurationError, match=f"{old} -> {new}"):
        ConfigLoader(config_file).load_config()
    monkeypatch.setenv(env_var(old), "m")
    with pytest.raises(ConfigurationError, match=f"moved: set {env_var(new)}"):
        ConfigLoader().load_config()


def test_the_old_output_cap_key_fails_naming_its_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cap applies to every provider's models, so it left aws.bedrock.
    old, new = (
        "aws.bedrock.default_max_output_tokens",
        "models.default_max_output_tokens",
    )
    assert MOVED_KEYS[old] == new
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(_nested(old, 4096)))
    with pytest.raises(ConfigurationError, match=f"{old} -> {new}"):
        ConfigLoader(config_file).load_config()
    monkeypatch.setenv(env_var(old), "4096")
    with pytest.raises(ConfigurationError, match=f"moved: set {env_var(new)}"):
        ConfigLoader().load_config()
    monkeypatch.delenv(env_var(old))
    config_file.write_text(yaml.safe_dump(_nested(new, 4096)))
    assert (
        ConfigLoader(config_file).load_config().models.default_max_output_tokens == 4096
    )


def test_the_model_refuses_unknown_keys_without_the_loader() -> None:
    # Config is public: a library caller validating a mapping got the
    # default for a misspelled key.
    from pydantic import ValidationError

    data = {"search": {"local_search": {"entity_frequency_treshold": 5}}, "serch": {}}
    with pytest.raises(ValidationError) as raised:
        Config.model_validate(data)
    message = str(raised.value)
    assert "search.local_search.entity_frequency_treshold" in message
    assert "serch" in message


def test_a_removed_key_still_validates_and_is_ignored() -> None:
    assert Config.model_validate({"fixing": {"enabled": False}}) == Config()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("processing.chunk_concurrency", "processing.max_concurrency"),
        ("processing.batch_size", "processing.max_concurrency (it counted"),
        ("indexing.cross_run_fuzzy_merge", "processing.fuzzy_entity_resolution"),
    ],
)
def test_the_renamed_0_2_3_keys_name_their_replacement(old: str, new: str) -> None:
    section, key = old.split(".")
    with pytest.raises(ValueError, match=re.escape(f"{old} -> {new}")):
        Config.model_validate({section: {key: 4}})
