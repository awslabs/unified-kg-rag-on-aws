# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The flat ``retry`` keys of 0.3.0 load as deprecated aliases of the
per-caller profiles.

Each loads into every profile it governed (``DEPRECATED_KEYS``, the one rule
of every deprecated key), with a WARNING that names them; it fails only
beside a profile key set to another value, naming both.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from unified_kg_rag.application.config import ConfigLoader, env_var
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import (
    DEPRECATED_KEYS,
    _root,
    deprecated_targets,
)
from unified_kg_rag.shared.exceptions import ConfigurationError

pytestmark = pytest.mark.unit

CONFIG_LOGGER = "unified_kg_rag.domain.models.config"


@pytest.fixture(autouse=True)
def _warn_afresh() -> Iterator[None]:
    # The WARNING is logged once per key and process.
    _root._warn_deprecated.cache_clear()
    yield
    _root._warn_deprecated.cache_clear()


def _deprecations(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "is deprecated" in r.getMessage()
    ]


# Old flat key -> the profile keys it governed, and a value to set.
_ALIASES: dict[str, tuple[tuple[str, ...], Any]] = {
    "retry.max_attempts": (
        ("retry.model.max_attempts", "retry.ingestion_item.max_attempts"),
        7,
    ),
    "retry.base_delay_seconds": (
        (
            "retry.model.base_delay_seconds",
            "retry.model_server.base_delay_seconds",
            "retry.graph_store.base_delay_seconds",
        ),
        0.25,
    ),
    "retry.max_delay_seconds": (
        ("retry.model.max_delay_seconds", "retry.model_server.max_delay_seconds"),
        9.0,
    ),
    "retry.max_total_seconds": (
        ("retry.model.max_total_seconds", "retry.model_server.max_total_seconds"),
        30.0,
    ),
    "retry.model_server_max_attempts": (("retry.model_server.max_attempts",), 6),
    "retry.graph_store_max_attempts": (("retry.graph_store.max_attempts",), 6),
    "retry.query_store_max_attempts": (("retry.query_store.max_attempts",), 6),
}


def _nested(path: str, value: Any) -> dict[str, Any]:
    data: Any = value
    for part in reversed(path.split(".")):
        data = {part: data}
    return data


def _value(config: Config, path: str) -> Any:
    node: Any = config
    for part in path.split("."):
        node = getattr(node, part)
    return node


def test_every_flat_key_is_an_alias() -> None:
    for old, (targets, _) in _ALIASES.items():
        assert deprecated_targets(old) == targets
    assert set(_ALIASES) <= set(DEPRECATED_KEYS)


@pytest.mark.parametrize("old", sorted(_ALIASES))
def test_an_old_key_loads_into_every_profile_it_governed(
    old: str, caplog: pytest.LogCaptureFixture
) -> None:
    targets, value = _ALIASES[old]
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        config = Config(**_nested(old, value))
    for target in targets:
        assert _value(config, target) == value, target
    [message] = _deprecations(caplog)
    assert old in message
    for target in targets:
        assert target in message
    if len(targets) > 1:
        assert "several callers" in message


def test_the_other_profile_keys_keep_their_defaults() -> None:
    config = Config(retry={"base_delay_seconds": 0.25})
    # The profiles it did not govern, and the rest of those it did.
    assert config.retry.ingestion_item.base_delay_seconds == 30.0
    assert config.retry.query_store.base_delay_seconds == 0.5
    assert config.retry.graph_store.jitter == "full"
    assert config.retry.model.max_attempts == 5


def test_an_old_key_beside_a_partly_set_profile_fills_it_in() -> None:
    config = Config(retry={"max_attempts": 3, "model": {"base_delay_seconds": 1.0}})
    assert config.retry.model.max_attempts == 3
    assert config.retry.model.base_delay_seconds == 1.0
    assert config.retry.model.max_delay_seconds == 16.0


def test_an_old_key_beside_its_replacement_at_the_same_value_loads() -> None:
    config = Config(retry={"max_attempts": 3, "ingestion_item": {"max_attempts": 3}})
    assert config.retry.ingestion_item.max_attempts == 3
    assert config.retry.model.max_attempts == 3


def test_an_old_key_beside_a_different_replacement_fails_naming_both() -> None:
    with pytest.raises(ValueError) as excinfo:
        Config(retry={"max_attempts": 3, "ingestion_item": {"max_attempts": 8}})
    assert "retry.max_attempts and retry.ingestion_item.max_attempts" in str(
        excinfo.value
    )


def test_a_file_and_an_environment_variable_load_the_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({"retry": {"graph_store_max_attempts": 6}}))
    config = ConfigLoader(config_file).load_config()
    assert config.retry.graph_store.max_attempts == 6
    monkeypatch.setenv(env_var("retry.max_attempts"), "2")
    with caplog.at_level(logging.WARNING):
        config = ConfigLoader().load_config()
    assert config.retry.model.max_attempts == 2
    assert config.retry.ingestion_item.max_attempts == 2
    assert any(
        "UKR__RETRY__MODEL__MAX_ATTEMPTS, UKR__RETRY__INGESTION_ITEM__MAX_ATTEMPTS"
        in message
        for message in _deprecations(caplog)
    )


def test_an_invalid_alias_value_fails_at_its_replacement() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        Config(retry={"query_store_max_attempts": 0})


def test_an_invalid_alias_variable_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env_var("retry.max_attempts"), "0")
    with pytest.raises(ConfigurationError, match="UKR__RETRY__MAX_ATTEMPTS"):
        ConfigLoader().load_config()


def test_an_alias_variable_overrides_the_file_as_its_replacements_would(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A variable sets what its replacement variables set, over the file.
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({"retry": {"model": {"max_attempts": 4}}}))
    monkeypatch.setenv(env_var("retry.max_attempts"), "2")
    config = ConfigLoader(config_file).load_config()
    assert config.retry.model.max_attempts == 2
    assert config.retry.ingestion_item.max_attempts == 2


def test_a_variable_wins_over_the_files_deprecated_key_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The deprecation warning tells a 0.3.0 user to set the profile key; set
    # by a variable it failed the load against the file's flat key.
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({"retry": {"max_attempts": 3}}))
    monkeypatch.setenv(env_var("retry.model.max_attempts"), "5")
    with caplog.at_level(logging.WARNING):
        config = ConfigLoader(config_file).load_config()
    assert config.retry.model.max_attempts == 5
    assert config.retry.ingestion_item.max_attempts == 3
    assert any(
        "UKR__RETRY__MODEL__MAX_ATTEMPTS sets retry.model.max_attempts over the "
        "file's deprecated retry.max_attempts" in r.getMessage()
        for r in caplog.records
    )


def test_a_deprecated_variable_beside_its_replacement_must_agree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # RV4-7: the replacement won with a WARNING; an old and a new name given
    # different values are refused, as a file's pair is.
    from unified_kg_rag.shared.exceptions import ConfigurationError

    monkeypatch.setenv(env_var("retry.max_attempts"), "2")
    monkeypatch.setenv(env_var("retry.model.max_attempts"), "5")
    with pytest.raises(ConfigurationError) as refused:
        ConfigLoader().load_config()
    assert (
        "UKR__RETRY__MAX_ATTEMPTS (deprecated) and UKR__RETRY__MODEL__MAX_ATTEMPTS "
        "both set retry.model.max_attempts, to different values"
    ) in str(refused.value)
    assert "2" not in str(refused.value).split(":")[-1]  # never a value
    # The same value: loaded, the deprecated one into its other keys too.
    monkeypatch.setenv(env_var("retry.model.max_attempts"), "2")
    config = ConfigLoader().load_config()
    assert config.retry.model.max_attempts == 2
    assert config.retry.ingestion_item.max_attempts == 2
