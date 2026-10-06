# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renamed config keys keep working, keep their effect, and say so in the log."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared.config import ConfigLoader

pytestmark = pytest.mark.unit


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records]


def test_neptune_max_retries_keeps_its_attempt_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # max_retries counted retries after the first try: 3 retries = 4 attempts.
    with caplog.at_level(logging.WARNING):
        config = Config.model_validate({"indexing": {"neptune": {"max_retries": 3}}})
    assert config.indexing.neptune.max_attempts == 4
    assert (
        "Config key 'indexing.neptune.max_retries' is deprecated; applied as "
        "'indexing.neptune.max_attempts: 4'. Rename it in your config."
    ) in _messages(caplog)


def test_neptune_max_retries_zero_is_a_single_attempt() -> None:
    config = Config.model_validate({"indexing": {"neptune": {"max_retries": 0}}})
    assert config.indexing.neptune.max_attempts == 1


@pytest.mark.parametrize(
    ("section", "old", "new"),
    [
        ("processing", "max_retries", "max_attempts"),
        ("evaluation", "ragas_max_retries", "ragas_max_attempts"),
    ],
)
def test_total_attempt_keys_keep_their_value(section: str, old: str, new: str) -> None:
    config = Config.model_validate({section: {old: 2}})
    assert getattr(getattr(config, section), new) == 2


def test_replacement_key_wins_over_the_legacy_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        config = Config.model_validate(
            {"indexing": {"neptune": {"max_retries": 9, "max_attempts": 2}}}
        )
    assert config.indexing.neptune.max_attempts == 2
    assert (
        "Deprecated config key 'indexing.neptune.max_retries' is ignored because "
        "'indexing.neptune.max_attempts' is set"
    ) in _messages(caplog)


def test_legacy_keys_are_not_reported_as_unknown(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "search": {"llm_retry": {"max_attempts": 2}},
                "processing": {"max_retries": 3},
                "indexing": {"neptune": {"max_retries": 1}},
                "evaluation": {"ragas_max_retries": 2},
            }
        )
    )
    with caplog.at_level(logging.WARNING):
        config = ConfigLoader(path).load_config()
    assert not [m for m in _messages(caplog) if "Unknown config key" in m]
    assert config.aws.bedrock.transient_retry.max_attempts == 2
    assert config.processing.max_attempts == 3
    assert config.indexing.neptune.max_attempts == 2
    assert config.evaluation.ragas_max_attempts == 2
