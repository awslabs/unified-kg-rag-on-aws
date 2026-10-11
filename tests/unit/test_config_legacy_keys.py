# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keys renamed before the first release are reported and ignored."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from unified_kg_rag.shared.config import ConfigLoader

pytestmark = pytest.mark.unit


def test_renamed_keys_are_reported_as_unknown(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "search": {"llm_retry": {"max_attempts": 2}},
                "processing": {"max_retries": 3},
                "indexing": {"neptune": {"max_retries": 9}},
            }
        )
    )
    with caplog.at_level(logging.WARNING):
        config = ConfigLoader(path).load_config()
    messages = [r.getMessage() for r in caplog.records]
    for key in (
        "search.llm_retry",
        "processing.max_retries",
        "indexing.neptune.max_retries",
    ):
        assert f"Unknown config key '{key}' is ignored" in messages
    assert config.indexing.neptune.max_attempts != 10
