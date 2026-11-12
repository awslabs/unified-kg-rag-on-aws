# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``shared.logging.warn_once``: the one "warn once" of the package."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from unified_kg_rag.shared import logging as shared_logging
from unified_kg_rag.shared.logging import warn_once

pytestmark = pytest.mark.unit

_LOGGER = logging.getLogger("tests.warn_once")


@pytest.fixture(autouse=True)
def _unwarned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shared_logging, "_warned_once", set())


def _levels(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.levelname for r in caplog.records if r.name == _LOGGER.name]


def test_once_then_nothing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger=_LOGGER.name):
        assert warn_once(_LOGGER, "k", "hit %s", 1)
        assert not warn_once(_LOGGER, "k", "hit %s", 2)
    assert _levels(caplog) == ["WARNING"]


def test_warning_then_debug(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger=_LOGGER.name):
        for n in range(3):
            warn_once(_LOGGER, ("cap", "x"), "hit %s", n, then_debug=True)
    assert _levels(caplog) == ["WARNING", "DEBUG", "DEBUG"]


def test_a_callers_own_keys(caplog: pytest.LogCaptureFixture) -> None:
    seen: set[Any] = set()
    with caplog.at_level(logging.WARNING, logger=_LOGGER.name):
        warn_once(_LOGGER, "k", "own", seen=seen)
        warn_once(_LOGGER, "k", "own", seen=seen)
        # The process's keys are another set.
        warn_once(_LOGGER, "k", "process")
    assert _levels(caplog) == ["WARNING", "WARNING"] and seen == {"k"}
