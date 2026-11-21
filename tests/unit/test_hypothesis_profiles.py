# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Hypothesis profiles tests/conftest.py registers: CI drops only the
wall-clock checks that flake on a shared runner, never the examples."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, settings

pytestmark = pytest.mark.unit


def test_ci_drops_the_wall_clock_checks_and_keeps_the_examples() -> None:
    ci, default = settings.get_profile("ci"), settings.get_profile("default")
    assert ci.deadline is None
    assert HealthCheck.too_slow in ci.suppress_health_check
    # Only that check: a strategy that filters too much still fails.
    assert set(ci.suppress_health_check) == {HealthCheck.too_slow}
    assert ci.max_examples == default.max_examples
    assert HealthCheck.too_slow not in default.suppress_health_check
