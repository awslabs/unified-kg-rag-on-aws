# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one quantile definition of the reports."""

from __future__ import annotations

import numpy as np
import pytest

from unified_kg_rag.shared.utils.stats import quantile

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.95, 1.0])
def test_quantile_matches_numpy(q: float) -> None:
    values = [7.0, 1.0, 3.0, 10.0, 2.5]
    assert quantile(values, q) == pytest.approx(float(np.quantile(values, q)))


def test_quantile_of_nothing_is_none() -> None:
    assert quantile([], 0.95) is None
    assert quantile([4.0], 0.95) == 4.0
