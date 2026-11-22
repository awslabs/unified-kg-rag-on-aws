# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The golden-metrics regression gate (docs/guides/evaluate.md "Golden
metrics"): every search strategy and search type answers the synthetic
structural and bridge datasets on fake models, and every score and answer
fingerprint of a gated partition (the structural benchmarks' held-out
``test`` half, the whole of the unsplit bridge datasets) must equal the
committed baseline (``tests/fixtures/golden_metrics.json``). The ``tune``
half's differences are printed, not gated.

About four minutes on one core (each case, search type and split is a
test of its own, so ``-n`` spreads them), so skipped unless
``GOLDEN_METRICS=1`` (the ``golden-metrics`` CI job sets it)::

    GOLDEN_METRICS=1 uv run pytest tests/integration/test_golden_metrics.py

A change you meant: ``uv run python scripts/gen_golden_metrics.py``
rewrites the baseline, and its diff shows every score that moved.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.fixtures import golden_metrics
from tests.integration.test_embedded_e2e import no_aws as _no_aws  # noqa: F401
from unified_kg_rag.domain.models import SearchType

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.usefixtures("_no_aws"),
    pytest.mark.skipif(
        os.getenv("GOLDEN_METRICS") != "1",
        reason="set GOLDEN_METRICS=1 (about four minutes)",
    ),
]


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    ("case", "search_type", "split"),
    golden_metrics.RUNS,
    ids=[
        "-".join([case, search_type.value, *([split] if split else [])])
        for case, search_type, split in golden_metrics.RUNS
    ],
)
def test_the_scores_equal_the_golden_baseline(
    tmp_path: Path, case: str, search_type: SearchType, split: str | None
) -> None:
    splits = (split,) if split else None
    measured = {
        "schema_version": golden_metrics.SCHEMA_VERSION,
        "cases": {
            case: golden_metrics.measure_case(
                case, tmp_path / case, (search_type,), splits
            )
        },
    }
    baseline = golden_metrics.select(
        golden_metrics.read_baseline(), case, search_type, split
    )
    gated, reported = golden_metrics.compare(baseline, measured)
    for line in reported:
        print(f"tune (not gated): {line}")
    assert not gated, (
        "The golden metrics moved (a gated partition):\n"
        + "\n".join(gated)
        + "\nIf the change is intended, run "
        "`uv run python scripts/gen_golden_metrics.py` and commit the baseline."
    )
