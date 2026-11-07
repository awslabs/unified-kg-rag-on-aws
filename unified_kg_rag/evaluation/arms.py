# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration arms of ``run-eval --arm NAME=overlay.yaml``: run labels
and how the comparison pairs them.

An arm answers the dataset with the run's configuration under an overlay
(another chunker setting, rerank off, citations on, an A/A copy), as its
own chain and evaluation manager, once per strategy: the run of strategy
``s`` under arm ``NAME`` is labelled ``s@NAME`` in the results, file names
and comparison. Each arm run is compared with the same strategy's run
under the base configuration (``s``), so its deltas measure the overlay;
the base strategies are compared with the first, as before.
"""

from __future__ import annotations

from collections.abc import Iterable

# Between a run's strategy and its arm in a run label.
ARM_SEPARATOR = "@"


def arm_label(strategy: str, arm: str | None) -> str:
    """``strategy@arm``, or ``strategy`` for the base configuration."""
    return strategy if arm is None else f"{strategy}{ARM_SEPARATOR}{arm}"


def label_strategy(label: str) -> str:
    """The strategy of a run label (``local`` of ``local@rerank_off``; a
    ``baseline:`` prefix is kept)."""
    return label.split(ARM_SEPARATOR, 1)[0]


def is_arm_label(label: str) -> bool:
    return ARM_SEPARATOR in label


def arm_pairs(labels: Iterable[str]) -> dict[str, str]:
    """Each compared run label's baseline when some label is an arm's: an
    arm run ``s@NAME`` against ``s`` (the first label when ``s`` did not
    run), every other label against the first. Empty without an arm."""
    names = list(labels)
    if not any(is_arm_label(name) for name in names):
        return {}
    first = names[0]
    pairs: dict[str, str] = {}
    for name in names[1:]:
        base = label_strategy(name)
        pairs[name] = base if is_arm_label(name) and base in names else first
    return pairs
