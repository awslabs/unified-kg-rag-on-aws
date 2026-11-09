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

from collections.abc import Iterable, Mapping
from typing import Any, Literal

# Between a run's strategy and its arm in a run label.
ARM_SEPARATOR = "@"

# A compared run that is no candidate, by what it measures: ``closed_book``
# answers without retrieval (``run-eval --closed-book``: a bound on the
# answer model's prior knowledge), ``aa`` runs the base configuration again
# (an A/A ``--arm``: the noise floor). A comparison names them under
# ``arm_roles``; Holm and the gates leave them out, and every surface (the
# Markdown report, the terminal table, the console) marks them with
# ``ARM_ROLE_MARKS`` and reads their deltas as no finding.
ArmRole = Literal["closed_book", "aa"]
ARM_ROLE_MARKS: dict[str, str] = {"closed_book": "no retrieval", "aa": "noise floor"}


def arm_roles(comparison: Mapping[str, Any]) -> dict[str, ArmRole]:
    """The compared runs of ``comparison`` that are no candidate, by role."""
    return dict(comparison.get("arm_roles") or {})


def arm_role_mark(label: str, roles: Mapping[str, str]) -> str:
    """What a table column of ``label`` (its means, its deltas) ends in:
    `` (<its role's mark>)``, or nothing for a candidate."""
    mark = ARM_ROLE_MARKS.get(roles.get(label, ""))
    return f" ({mark})" if mark else ""


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


def aa_arms(manifests: Iterable[tuple[str, Mapping[str, Any] | None]]) -> list[str]:
    """The A/A arms among ``(run label, run manifest)`` pairs: those whose
    manifest records an arm that ran the base configuration unchanged
    (``arm.aa``: an empty overlay on the run's own index). Their deltas are
    the noise floor of the comparison, not a finding."""
    return list(
        dict.fromkeys(
            label
            for label, manifest in manifests
            if ((manifest or {}).get("arm") or {}).get("aa") is True
        )
    )
