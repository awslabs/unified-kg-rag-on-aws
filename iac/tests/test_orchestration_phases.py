# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Step Functions phases cover every pipeline stage, in pipeline order."""

from __future__ import annotations

import ast
from pathlib import Path

from iac.stacks.orchestration_stack import PHASES

# The IaC environment does not install the package, so read the stage enum from
# its source instead of importing it.
_STAGES_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "unified_kg_rag"
    / "domain"
    / "models"
    / "config.py"
)


def _pipeline_stages() -> list[str]:
    tree = ast.parse(_STAGES_SOURCE.read_text(encoding="utf-8"))
    (enum,) = (
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PipelineStageType"
    )
    return [
        node.value.value
        for node in enum.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]


def test_phases_run_every_pipeline_stage_in_order() -> None:
    stages = _pipeline_stages()
    assert stages, "PipelineStageType not found"
    # A stage missing here would never run on the deployed pipeline: each
    # phase passes only its own stages to --enabled-stages.
    assert [stage for _, window in PHASES for stage in window] == stages
