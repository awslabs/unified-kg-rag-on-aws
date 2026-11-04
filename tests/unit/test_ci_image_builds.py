# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CI quality workflow builds both data-plane images.

The default image, and the one with the opt-in ``unstructured`` extra
(Markdown and HTML), whose build bakes in the spaCy model a private VPC
cannot download: that build path was exercised nowhere before a deploy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/quality.yml"


def _steps(job: str) -> list[dict[str, Any]]:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"][job]["steps"]


def test_the_default_image_is_built_and_run() -> None:
    runs = " ".join(step.get("run", "") for step in _steps("docker"))
    assert "docker build -f docker/Dockerfile" in runs
    assert "UV_EXTRAS" not in runs


def test_the_unstructured_image_is_built_with_its_spacy_model() -> None:
    steps = _steps("docker-unstructured")
    build, check = (step.get("run", "") for step in steps[1:3])
    assert "docker build -f docker/Dockerfile" in build
    assert '--build-arg "UV_EXTRAS=--extra unstructured"' in build
    # Checked inside the image it built, with no network.
    assert "ci-unstructured" in build and "ci-unstructured" in check
    assert "--network none" in check and "import en_core_web_sm" in check
