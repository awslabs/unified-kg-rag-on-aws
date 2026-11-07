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


_TRIVY = (
    "aquasec/trivy:0.76.1@sha256:"
    "790fd66d1697395d8693d1acaa1ea702406a20f109b6fc7bac8db623eca6c06c"
)


@pytest.mark.parametrize(
    ("job", "image"),
    [
        ("docker", "unified-kg-rag:ci"),
        ("docker-unstructured", "unified-kg-rag:ci-unstructured"),
    ],
)
def test_both_images_are_scanned_for_fixable_vulnerabilities(
    job: str, image: str
) -> None:
    (scan,) = (s["run"] for s in _steps(job) if "trivy" in s.get("run", ""))
    # A pinned scanner, the accepted findings and the gate: fixable HIGH and
    # CRITICAL findings fail the job.
    assert _TRIVY in scan
    assert "--severity HIGH,CRITICAL --ignore-unfixed --exit-code 1" in scan
    assert scan.endswith(f"--ignorefile /trivyignore.yaml {image}")
    assert (_WORKFLOW.parents[2] / ".trivyignore.yaml").is_file()


def test_the_runtime_image_has_no_uv_or_aws_cli_and_a_read_only_venv() -> None:
    runs = " ".join(step.get("run", "") for step in _steps("docker"))
    assert "! command -v uv && ! command -v aws" in runs
    assert "! touch /app/.venv/probe" in runs
