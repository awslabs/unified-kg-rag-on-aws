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


def test_the_eval_extra_is_opt_in_in_the_image() -> None:
    # run-eval's default evaluators are LLM-free: the default image installs
    # no extra, and the opt-in image is the one built with the eval extra.
    dockerfile = (_WORKFLOW.parents[2] / "docker" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    syncs = [line for line in dockerfile.splitlines() if "uv sync" in line]
    assert len(syncs) == 2
    assert all("--extra" not in line and "${UV_EXTRAS}" in line for line in syncs)
    default = " ".join(step.get("run", "") for step in _steps("docker"))
    assert "'ragas', 'datasets', 'pandas'" in default and "sys.exit" in default
    extras = " ".join(step.get("run", "") for step in _steps("docker-unstructured"))
    assert (
        'ci-unstructured python -c "import datasets, langchain_openai, pandas, ragas"'
        in extras
    )


def _step(job: str, name: str) -> dict[str, Any]:
    (step,) = (s for s in _steps(job) if s.get("name", "").startswith(name))
    return step


def test_the_unstructured_image_is_built_with_its_spacy_model() -> None:
    build = _step("docker-unstructured", "Build the image")["run"]
    check = _step("docker-unstructured", "Check the image has the spaCy")["run"]
    assert "docker build -f docker/Dockerfile" in build
    assert '--build-arg "UV_EXTRAS=--extra unstructured --extra eval"' in build
    # Checked inside the image it built, with no network.
    assert "ci-unstructured" in build and "ci-unstructured" in check
    assert "--network none" in check and "import en_core_web_sm" in check


_SPACY_CACHE = "~/.cache/unified-kg-rag/spacy-models"


def test_the_spacy_model_wheel_is_cached_and_installed_with_retries() -> None:
    # A 502 from GitHub releases, where unstructured downloads the model in
    # one attempt, failed the optional-parsers and docker-unstructured jobs.
    version = _step("optional-parsers", "Read the spaCy model version")
    assert "install_spacy_model.py --print-version" in version["run"]
    cache = _step("optional-parsers", "Cache the spaCy model wheel")
    assert cache["uses"].startswith("actions/cache@")
    assert cache["with"]["path"] == _SPACY_CACHE
    assert cache["with"]["key"] == (
        f"spacy-en_core_web_sm-${{{{ steps.{version['id']}.outputs.version }}}}"
    )
    steps = _steps("optional-parsers")
    install = _step("optional-parsers", "Install the spaCy model")
    tests = _step("optional-parsers", "Check optional parsing")
    assert install["run"].endswith("python scripts/install_spacy_model.py")
    assert steps.index(cache) < steps.index(install) < steps.index(tests)

    # The image build takes the cached wheel as its spacy-wheels context and
    # installs it through the same script.
    restore = _step("docker-unstructured", "Restore the spaCy model wheel")
    assert restore["uses"].startswith("actions/cache/restore@")
    assert restore["uses"].split("@")[1] == cache["uses"].split("@")[1]
    assert restore["with"]["path"] == _SPACY_CACHE
    assert restore["with"]["restore-keys"] == "spacy-en_core_web_sm-"
    build = _step("docker-unstructured", "Build the image")["run"]
    assert (
        '--build-context "spacy-wheels=$HOME/.cache/unified-kg-rag/spacy-models"'
        in build
    )
    dockerfile = (_WORKFLOW.parents[2] / "docker" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "FROM scratch AS spacy-wheels" in dockerfile
    assert "install_spacy_model.py --cache-dir /var/cache/spacy-wheels" in dockerfile


_TRIVY = (
    "aquasec/trivy:0.76.2@sha256:"
    "8f6bb947dbc528902e11f2f5be52f7b519f6a39d5d148f41166f2040499768fb"
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


def test_the_image_holds_the_source_it_was_built_from(tmp_path: Path) -> None:
    # uv keys the project's own wheel by its version: a warm cache installed
    # an earlier commit's package. The build reinstalls it, checks the
    # installed files against the copied source and bakes the digest in;
    # the CI job compares it with the checkout's.
    import importlib.util

    root = _WORKFLOW.parents[2]
    dockerfile = (root / "docker" / "Dockerfile").read_text(encoding="utf-8")
    (install,) = (
        line
        for line in dockerfile.splitlines()
        if "uv sync" in line and "--no-editable" in line
    )
    assert "--reinstall-package unified-kg-rag-on-aws" in install
    assert "package_digest.py --installed" in dockerfile
    assert "/app/.venv/unified-kg-rag-source.sha256" in dockerfile
    check = _step("docker", "Check the image holds this checkout's package")["run"]
    assert "scripts/package_digest.py unified_kg_rag" in check
    assert "unified-kg-rag-source.sha256" in check and "--installed" in check

    spec = importlib.util.spec_from_file_location(
        "package_digest", root / "scripts" / "package_digest.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = tmp_path / "pkg"
    (package / "sub" / "__pycache__").mkdir(parents=True)
    (package / "a.py").write_text("A = 1\n")
    (package / "sub" / "b.py").write_text("B = 1\n")
    before = module.package_digest(package)
    (package / "sub" / "__pycache__" / "b.cpython-312.pyc").write_bytes(b"\0")
    assert module.package_digest(package) == before  # bytecode aside
    (package / "sub" / "b.py").write_text("B = 2\n")
    assert module.package_digest(package) != before
