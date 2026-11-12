# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
""".github/scripts/sbom_licenses.py: the release SBOM carries licences.

``uv export --format cyclonedx1.5`` writes no licence for any component and
no version for the root one; the release workflow fills both in from an
environment synced from the same lock.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "sbom_licenses.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sbom_licenses", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sbom_licenses = _load()


def _sbom() -> dict:
    # The shape uv 0.12 exports.
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "metadata": {
            "component": {
                "type": "library",
                "bom-ref": "unified-kg-rag-on-aws-1",
                "name": "unified-kg-rag-on-aws",
            }
        },
        "components": [
            {"name": "Jinja2", "version": "3.1.6", "purl": "pkg:pypi/jinja2@3.1.6"},
            {"name": "typing_extensions", "version": "4.15.0"},
            {"name": "pywin32", "version": "312"},
            {"name": "unlabeled-pkg", "version": "1.0"},
        ],
    }


_LICENSES = [
    {"Name": "jinja2", "Version": "3.1.6", "License": "BSD License"},
    {"Name": "typing-extensions", "Version": "4.15.0", "License": "PSF-2.0"},
    {"Name": "unlabeled-pkg", "Version": "1.0", "License": "UNKNOWN"},
    # Another version than the locked one is not this component's licence.
    {"Name": "pywin32", "Version": "311", "License": "PSF"},
]


def test_components_get_the_synced_environments_licences() -> None:
    sbom = _sbom()
    missing = sbom_licenses.add_licenses(sbom, _LICENSES, "0.3.0")
    licences = {c["name"]: c.get("licenses") for c in sbom["components"]}
    # Matched by normalized name (PEP 503) and version.
    assert licences["Jinja2"] == [{"license": {"name": "BSD License"}}]
    assert licences["typing_extensions"] == [{"license": {"name": "PSF-2.0"}}]
    assert licences["pywin32"] is None
    assert licences["unlabeled-pkg"] is None
    assert missing == ["pywin32 312", "unlabeled-pkg 1.0"]


def test_the_root_component_gets_the_package_version_and_licence() -> None:
    sbom = _sbom()
    sbom_licenses.add_licenses(sbom, [], "0.3.0rc1")
    root = sbom["metadata"]["component"]
    assert root["version"] == "0.3.0rc1"
    assert root["purl"] == "pkg:pypi/unified-kg-rag-on-aws@0.3.0rc1"
    assert root["licenses"] == [{"license": {"id": "Apache-2.0"}}]


def test_the_project_licence_matches_pyproject() -> None:
    import tomllib

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    assert pyproject["project"]["license"] == {"text": sbom_licenses.PROJECT_LICENSE}


def test_main_rewrites_the_sbom_in_place(tmp_path: Path) -> None:
    sbom_path = tmp_path / "sbom.cdx.json"
    sbom_path.write_text(json.dumps(_sbom()), encoding="utf-8")
    licenses_path = tmp_path / "licenses.json"
    licenses_path.write_text(json.dumps(_LICENSES), encoding="utf-8")
    argv = [str(sbom_path), str(licenses_path), "--version", "0.3.0"]
    assert sbom_licenses.main(argv) == 0
    written = json.loads(sbom_path.read_text(encoding="utf-8"))
    assert written["metadata"]["component"]["version"] == "0.3.0"
    assert written["components"][0]["licenses"]


def test_the_release_publishes_the_licensed_sbom() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    runs = "\n".join(step.get("run", "") for step in jobs["sbom"]["steps"])
    assert "uv sync --locked --all-extras --no-dev" in runs
    assert ".github/scripts/sbom_licenses.py" in runs
    # The dry run builds it too (no event condition); the release attests it.
    assert "if" not in jobs["sbom"]
    assert "sbom" in jobs["github-release"]["needs"]
    attest = [
        s for s in jobs["github-release"]["steps"] if "attest" in s.get("uses", "")
    ]
    assert "release-assets/*.cdx.json" in attest[0]["with"]["subject-path"]
