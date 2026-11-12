# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add licences and the root component's version to a CycloneDX SBOM.

``uv export --format cyclonedx1.5`` lists every locked package with its
hashes but no licence, and its root component has no version. This reads the
licences of an environment synced from the same lock (``pip-licenses
--format=json``, as the licenses workflow runs it) and writes each matching
component's licence into the SBOM in place, matched by normalized name and
version. The root component gets the package version, its purl and its own
licence. A component the environment does not hold (an environment marker
of another platform or Python) is left without one and listed on stderr.

Usage::

    pip-licenses --format=json > licenses.json
    python sbom_licenses.py SBOM.cdx.json licenses.json --version 0.3.0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# The project's own licence (pyproject.toml `license`); the project is not
# installed in the synced environment, so pip-licenses does not list it.
PROJECT_LICENSE = "Apache-2.0"
# pip-licenses reports a package without licence metadata as UNKNOWN.
UNLABELED = {"UNKNOWN", ""}


def normalize(name: str) -> str:
    """PEP 503 name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def add_licenses(sbom: dict, licenses: list[dict[str, str]], version: str) -> list[str]:
    """Write the licences into ``sbom``; return the components left without."""
    by_package = {
        (normalize(p["Name"]), p["Version"]): p["License"]
        for p in licenses
        if p.get("License", "") not in UNLABELED
    }
    missing = []
    for component in sbom.get("components", []):
        license_ = by_package.get(
            (normalize(component["name"]), component.get("version", ""))
        )
        if license_ is None:
            missing.append(f"{component['name']} {component.get('version', '')}")
            continue
        component["licenses"] = [{"license": {"name": license_}}]
    root = sbom["metadata"]["component"]
    root["version"] = version
    root["purl"] = f"pkg:pypi/{normalize(root['name'])}@{version}"
    root["licenses"] = [{"license": {"id": PROJECT_LICENSE}}]
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sbom", type=Path, help="CycloneDX JSON, rewritten in place")
    parser.add_argument("licenses", type=Path, help="pip-licenses --format=json")
    parser.add_argument("--version", required=True, help="the package version")
    args = parser.parse_args(argv)
    sbom = json.loads(args.sbom.read_text(encoding="utf-8"))
    licenses = json.loads(args.licenses.read_text(encoding="utf-8"))
    missing = add_licenses(sbom, licenses, args.version)
    args.sbom.write_text(json.dumps(sbom, indent=2) + "\n", encoding="utf-8")
    total = len(sbom.get("components", []))
    print(f"licences for {total - len(missing)} of {total} components")
    if missing:
        print(
            "not in the synced environment (no licence): " + ", ".join(missing),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
