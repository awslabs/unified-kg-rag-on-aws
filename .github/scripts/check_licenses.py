# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail on copyleft, source-available or unlabeled dependency licenses.

Reads ``pip-licenses --format=json`` output on stdin (run it in an environment
synced with every extra) and exits non-zero when a package's license matches
DENIED and the package is not on ALLOWED. Each ALLOWED entry names why it is
acceptable; add one only after reviewing the license.

With ``--npm-lock console/web/package-lock.json`` it checks the console web
client's npm dependencies instead, from the lockfile's own license fields
(no network): the runtime ones, which the built client bundles; the
devDependencies only build and test it and are not redistributed.

Usage::

    pip-licenses --format=json | python check_licenses.py
    python check_licenses.py --npm-lock console/web/package-lock.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Copyleft (strong and weak), source-available and proprietary terms. Weak
# copyleft is denied too so each such package gets an explicit review below.
DENIED = re.compile(
    r"\b(?:[AL]?GPL|MPL|EPL|CDDL|EUPL|SSPL|BUSL)\b|General Public License"
    r"|Mozilla Public|Eclipse Public|Server Side Public|Business Source"
    r"|Commons Clause|Proprietary",
    re.IGNORECASE,
)
# pip-licenses reports a package without license metadata as UNKNOWN.
UNLABELED = {"UNKNOWN", ""}

# name (or name prefix ending in "-") -> reason
ALLOWED = {
    "certifi": "MPL-2.0 (weak, file-level copyleft), unmodified",
    "orjson": "MPL-2.0 AND (Apache-2.0 OR MIT), unmodified",
    "tqdm": "MPL-2.0 AND MIT, unmodified",
    "nvidia-": "NVIDIA CUDA runtime libraries, pulled in by torch in [cross-encoder]",
    "cuda-": "NVIDIA CUDA toolkit metapackage, pulled in by torch in [cross-encoder]",
}


def _allowed(name: str) -> bool:
    name = name.lower()
    return any(
        name.startswith(key) if key.endswith("-") else name == key for key in ALLOWED
    )


def npm_packages(lock: dict[str, object]) -> list[dict[str, str]]:
    """The runtime packages of an npm lockfile (v2/v3) in pip-licenses' shape:
    every installed package not marked ``dev`` (nor a workspace link)."""
    found = []
    for path, entry in (lock.get("packages") or {}).items():  # type: ignore[attr-defined]
        if not path or entry.get("dev") or entry.get("link"):
            continue
        license_ = entry.get("license", "")
        found.append(
            {
                "Name": path.rsplit("node_modules/", 1)[-1],
                "Version": entry.get("version", ""),
                "License": (
                    license_ if isinstance(license_, str) else json.dumps(license_)
                ),
            }
        )
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--npm-lock", type=Path, help="check this npm lockfile's runtime packages"
    )
    args = parser.parse_args(argv)
    if args.npm_lock:
        packages = npm_packages(json.loads(args.npm_lock.read_text(encoding="utf-8")))
    else:
        packages = json.load(sys.stdin)
    if not packages:
        # An empty report means pip-licenses saw no environment, not that
        # every license passed.
        print("No packages to check: is the environment synced?")
        return 1
    denied = [
        (p["Name"], p["Version"], " ".join(p["License"].split())[:80])
        for p in packages
        if (DENIED.search(p["License"]) or p["License"].strip() in UNLABELED)
        and not _allowed(p["Name"])
    ]
    print(f"Checked {len(packages)} packages.")
    for name, version, license_ in denied:
        print(f"DENIED {name} {version}: {license_}")
    return 1 if denied else 0


if __name__ == "__main__":
    sys.exit(main())
