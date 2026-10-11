# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail on copyleft, source-available or unlabeled dependency licenses.

Reads ``pip-licenses --format=json`` output on stdin (run it in an environment
synced with every extra) and exits non-zero when a package's license matches
DENIED and the package is not on ALLOWED. Each ALLOWED entry names why it is
acceptable; add one only after reviewing the license.
"""

from __future__ import annotations

import json
import re
import sys

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
    "gensim": "LGPL-2.1, unmodified dynamic dependency of graspologic",
    "certifi": "MPL-2.0 (weak, file-level copyleft), unmodified",
    "orjson": "MPL-2.0 AND (Apache-2.0 OR MIT), unmodified",
    "tqdm": "MPL-2.0 AND MIT, unmodified",
    "nvidia-": "NVIDIA CUDA runtime libraries, pulled in by torch in [local]",
    "cuda-": "NVIDIA CUDA toolkit metapackage, pulled in by torch in [local]",
}


def _allowed(name: str) -> bool:
    name = name.lower()
    return any(
        name.startswith(key) if key.endswith("-") else name == key for key in ALLOWED
    )


def main() -> int:
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
