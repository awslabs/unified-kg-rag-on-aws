# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The digest of the ``unified_kg_rag`` package files, to prove an image
holds the source it was built from.

uv caches the wheel it builds of the project itself and keys it by the
project's metadata, not its source: an image built after a source change,
with the version unchanged and the build cache warm, installed the earlier
wheel. The Dockerfile reinstalls the project on every build and checks the
installed package against the copied source with this digest, and bakes the
digest into the image (``/app/.venv/unified-kg-rag-source.sha256``); the CI
docker job compares it with the checkout's.

Usage::

    python scripts/package_digest.py [PACKAGE_DIR | --installed]

prints the SHA-256 over every file under ``PACKAGE_DIR`` (default: the
checkout's ``unified_kg_rag``) or, with ``--installed``, under the package
the running interpreter imports, by relative path and content; bytecode
(``__pycache__``, ``*.pyc``) is left out on both sides. Run it with ``-I`` so
the working directory's source is not the package found.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

PACKAGE = "unified_kg_rag"


def package_digest(root: Path) -> str:
    """SHA-256 over the files under ``root``, bytecode aside."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        digest.update(relative.as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def installed_root() -> Path:
    """The directory of the package the interpreter imports (not run)."""
    spec = importlib.util.find_spec(PACKAGE)
    if spec is None or spec.origin is None:
        raise SystemExit(f"{PACKAGE} is not installed")
    return Path(spec.origin).parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "package_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / PACKAGE,
    )
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args(argv)
    print(package_digest(installed_root() if args.installed else args.package_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
