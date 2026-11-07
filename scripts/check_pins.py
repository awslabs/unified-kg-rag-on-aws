# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report the hand-maintained pins that are behind upstream.

Dependabot does not move these (CONTRIBUTING.md, "Pins Dependabot does not
move"), so ``.github/workflows/pin-freshness.yml`` runs this weekly and opens
an issue with the report when one is behind:

- the data-plane base image (``ARG BASE_IMAGE`` in docker/Dockerfile): the
  digest its tag points to now, against the pinned one (a new digest is the
  same Python line with OS and Python patch updates);
- the uv image the Dockerfile copies uv from: the latest uv release, and
  whether the pinned tag still has the pinned digest;
- the ADOT collector sidecar (iac/stacks/compute_stack.py): the latest
  release, and whether the pinned tag still has the pinned digest;
- the AWS CDK CLI CI installs (``npm install -g aws-cdk@...`` in
  .github/workflows/quality.yml): the latest npm release.

Usage::

    python scripts/check_pins.py [--report report.md]

Exit status 0 when every pin is current, 1 when one is behind (the report
says which), 2 when a lookup failed. Image digests come from
``docker buildx imagetools inspect``; release versions from the GitHub API
(``GITHUB_TOKEN``, if set, raises its rate limit) and the npm registry.
``tests/unit/test_check_pins.py`` keeps the pin patterns matching the files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess  # nosec B404 - fixed argv, no shell
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
COMPUTE_STACK = REPO_ROOT / "iac" / "stacks" / "compute_stack.py"
QUALITY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "quality.yml"

_DIGEST = r"sha256:[0-9a-f]{64}"
_BASE_IMAGE = re.compile(
    rf"^ARG BASE_IMAGE=(?P<ref>[^@\s]+)@(?P<digest>{_DIGEST})$", re.M
)
_UV_IMAGE = re.compile(
    rf"^COPY --from=(?P<repo>ghcr\.io/astral-sh/uv):(?P<version>[0-9.]+)@(?P<digest>{_DIGEST}) ",
    re.M,
)
_COLLECTOR_VERSION = re.compile(r'^COLLECTOR_VERSION = "(?P<version>v[0-9.]+)"$', re.M)
_COLLECTOR_DIGEST = re.compile(rf'^\s+"(?P<digest>{_DIGEST})"$', re.M)
_COLLECTOR_REPOSITORY = "public.ecr.aws/aws-observability/aws-otel-collector"
_CDK_CLI = re.compile(r"npm install -g aws-cdk@(?P<version>[0-9.]+)")

UV_RELEASES = "https://api.github.com/repos/astral-sh/uv/releases/latest"
ADOT_RELEASES = (
    "https://api.github.com/repos/aws-observability/aws-otel-collector/releases/latest"
)
CDK_CLI_LATEST = "https://registry.npmjs.org/aws-cdk/latest"


@dataclass(frozen=True)
class Pins:
    base_ref: str
    base_digest: str
    uv_repo: str
    uv_version: str
    uv_digest: str
    collector_version: str
    collector_digest: str
    cdk_cli_version: str


def _match(pattern: re.Pattern[str], text: str, where: Path) -> re.Match[str]:
    found = pattern.search(text)
    if found is None:
        raise ValueError(
            f"{where.relative_to(REPO_ROOT)}: no pin matches {pattern.pattern}"
        )
    return found


def read_pins(root: Path = REPO_ROOT) -> Pins:
    """The pins as the repository states them."""
    dockerfile = (root / DOCKERFILE.relative_to(REPO_ROOT)).read_text(encoding="utf-8")
    compute = (root / COMPUTE_STACK.relative_to(REPO_ROOT)).read_text(encoding="utf-8")
    quality = (root / QUALITY_WORKFLOW.relative_to(REPO_ROOT)).read_text(
        encoding="utf-8"
    )
    base = _match(_BASE_IMAGE, dockerfile, DOCKERFILE)
    uv = _match(_UV_IMAGE, dockerfile, DOCKERFILE)
    return Pins(
        base_ref=base["ref"],
        base_digest=base["digest"],
        uv_repo=uv["repo"],
        uv_version=uv["version"],
        uv_digest=uv["digest"],
        collector_version=_match(_COLLECTOR_VERSION, compute, COMPUTE_STACK)["version"],
        collector_digest=_match(_COLLECTOR_DIGEST, compute, COMPUTE_STACK)["digest"],
        cdk_cli_version=_match(_CDK_CLI, quality, QUALITY_WORKFLOW)["version"],
    )


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text))


@dataclass(frozen=True)
class Lookups:
    """The upstream lookups (network calls), replaceable in tests."""

    digest: Callable[[str], str]
    latest_release: Callable[[str], str]
    latest_npm: Callable[[str], str]


def check(pins: Pins, lookups: Lookups) -> list[str]:
    """One report line per pin that is behind (empty: all current)."""
    behind = []
    digest = lookups.digest(pins.base_ref)
    if digest != pins.base_digest:
        behind.append(
            f"- Base image `{pins.base_ref}` now points to `{digest}` "
            f"(pinned: `{pins.base_digest}`) in docker/Dockerfile. Bump the "
            "digest, re-check the apt pins, and build both images."
        )
    latest_uv = lookups.latest_release(UV_RELEASES).lstrip("v")
    if _version(latest_uv) > _version(pins.uv_version):
        behind.append(
            f"- uv {latest_uv} is released (pinned: {pins.uv_version}, the image "
            "in docker/Dockerfile and `version:` of setup-uv in the workflows)."
        )
    uv_digest = lookups.digest(f"{pins.uv_repo}:{pins.uv_version}")
    if uv_digest != pins.uv_digest:
        behind.append(
            f"- `{pins.uv_repo}:{pins.uv_version}` now has digest `{uv_digest}` "
            f"(pinned: `{pins.uv_digest}`): the tag was re-pushed; check why "
            "before taking it."
        )
    latest_adot = lookups.latest_release(ADOT_RELEASES)
    if _version(latest_adot) > _version(pins.collector_version):
        behind.append(
            f"- ADOT collector {latest_adot} is released (pinned: "
            f"{pins.collector_version}, iac/stacks/compute_stack.py)."
        )
    collector = f"{_COLLECTOR_REPOSITORY}:{pins.collector_version}"
    collector_digest = lookups.digest(collector)
    if collector_digest != pins.collector_digest:
        behind.append(
            f"- `{collector}` now has digest `{collector_digest}` (pinned: "
            f"`{pins.collector_digest}`): the tag was re-pushed; check why "
            "before taking it."
        )
    latest_cdk = lookups.latest_npm(CDK_CLI_LATEST)
    if _version(latest_cdk) > _version(pins.cdk_cli_version):
        behind.append(
            f"- AWS CDK CLI {latest_cdk} is released (pinned: "
            f"{pins.cdk_cli_version}, .github/workflows/quality.yml and "
            "iac/README.md)."
        )
    return behind


def _image_digest(ref: str) -> str:
    """The digest ``ref`` (a tag) points to: its multi-arch index."""
    out = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            ref,
            "--format",
            "{{json .Manifest}}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    digest = json.loads(out)["digest"]
    if not re.fullmatch(_DIGEST, digest):
        raise ValueError(f"{ref}: unexpected digest {digest!r}")
    return str(digest)


def _get_json(url: str) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(
        request, timeout=60
    ) as response:  # nosec B310 - constant https URLs
        data: dict[str, object] = json.load(response)
    return data


def _latest_release(url: str) -> str:
    return str(_get_json(url)["tag_name"])


def _latest_npm(url: str) -> str:
    return str(_get_json(url)["version"])


NETWORK = Lookups(
    digest=_image_digest, latest_release=_latest_release, latest_npm=_latest_npm
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--report", type=Path, help="write the Markdown report here")
    args = parser.parse_args(argv)
    try:
        behind = check(read_pins(), NETWORK)
    except Exception as e:  # noqa: BLE001 - any failed lookup is exit 2
        print(f"check_pins: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    report = (
        "These pins are behind upstream (weekly check, "
        "`scripts/check_pins.py`):\n\n" + "\n".join(behind) + "\n"
        if behind
        else "Every pin is current.\n"
    )
    print(report, end="")
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    return 1 if behind else 0


if __name__ == "__main__":
    sys.exit(main())
