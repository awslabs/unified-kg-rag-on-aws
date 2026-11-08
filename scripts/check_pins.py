# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report the hand-maintained pins that are behind upstream.

Dependabot does not move these (CONTRIBUTING.md, "Pins Dependabot does not
move"), so ``.github/workflows/pin-freshness.yml`` runs this weekly and opens
an issue with the report when one is behind:

- the data-plane base image (``ARG BASE_IMAGE`` in docker/Dockerfile): the
  digest its tag points to now, against the pinned one (a new digest is the
  same Python line with OS and Python patch updates). A new index whose
  images for the platforms the image builds for are the pinned ones (an
  attestation re-pushed, say) is not reported;
- the uv image the Dockerfile copies uv from: the latest uv release, and
  whether the pinned tag still has the pinned digest;
- the ADOT collector sidecar (iac/stacks/compute_stack.py): the latest
  release, and whether the pinned tag still has the pinned digest;
- the AWS CDK CLI CI installs (``npm install -g aws-cdk@...`` in
  .github/workflows/quality.yml): the latest npm release;
- the Trivy image the image scans run (``aquasec/trivy:...@sha256:...`` in
  .github/workflows/quality.yml, one pin for every job): the latest release,
  and whether the pinned tag still has the pinned digest;
- the apt pins of docker/Dockerfile (``libgomp1=...``, ``zlib1g=...``): the
  version the base image's Debian repositories now offer (a superseded
  version leaves the repository and the build fails), and whether the base
  image already ships the pinned version (the pin can go).

Usage::

    python scripts/check_pins.py [--report report.md]

Exit status 0 when every pin is current, 1 when one is behind (the report
says which), 2 when a lookup failed. A release past a pin held on purpose
(``HELD``) is listed in the report as held and does not count as behind. Image digests come from
``docker buildx imagetools inspect``; release versions from the GitHub API
(``GITHUB_TOKEN``, if set, raises its rate limit) and the npm registry; the
apt versions from ``apt-cache policy`` in the pinned base image (``docker
run``).
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
_TRIVY_IMAGE = re.compile(
    rf"(?P<repo>aquasec/trivy):(?P<version>[0-9.]+)@(?P<digest>{_DIGEST})"
)
# A package of the runtime stage's `apt-get install` line, pinned: one per
# line, `        name=version \`.
_APT_PIN = re.compile(
    r"^\s+(?P<package>[a-z0-9][a-z0-9.+-]*)=(?P<version>\S+) \\$", re.M
)

# The platforms docker/Dockerfile builds for (Fargate x86_64 and Graviton).
IMAGE_PLATFORMS = ("linux/amd64", "linux/arm64")

UV_RELEASES = "https://api.github.com/repos/astral-sh/uv/releases/latest"
ADOT_RELEASES = (
    "https://api.github.com/repos/aws-observability/aws-otel-collector/releases/latest"
)
CDK_CLI_LATEST = "https://registry.npmjs.org/aws-cdk/latest"
TRIVY_RELEASES = "https://api.github.com/repos/aquasecurity/trivy/releases/latest"


@dataclass(frozen=True)
class Held:
    """A pin kept on a release series on purpose, with the reason."""

    series: str
    reason: str


# Releases past these series are reported as held, not behind (exit 0).
# Lift a hold by deleting its entry (CONTRIBUTING.md, "Pins Dependabot does
# not move").
HELD: dict[str, Held] = {
    "uv": Held(
        series="0.12",
        reason="0.13.0 has breaking changes (Python 3.15 as the default, the "
        "tar codec, constraint-file parsing) to check against the locks, the "
        "workflows and the image",
    ),
}


def _in_series(version: str, series: str) -> bool:
    return _version(version)[: len(_version(series))] == _version(series)


@dataclass(frozen=True)
class Report:
    """The pins behind upstream, and the releases past a held pin."""

    behind: list[str]
    held: list[str]


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
    trivy_repo: str
    trivy_version: str
    trivy_digest: str
    # (package, version) of each apt pin, in the Dockerfile's order.
    apt_pins: tuple[tuple[str, str], ...]


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
    trivy = _match(_TRIVY_IMAGE, quality, QUALITY_WORKFLOW)
    others = {m.group(0) for m in _TRIVY_IMAGE.finditer(quality)} - {trivy.group(0)}
    if others:
        raise ValueError(
            f"{QUALITY_WORKFLOW.relative_to(REPO_ROOT)}: the image scans pin "
            f"different Trivy images: {sorted({trivy.group(0), *others})}"
        )
    apt = tuple((m["package"], m["version"]) for m in _APT_PIN.finditer(dockerfile))
    if not apt:
        raise ValueError(
            f"{DOCKERFILE.relative_to(REPO_ROOT)}: no pin matches {_APT_PIN.pattern}"
        )
    return Pins(
        base_ref=base["ref"],
        base_digest=base["digest"],
        uv_repo=uv["repo"],
        uv_version=uv["version"],
        uv_digest=uv["digest"],
        collector_version=_match(_COLLECTOR_VERSION, compute, COMPUTE_STACK)["version"],
        collector_digest=_match(_COLLECTOR_DIGEST, compute, COMPUTE_STACK)["digest"],
        cdk_cli_version=_match(_CDK_CLI, quality, QUALITY_WORKFLOW)["version"],
        trivy_repo=trivy["repo"],
        trivy_version=trivy["version"],
        trivy_digest=trivy["digest"],
        apt_pins=apt,
    )


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text))


@dataclass(frozen=True)
class Lookups:
    """The upstream lookups (network calls), replaceable in tests."""

    digest: Callable[[str], str]
    # ref -> {"os/arch": digest} of the images an index lists.
    platform_digests: Callable[[str], dict[str, str]]
    latest_release: Callable[[str], str]
    latest_npm: Callable[[str], str]
    # (image, packages) -> {package: (installed, candidate)} as apt-cache
    # policy reports them in that image ("(none)": not installed).
    apt_policy: Callable[[str, tuple[str, ...]], dict[str, tuple[str, str]]]


def _same_images(pins: Pins, lookups: Lookups) -> bool:
    """Whether the base tag's index now lists the pinned images for every
    platform the image builds for."""
    pinned = lookups.platform_digests(f"{pins.base_ref}@{pins.base_digest}")
    current = lookups.platform_digests(pins.base_ref)
    return all(
        pinned.get(platform) is not None
        and pinned.get(platform) == current.get(platform)
        for platform in IMAGE_PLATFORMS
    )


def check(pins: Pins, lookups: Lookups) -> Report:
    """One report line per pin that is behind, and per held release."""
    behind: list[str] = []
    held: list[str] = []
    digest = lookups.digest(pins.base_ref)
    if digest != pins.base_digest and not _same_images(pins, lookups):
        behind.append(
            f"- Base image `{pins.base_ref}` now points to `{digest}` "
            f"(pinned: `{pins.base_digest}`) in docker/Dockerfile. Bump the "
            "digest, re-check the apt pins, and build both images."
        )
    latest_uv = lookups.latest_release(UV_RELEASES).lstrip("v")
    uv_hold = HELD.get("uv")
    if _version(latest_uv) <= _version(pins.uv_version):
        pass
    elif uv_hold is not None and not _in_series(latest_uv, uv_hold.series):
        held.append(
            f"- uv {latest_uv} is released; held at {uv_hold.series} "
            f"(pinned: {pins.uv_version}): {uv_hold.reason}."
        )
    else:
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
    latest_trivy = lookups.latest_release(TRIVY_RELEASES).lstrip("v")
    if _version(latest_trivy) > _version(pins.trivy_version):
        behind.append(
            f"- Trivy {latest_trivy} is released (pinned: {pins.trivy_version}, "
            "the image scans in .github/workflows/quality.yml and "
            ".github/workflows/pin-freshness.yml reads it from there)."
        )
    trivy = f"{pins.trivy_repo}:{pins.trivy_version}"
    trivy_digest = lookups.digest(trivy)
    if trivy_digest != pins.trivy_digest:
        behind.append(
            f"- `{trivy}` now has digest `{trivy_digest}` (pinned: "
            f"`{pins.trivy_digest}`): the tag was re-pushed; check why before "
            "taking it."
        )
    base = f"{pins.base_ref}@{pins.base_digest}"
    policy = lookups.apt_policy(base, tuple(name for name, _ in pins.apt_pins))
    for name, pinned in pins.apt_pins:
        installed, candidate = policy[name]
        if candidate != pinned:
            behind.append(
                f"- apt `{name}`: the base image's repositories now offer "
                f"`{candidate}` (pinned: `{pinned}`, docker/Dockerfile). Bump "
                "the pin: a superseded version leaves the repository and the "
                "image build fails."
            )
        elif installed == pinned:
            behind.append(
                f"- apt `{name}`: the base image already ships the pinned "
                f"`{pinned}` (a security upgrade the base digest now "
                "includes): drop it from docker/Dockerfile's apt-get line."
            )
    return Report(behind=behind, held=held)


def _manifest(ref: str) -> dict[str, object]:
    """``ref``'s manifest (a multi-arch index for a tag) as buildx reads it."""
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
    manifest: dict[str, object] = json.loads(out)
    return manifest


def _image_digest(ref: str) -> str:
    """The digest ``ref`` (a tag) points to: its multi-arch index."""
    digest = str(_manifest(ref)["digest"])
    if not re.fullmatch(_DIGEST, digest):
        raise ValueError(f"{ref}: unexpected digest {digest!r}")
    return digest


def platform_digests(manifest: dict[str, object]) -> dict[str, str]:
    """{"os/arch[/variant]": digest} of the images an index lists, without
    its attestation manifests (platform ``unknown/unknown``). arm64's
    default variant (v8) is left out, as Docker names the platform."""
    found: dict[str, str] = {}
    for entry in manifest.get("manifests") or []:  # type: ignore[attr-defined]
        platform = entry.get("platform") or {}
        if platform.get("os", "unknown") == "unknown":
            continue
        variant = platform.get("variant")
        if platform["architecture"] == "arm64" and variant == "v8":
            variant = None
        key = "/".join(
            part for part in (platform["os"], platform["architecture"], variant) if part
        )
        found[key] = entry["digest"]
    return found


def _platform_digests(ref: str) -> dict[str, str]:
    return platform_digests(_manifest(ref))


def _apt_policy(image: str, packages: tuple[str, ...]) -> dict[str, tuple[str, str]]:
    """``apt-cache policy`` of ``packages`` in ``image``, after an update:
    {package: (installed, candidate)}."""
    out = subprocess.run(  # nosec B603 B607 - fixed argv, no shell on the host
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "sh",
            image,
            "-c",
            'apt-get update -qq > /dev/null && apt-cache policy "$@"',
            "sh",
            *packages,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    ).stdout
    return parse_apt_policy(out, packages)


def parse_apt_policy(
    text: str, packages: tuple[str, ...]
) -> dict[str, tuple[str, str]]:
    """{package: (installed, candidate)} from ``apt-cache policy`` output."""
    found: dict[str, tuple[str, str]] = {}
    for block in re.split(r"^(?=\S+:$)", text, flags=re.M):
        header = re.match(r"^(\S+):$", block, re.M)
        installed = re.search(r"^\s+Installed: (\S+)$", block, re.M)
        candidate = re.search(r"^\s+Candidate: (\S+)$", block, re.M)
        if header and installed and candidate:
            found[header.group(1)] = (installed.group(1), candidate.group(1))
    missing = [p for p in packages if p not in found]
    if missing:
        raise ValueError(f"apt-cache policy reported no candidate for {missing}")
    return found


_ALLOWED_URLS = frozenset({UV_RELEASES, ADOT_RELEASES, CDK_CLI_LATEST, TRIVY_RELEASES})


def _get_json(url: str) -> dict[str, object]:
    if url not in _ALLOWED_URLS:
        raise ValueError(f"refusing to fetch {url!r}: not one of the pinned sources")
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        request.add_header("Authorization", f"Bearer {token}")
    # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected - only the allowlisted https URLs
    with urllib.request.urlopen(  # nosec B310 - only the allowlisted https URLs
        request, timeout=60
    ) as response:
        data: dict[str, object] = json.load(response)
    return data


def _latest_release(url: str) -> str:
    return str(_get_json(url)["tag_name"])


def _latest_npm(url: str) -> str:
    return str(_get_json(url)["version"])


NETWORK = Lookups(
    digest=_image_digest,
    platform_digests=_platform_digests,
    latest_release=_latest_release,
    latest_npm=_latest_npm,
    apt_policy=_apt_policy,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--report", type=Path, help="write the Markdown report here")
    args = parser.parse_args(argv)
    try:
        result = check(read_pins(), NETWORK)
    except Exception as e:  # noqa: BLE001 - any failed lookup is exit 2
        print(f"check_pins: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    report = render(result)
    print(report, end="")
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    return 1 if result.behind else 0


def render(result: Report) -> str:
    """The Markdown report: what is behind, then what is held."""
    report = (
        "These pins are behind upstream (weekly check, "
        "`scripts/check_pins.py`):\n\n" + "\n".join(result.behind) + "\n"
        if result.behind
        else "Every pin is current.\n"
    )
    if result.held:
        report += (
            "\nHeld on purpose (`HELD` in `scripts/check_pins.py`; not counted "
            "as behind):\n\n" + "\n".join(result.held) + "\n"
        )
    return report


if __name__ == "__main__":
    sys.exit(main())
