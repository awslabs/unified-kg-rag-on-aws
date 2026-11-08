# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install the spaCy model the ``unstructured`` extra tokenizes with.

unstructured downloads ``en_core_web_sm`` from GitHub releases at its first
Markdown or HTML file, in one attempt: a private VPC cannot reach GitHub, and
a 502 from it failed CI runs. This installs the same wheel ahead of time,
with retries and a reusable cache:

- the name, version, URL and SHA-256 come from unstructured itself
  (``unstructured.nlp.tokenize``), so they follow the locked unstructured;
- a wheel already in ``--cache-dir`` with the pinned SHA-256 is reused (the
  CI cache and the Docker build cache keep it there), else it is downloaded,
  retrying network errors and 5xx responses with backoff;
- the wheel is checked against the pinned SHA-256 before it is cached or
  installed, and installed into the running interpreter's environment.

Usage::

    python scripts/install_spacy_model.py [--cache-dir DIR] [--print-version]

``--print-version`` prints the pinned version and exits (the CI cache key).
Exit status 0 when the model is installed, 1 when it could not be.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.metadata
import importlib.util
import os
import sys
import sysconfig
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

ATTEMPTS = 5
TIMEOUT_SECONDS = 120
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "unified-kg-rag" / "spacy-models"


class InstallError(Exception):
    """The model could not be downloaded, verified or installed."""


@dataclass(frozen=True)
class ModelPin:
    """The model wheel unstructured pins."""

    name: str
    version: str
    url: str
    sha256: str

    @property
    def wheel_name(self) -> str:
        return f"{self.name}-{self.version}-py3-none-any.whl"


def pinned_model() -> ModelPin:
    """The wheel the installed unstructured would download.

    Raises:
        InstallError: unstructured is missing or no longer pins the model
            where this reads it.
    """
    try:
        from unstructured.nlp import tokenize
    except ImportError as exc:
        raise InstallError(f"unstructured is not installed: {exc}") from exc
    try:
        return ModelPin(
            name=tokenize._SPACY_MODEL_NAME,
            version=tokenize._SPACY_MODEL_VERSION,
            url=tokenize._SPACY_MODEL_URL,
            sha256=tokenize._SPACY_MODEL_SHA256,
        )
    except AttributeError as exc:
        raise InstallError(
            "unstructured.nlp.tokenize no longer names the spaCy model it "
            f"downloads ({exc}); update scripts/install_spacy_model.py"
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500 or exc.code == 429
    return isinstance(exc, (OSError, http.client.HTTPException))


def _open(url: str, timeout: float) -> BinaryIO:
    # A fixed https URL that unstructured pins, not user input.
    return urllib.request.urlopen(url, timeout=timeout)  # nosec B310  # noqa: S310


def fetch_wheel(
    pin: ModelPin,
    cache_dir: Path,
    *,
    opener: Callable[[str, float], BinaryIO] = _open,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = ATTEMPTS,
) -> Path:
    """The verified wheel in ``cache_dir``, downloaded when it is not there.

    Raises:
        InstallError: Every attempt failed, a response was a client error,
            or the download does not have the pinned SHA-256.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    wheel = cache_dir / pin.wheel_name
    if wheel.is_file():
        if _sha256(wheel) == pin.sha256:
            print(f"Using the cached {wheel.name}")
            return wheel
        print(f"Discarding {wheel.name} from the cache: wrong SHA-256")
        wheel.unlink()
    for attempt in range(1, attempts + 1):
        fd, partial = tempfile.mkstemp(dir=cache_dir, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as out, opener(pin.url, TIMEOUT_SECONDS) as resp:
                for block in iter(lambda: resp.read(1 << 20), b""):
                    out.write(block)
        except Exception as exc:  # noqa: BLE001 - classified just below
            Path(partial).unlink(missing_ok=True)
            if not _retryable(exc) or attempt == attempts:
                raise InstallError(
                    f"Could not download {pin.url} (attempt {attempt}): {exc}"
                ) from exc
            delay = 5.0 * 2 ** (attempt - 1)
            print(
                f"Download attempt {attempt} failed ({exc}); retrying in {delay:.0f} s"
            )
            sleep(delay)
            continue
        actual = _sha256(Path(partial))
        if actual != pin.sha256:
            Path(partial).unlink(missing_ok=True)
            # A complete response with other bytes is not a network error:
            # never retry it into the cache.
            raise InstallError(
                f"{pin.wheel_name} has SHA-256 {actual}, expected {pin.sha256}"
            )
        os.chmod(partial, 0o644)  # nosec B103 - a public wheel
        os.replace(partial, wheel)
        return wheel
    raise InstallError(f"Could not download {pin.url}")  # pragma: no cover


def installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def install_wheel(wheel: Path) -> None:
    """Install ``wheel`` into this interpreter's environment."""
    from installer import install
    from installer.destinations import SchemeDictionaryDestination
    from installer.sources import WheelFile
    from installer.utils import get_launcher_kind

    destination = SchemeDictionaryDestination(
        sysconfig.get_paths(),
        interpreter=sys.executable,
        script_kind=get_launcher_kind(),
    )
    with WheelFile.open(wheel) as source:
        install(source=source, destination=destination, additional_metadata={})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--print-version", action="store_true")
    args = parser.parse_args(argv)
    try:
        pin = pinned_model()
        if args.print_version:
            print(pin.version)
            return 0
        if installed_version(pin.name) == pin.version:
            print(f"{pin.name} {pin.version} is already installed")
            return 0
        install_wheel(fetch_wheel(pin, args.cache_dir))
        importlib.invalidate_caches()
        if importlib.util.find_spec(pin.name) is None:
            raise InstallError(f"{pin.name} is not importable after installing it")
    except InstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Installed {pin.name} {pin.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
