# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""scripts/install_spacy_model.py, the retried and cached spaCy model install.

unstructured downloads the model in one attempt; a 502 from GitHub failed CI
runs. The script retries network errors, reuses a cached wheel, and never
caches or installs bytes without the pinned SHA-256. No network: the
download is replaced.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import urllib.error
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "install_spacy_model.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("install_spacy_model", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load()

WHEEL = b"a wheel's bytes"
PIN = script.ModelPin(
    name="en_core_web_sm",
    version="3.8.0",
    url="https://example.invalid/en_core_web_sm-3.8.0-py3-none-any.whl",
    sha256=hashlib.sha256(WHEEL).hexdigest(),
)


class _Opener:
    """Fails with the queued errors, then serves ``body``."""

    def __init__(self, *errors: Exception, body: bytes = WHEEL) -> None:
        self.errors = list(errors)
        self.body = body
        self.calls = 0

    def __call__(self, url: str, timeout: float) -> io.BytesIO:
        self.calls += 1
        assert url == PIN.url and timeout > 0
        if self.errors:
            raise self.errors.pop(0)
        return io.BytesIO(self.body)


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(PIN.url, code, "error", {}, None)  # type: ignore[arg-type]


def test_network_errors_and_5xx_are_retried_with_backoff(tmp_path: Path) -> None:
    opener = _Opener(_http_error(502), urllib.error.URLError("reset"), _http_error(504))
    slept: list[float] = []

    wheel = script.fetch_wheel(PIN, tmp_path, opener=opener, sleep=slept.append)

    assert wheel == tmp_path / PIN.wheel_name and wheel.read_bytes() == WHEEL
    assert opener.calls == 4
    assert slept == [5.0, 10.0, 20.0]
    assert not list(tmp_path.glob("*.part"))


def test_a_cached_wheel_is_reused_without_a_download(tmp_path: Path) -> None:
    (tmp_path / PIN.wheel_name).write_bytes(WHEEL)
    opener = _Opener()

    assert script.fetch_wheel(PIN, tmp_path, opener=opener).read_bytes() == WHEEL
    assert opener.calls == 0


def test_a_cached_wheel_with_another_digest_is_downloaded_again(
    tmp_path: Path,
) -> None:
    (tmp_path / PIN.wheel_name).write_bytes(b"truncated")
    opener = _Opener()

    assert script.fetch_wheel(PIN, tmp_path, opener=opener).read_bytes() == WHEEL
    assert opener.calls == 1


def test_a_download_with_another_digest_is_neither_cached_nor_retried(
    tmp_path: Path,
) -> None:
    opener = _Opener(body=b"other bytes")

    with pytest.raises(script.InstallError, match="SHA-256"):
        script.fetch_wheel(PIN, tmp_path, opener=opener, sleep=lambda _s: None)
    assert opener.calls == 1
    assert list(tmp_path.iterdir()) == []


def test_a_client_error_is_not_retried(tmp_path: Path) -> None:
    opener = _Opener(_http_error(404))

    with pytest.raises(script.InstallError, match="attempt 1"):
        script.fetch_wheel(PIN, tmp_path, opener=opener, sleep=lambda _s: None)
    assert opener.calls == 1


def test_the_last_failed_attempt_fails(tmp_path: Path) -> None:
    opener = _Opener(*(_http_error(503) for _ in range(script.ATTEMPTS)))

    with pytest.raises(script.InstallError, match=f"attempt {script.ATTEMPTS}"):
        script.fetch_wheel(PIN, tmp_path, opener=opener, sleep=lambda _s: None)
    assert opener.calls == script.ATTEMPTS
    assert list(tmp_path.iterdir()) == []
