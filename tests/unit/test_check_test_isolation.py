# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""scripts/check_test_isolation.py fails a test file that passes only after
another file ran in its process, and CI runs it over tests/unit."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_test_isolation.py"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_test_isolation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.timeout(60)
def test_a_file_that_needs_another_first_fails(tmp_path: Path) -> None:
    script = _script()
    (tmp_path / "test_a_sets.py").write_text(
        "import builtins\n\n\ndef test_sets():\n    builtins.SHARED = 1\n"
    )
    # Passes after test_a_sets.py in one process, fails alone.
    (tmp_path / "test_b_needs.py").write_text(
        "import builtins\n\n\ndef test_needs():\n    assert builtins.SHARED == 1\n"
    )
    (tmp_path / "test_c_skips.py").write_text(
        "import pytest\n\npytest.importorskip('no_such_module_anywhere')\n"
    )
    failed = script.failures([tmp_path], jobs=3)
    assert [(path.name, code) for path, code, _ in failed] == [("test_b_needs.py", 1)]
    assert script.main([str(tmp_path), "--jobs", "2"]) == 1


@pytest.mark.timeout(60)
def test_a_file_past_the_timeout_fails_named(tmp_path: Path) -> None:
    # A hung file is stopped and named, not left to run the job out.
    script = _script()
    (tmp_path / "test_hangs.py").write_text(
        "import time\n\n\ndef test_hangs():\n    print('started')\n"
        "    time.sleep(120)\n"
    )
    (tmp_path / "test_quick.py").write_text("def test_quick():\n    pass\n")
    failed = script.failures([tmp_path], jobs=2, timeout=8)
    assert [(path.name, code) for path, code, _ in failed] == [
        ("test_hangs.py", script.TIMED_OUT)
    ]
    assert failed[0][2].endswith("timed out after 8 s")


@pytest.mark.timeout(60)
@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_a_timed_out_file_s_children_are_stopped_with_it(tmp_path: Path) -> None:
    # A hung test that started a server or a worker leaves it orphaned (and
    # holding the output pipe) unless its whole process group is killed.
    script = _script()
    pid_file = tmp_path / "child.pid"
    (tmp_path / "test_spawns.py").write_text(
        "import subprocess, time\nfrom pathlib import Path\n\n\n"
        "def test_spawns():\n"
        "    child = subprocess.Popen(['sleep', '300'])\n"
        f"    Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        "    time.sleep(120)\n"
    )
    started = time.monotonic()
    failed = script.failures([tmp_path], jobs=1, timeout=6)
    assert [code for _, code, _ in failed] == [script.TIMED_OUT]
    assert time.monotonic() - started < 30
    child = int(pid_file.read_text())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(child, 9)
        pytest.fail("the timed-out file's child outlived it")


@pytest.mark.timeout(90)
@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc")
def test_a_grandchild_in_its_own_session_does_not_hang_the_check(
    tmp_path: Path,
) -> None:
    # A worker started in a session of its own (start_new_session=True)
    # is in no group the kill reaches: it lives on, and while it holds the
    # file's output pipe, reading the killed file's output to the end
    # would wait for it. The read after the kill is bounded instead.
    script = _script()
    pid_file = tmp_path / "grandchild.pid"
    (tmp_path / "test_detaches.py").write_text(
        "import os, subprocess, time\nfrom pathlib import Path\n\n\n"
        "def test_detaches():\n"
        "    # The pipes pytest's capture saved: the checker's output pipes.\n"
        "    held = []\n"
        "    for fd in os.listdir('/proc/self/fd'):\n"
        "        try:\n"
        "            if os.readlink(f'/proc/self/fd/{fd}').startswith('pipe:'):\n"
        "                held.append(int(fd))\n"
        "        except OSError:\n"
        "            pass\n"
        "    child = subprocess.Popen(['sleep', '45'], stdout=held[0],\n"
        "                             pass_fds=held, start_new_session=True)\n"
        f"    Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        "    time.sleep(120)\n"
    )
    started = time.monotonic()
    try:
        failed = script.failures([tmp_path], jobs=1, timeout=6)
        assert [code for _, code, _ in failed] == [script.TIMED_OUT]
        assert time.monotonic() - started < 30
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), 9)
            except ProcessLookupError:
                pass


def test_ci_runs_it_over_every_test_directory() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8")
    )
    runs = [step.get("run", "") for step in workflow["jobs"]["test-isolation"]["steps"]]
    assert any(
        "scripts/check_test_isolation.py tests/unit tests/integration tests/property"
        in run
        for run in runs
    )
