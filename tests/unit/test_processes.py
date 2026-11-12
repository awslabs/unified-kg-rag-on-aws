# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whether the process an identity names is gone (``shared.utils.processes``),
and the local registry's lease taken at once from a killed run (rv24
quality 6: a run killed with SIGKILL locked its namespace for 600 s)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from unified_kg_rag.adapters.doc_status.sqlite import SQLiteDocStatusStore
from unified_kg_rag.application.ingestion.namespace_lease import NamespaceLease
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared.utils.processes import process_gone, process_identity

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(os.name != "posix", reason="POSIX processes"),
]

_IDENTITY = (
    "from unified_kg_rag.shared.utils.processes import process_identity; "
    "print(process_identity(), flush=True)"
)


def _exited_identity() -> str:
    """The identity of a process that has exited."""
    done = subprocess.run(
        [sys.executable, "-c", _IDENTITY], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def test_this_process_is_not_gone() -> None:
    assert not process_gone(process_identity())
    assert not process_gone(f"{process_identity()}:a1b2c3d4")


def test_an_exited_process_is_gone() -> None:
    assert process_gone(_exited_identity())


def test_a_reused_pid_is_gone() -> None:
    host, pid, started, namespace = process_identity().split(":")
    if not started:
        pytest.skip("no /proc start times on this platform")
    # This pid, but started at another time: a later process took it.
    assert process_gone(f"{host}:{pid}:{int(started) + 1}:{namespace}")


@pytest.mark.parametrize(
    "identity",
    [
        "run-a",
        "another-host:1:2:3:uuid",
        # Its owner format before process identities: host, pid, uuid.
        "{host}:{pid}:a1b2c3d4",
    ],
)
def test_what_it_cannot_judge_is_alive(identity: str) -> None:
    host, pid, *_ = process_identity().split(":")
    assert not process_gone(identity.format(host=host, pid=pid))


def test_another_pid_namespace_is_alive() -> None:
    host, _pid, started, namespace = _exited_identity().split(":")
    if not namespace:
        pytest.skip("no /proc pid namespaces on this platform")
    assert not process_gone(f"{host}:{_pid}:{started}:{int(namespace) + 1}")


def test_the_sqlite_lease_of_a_killed_run_is_taken_at_once(tmp_path: Path) -> None:
    path = tmp_path / "doc-status.db"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\n"
            "from pathlib import Path\n"
            "from unified_kg_rag.adapters.doc_status.sqlite import "
            "SQLiteDocStatusStore\n"
            "from unified_kg_rag.application.ingestion.namespace_lease import "
            "NamespaceLease\n"
            "from unified_kg_rag.domain.models import Config\n"
            "store = SQLiteDocStatusStore(Config(), Path(sys.argv[1]))\n"
            "with NamespaceLease(store, 'ns', 600):\n"
            "    print('held', flush=True)\n"
            "    time.sleep(120)\n",
            str(path),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline() == "held\n"
        store = SQLiteDocStatusStore(Config(), path)
        owner, expires = store.lease_holder("ns") or ("", 0.0)
        assert expires > time.time() + 500
        # Alive: refused.
        assert not store.try_acquire_lease("ns", "next-run", 60)
        holder.send_signal(signal.SIGKILL)
        holder.wait(30)
        with NamespaceLease(
            store, "ns", 60, in_process_stores=True, in_process_write_seconds=0
        ) as lease:
            assert store.lease_holder("ns")[0] == lease.owner != owner
            # Taken over as if it expired now: no write of the killed run is
            # left (its stores were in its process), bar the margin.
            assert lease.writes_from <= time.time() + 6
        store.close()
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(30)
