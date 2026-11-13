# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which process holds something, and whether that process is gone.

A process's identity is its host, its pid and, on Linux, its start time
and pid namespace (``/proc``): a pid reused by a later process starts at
another time, and a pid of another namespace (another container sharing
the host's name) is not ours to judge. :func:`process_gone` answers only
for a process of this host and namespace; any other is taken as alive.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

__all__ = ["process_gone", "process_identity"]


def _start_time(pid: int) -> str:
    """When ``pid`` started (clock ticks since boot, ``/proc/<pid>/stat``
    field 22), or ``""`` where ``/proc`` does not tell (not Linux, or no
    such process)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return ""
    # The command name (field 2) is in parentheses and may hold spaces.
    fields = stat[stat.rfind(")") + 2 :].split()
    return fields[19] if len(fields) > 19 else ""


def _pid_namespace() -> str:
    """This process's pid namespace (its ``/proc/self/ns/pid`` inode), or
    ``""`` where ``/proc`` does not tell."""
    try:
        return os.readlink("/proc/self/ns/pid").strip("pid:[]")
    except OSError:
        return ""


def process_identity(pid: int | None = None) -> str:
    """This process (or the process ``pid`` of this host and pid namespace,
    a child it started) as ``host:pid:start:namespace`` (the last two empty
    where ``/proc`` does not tell); no field holds a colon."""
    pid = os.getpid() if pid is None else pid
    host = socket.gethostname().replace(":", "-")
    return f"{host}:{pid}:{_start_time(pid)}:{_pid_namespace()}"


def process_gone(identity: str) -> bool:
    """Whether the process ``identity`` names (:func:`process_identity`,
    optionally followed by more ``:`` fields) is known to have exited: it
    ran on this host, in this pid namespace, and its pid is free or held by
    a process that started at another time. ``False`` for anything it
    cannot judge (another host or namespace, another format, no POSIX)."""
    parts = identity.split(":")
    if len(parts) < 4 or os.name != "posix":
        return False
    host, pid_text, started, namespace = parts[:4]
    if host != socket.gethostname().replace(":", "-") or not pid_text.isdigit():
        return False
    if namespace != _pid_namespace():
        return False
    pid = int(pid_text)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        pass  # alive, another user's
    return bool(started) and _start_time(pid) not in ("", started)
