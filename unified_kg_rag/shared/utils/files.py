# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Crash-safe file writes."""

import os
import uuid
from pathlib import Path


def temporary_sibling(path: Path) -> Path:
    """A uniquely named hidden file beside ``path`` for writing it atomically.

    The name ends in ``.tmp``, not the target's suffix, so a sync or a glob
    over ``*.json`` never picks up a half-written file.
    """
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def atomic_write_text(path: Path, text: str, *, fsync: bool = False) -> None:
    """Write ``text`` to ``path`` so a reader sees the old file or the new one.

    The data goes to a uniquely named sibling first and is then renamed over
    ``path`` (atomic on POSIX and Windows), so a crash mid-write leaves the
    previous file intact instead of a truncated one, and a failed write leaves
    no temporary file behind. ``fsync`` flushes the data to disk before the
    rename, for state that must survive a power loss, not just a crash.
    """
    temp_path = temporary_sibling(path)
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            if fsync:
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
