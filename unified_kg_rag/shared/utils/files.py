# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Crash-safe file writes."""

import os
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any


def temporary_sibling(path: Path) -> Path:
    """A uniquely named hidden file beside ``path`` for writing it atomically.

    The name ends in ``.tmp``, not the target's suffix, so a sync or a glob
    over ``*.json`` never picks up a half-written file.
    """
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


@contextmanager
def atomic_open(
    path: Path,
    mode: str = "w",
    *,
    fsync: bool = False,
    encoding: str | None = "utf-8",
    newline: str | None = None,
) -> Iterator[IO[Any]]:
    """Open a uniquely named sibling of ``path`` for writing (``mode`` ``w``
    or ``wb``) and rename it over ``path`` when the block ends, so a reader
    sees the old file or the new one, never a truncated one; a block that
    raises leaves the previous file and no temporary one. For a write that
    streams (an export, a download); ``atomic_write_text`` for a string.

    The file keeps what it is: a symlink stays one (the file it points to
    is replaced), a replaced file keeps its permission bits, and a target
    that is no regular file (``/dev/stdout``, a fifo) has no sibling to
    rename over it, so it is written in place as the block streams.
    ``fsync`` flushes the data to disk before the rename and the directory
    after it, for state that must survive a power loss, not just a crash.
    """
    if mode not in ("w", "wb"):
        raise ValueError(f"atomic_open writes a whole file ('w' or 'wb'), not {mode!r}")

    def opened(target: Path) -> IO[Any]:
        if mode == "wb":
            return target.open("wb")
        return target.open("w", encoding=encoding, newline=newline)

    try:
        current: os.stat_result | None = path.stat()
    except FileNotFoundError:  # no file yet, or a link to none
        current = None
    if current is not None and not stat.S_ISREG(current.st_mode):
        with opened(path) as handle:
            yield handle
        return
    target = path.resolve() if path.is_symlink() else path
    temp_path = temporary_sibling(target)
    try:
        with opened(temp_path) as handle:
            yield handle
            if fsync:
                handle.flush()
                os.fsync(handle.fileno())
        if current is not None:
            os.chmod(temp_path, stat.S_IMODE(current.st_mode))
        os.replace(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    if fsync:
        _fsync_directory(target.parent)


def _fsync_directory(directory: Path) -> None:
    """Flush ``directory``'s entries (a rename into it) to disk. POSIX
    only: Windows opens no directory to flush."""
    if os.name != "posix":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str, *, fsync: bool = False) -> None:
    """Write ``text`` to ``path`` so a reader sees the old file or the new one.

    The data goes to a uniquely named sibling first and is then renamed over
    ``path`` (atomic on POSIX and Windows), so a crash mid-write leaves the
    previous file intact instead of a truncated one, and a failed write leaves
    no temporary file behind. ``fsync`` flushes the data to disk before the
    rename (and the directory after it), for state that must survive a power
    loss, not just a crash. ``atomic_open`` says what a symlink, the file's
    mode and a target that is no regular file get.
    """
    with atomic_open(path, fsync=fsync) as handle:
        handle.write(text)
