# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Crash-safe file writes."""

import errno
import logging
import os
import re
import shutil
import stat
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)


# The longest file name, in bytes, of the file systems written to (ext4,
# XFS, APFS, NTFS's 255 UTF-16 units hold at least as many characters).
NAME_MAX = 255


def temporary_sibling(path: Path) -> Path:
    """A uniquely named hidden file beside ``path`` for writing it atomically.

    The name ends in ``.tmp``, not the target's suffix, so a sync or a glob
    over ``*.json`` never picks up a half-written file. It begins with as
    much of ``path``'s name as fits :data:`NAME_MAX` bytes with the rest,
    cut at a character boundary: a valid long name (80 Korean characters
    are 240 bytes) gets a valid temporary name.
    """
    unique = f".{uuid.uuid4().hex}.tmp"
    room = NAME_MAX - 1 - len(unique)
    name = path.name
    while len(os.fsencode(name)) > room:
        name = name[:-1]
    return path.with_name(f".{name}{unique}")


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
    is replaced), and a replaced file keeps its permission bits (it is a new
    file: a hard link to the old one keeps the old text, and the writer owns
    it). What has no sibling to rename over it is written in place as the
    block streams: a target that is no regular file (a fifo, ``/dev/null``),
    and a descriptor's alias (``/dev/stdout``, ``/dev/fd/N``,
    ``/proc/self/fd/N``, or a link into them), which is written through the
    descriptor even when it is open on a regular file (``cmd > out.txt``):
    a rename would unlink the file the shell writes to. A file whose
    sibling cannot be made or renamed over it (a writable file in a
    read-only directory, a bind-mounted file) is staged in a temporary file
    (beside it, or in :func:`tempfile.gettempdir` when its directory refuses
    one) and copied into it when the block ends: a block that raises still
    leaves it as it was, but a reader may see the copy half made.
    ``fsync`` flushes the data to disk before the rename and the directory
    after it (a copied-in file, once copied), for state that must survive
    a power loss, not just a crash; a directory that cannot be flushed (no
    read permission) is logged, not raised: the file is written by then.
    """
    if mode not in ("w", "wb"):
        raise ValueError(f"atomic_open writes a whole file ('w' or 'wb'), not {mode!r}")

    def opened(target: Path | int) -> IO[Any]:
        if mode == "wb":
            return open(target, "wb")
        return open(target, "w", encoding=encoding, newline=newline)

    alias = _descriptor_alias(path)
    if alias is not None:
        (fd,) = alias
        if fd is None:  # another process's: what it is open on, in place
            with opened(path) as handle:
                yield handle
        else:
            with _written_through(path, fd, opened) as handle:
                yield handle
        return
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
    beside = True
    try:
        temp = opened(temp_path)
    except PermissionError:
        if current is None:
            raise
        logger.debug("cannot create %s; staging %s elsewhere", temp_path, target)
        temp_path = temporary_sibling(Path(tempfile.gettempdir()) / target.name)
        temp = opened(temp_path)
        beside = False
    try:
        with temp as handle:
            yield handle
            if fsync and beside:
                handle.flush()
                os.fsync(handle.fileno())
        if not beside:
            _copy_in(temp_path, target, fsync=fsync)
            temp_path.unlink()
            return
        if current is not None:
            os.chmod(temp_path, stat.S_IMODE(current.st_mode))
        try:
            os.replace(temp_path, target)
        except OSError as error:
            if current is None or error.errno not in _IN_PLACE_ERRORS:
                raise
            logger.debug("cannot rename over %s (%s); writing in place", target, error)
            _copy_in(temp_path, target, fsync=fsync)
            temp_path.unlink()
            return
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    if fsync:
        _fsync_directory(target.parent)


def _copy_in(staged: Path, target: Path, *, fsync: bool) -> None:
    """Write ``staged``'s bytes into ``target`` in place (it keeps its
    inode, mode and links), flushed to disk with ``fsync``: the finished
    text of a file that cannot be renamed over."""
    with staged.open("rb") as source, target.open("wb") as out:
        shutil.copyfileobj(source, out)
        if fsync:
            out.flush()
            os.fsync(out.fileno())


# A rename over an existing file that a write in place still makes: a
# bind-mounted file (EBUSY), another user's file in a sticky directory.
_IN_PLACE_ERRORS = frozenset({errno.EBUSY, errno.EPERM, errno.EACCES})
# A path naming a descriptor rather than a file: its number, or None for
# another process's.
_STANDARD_STREAMS = {"/dev/stdin": 0, "/dev/stdout": 1, "/dev/stderr": 2}
_DEV_FD = re.compile(r"/dev/fd/(\d+)")
_PROC_FD = re.compile(r"/proc/(self|thread-self|\d+)(?:/task/\d+)?/fd/(\d+)")
_MAX_LINKS = 40  # Linux's MAXSYMLINKS


def _descriptor_alias(path: Path) -> tuple[int | None] | None:
    """Whether ``path`` names a descriptor (``/dev/stdout``, ``/dev/fd/N``,
    ``/proc/<pid>/fd/N``, anything else under ``/proc``), itself or through
    the links it follows: ``(fd,)`` with the descriptor's number when it is
    this process's, ``(None,)`` when not; None for a file."""
    hop = os.path.abspath(path)
    for _ in range(_MAX_LINKS):
        parent, name = os.path.split(hop)
        for spelled in (hop, os.path.join(os.path.realpath(parent), name)):
            if spelled in _STANDARD_STREAMS:
                return (_STANDARD_STREAMS[spelled],)
            if match := _DEV_FD.fullmatch(spelled):
                return (int(match[1]),)
            if match := _PROC_FD.fullmatch(spelled):
                own = match[1] in ("self", "thread-self", str(os.getpid()))
                return (int(match[2]) if own else None,)
            if spelled.startswith("/proc/"):
                return (None,)
        if not os.path.islink(hop):
            return None
        hop = os.path.normpath(os.path.join(parent, os.readlink(hop)))
    return None  # a loop: opening it raises ELOOP


@contextmanager
def _written_through(
    path: Path, fd: int, opened: Callable[[int], IO[Any]]
) -> Iterator[IO[Any]]:
    """This process's descriptor ``fd`` (``path`` names it), written through
    a duplicate of it: at its offset and with its flags (``>>`` appends),
    after what Python buffered for it (a ``print`` before the export comes
    before it). A regular file open to update in place (``<>``, ``r+``) ends
    where the write does, as opening ``path`` to write would end it: no
    tail of its old text is left past the new."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and stream.fileno() == fd:
                stream.flush()
        except (OSError, ValueError):  # no descriptor (captured), or closed
            continue
    try:
        duplicate = os.dup(fd)
    except OSError as error:  # a closed descriptor: say which path named it
        raise OSError(error.errno, error.strerror, str(path)) from error
    with opened(duplicate) as handle:
        yield handle
    if _updated_in_place(fd):
        os.ftruncate(fd, os.lseek(fd, 0, os.SEEK_CUR))


def _updated_in_place(fd: int) -> bool:
    """Whether ``fd`` is a regular file open to write at its offset, not to
    append: one whose old text past a write would survive it."""
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        return False
    import fcntl  # POSIX only, as the descriptor aliases are

    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    return flags & os.O_ACCMODE != os.O_RDONLY and not flags & os.O_APPEND


def _fsync_directory(directory: Path) -> None:
    """Flush ``directory``'s entries (a rename into it) to disk, as far as
    it can: POSIX only (Windows opens no directory to flush), and a
    directory it may not read (``-wx``) is logged, since the rename is
    made already."""
    if os.name != "posix":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError as error:
        logger.debug("cannot flush directory %s: %s", directory, error)
        return
    try:
        os.fsync(fd)
    except OSError as error:
        logger.debug("cannot flush directory %s: %s", directory, error)
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
