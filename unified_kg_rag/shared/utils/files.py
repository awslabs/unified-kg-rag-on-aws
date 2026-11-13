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
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)


# The longest file name, in bytes, of the file systems written to (ext4,
# XFS, APFS, NTFS's 255 UTF-16 units hold at least as many characters):
# the limit where a directory does not say its own (``PC_NAME_MAX``).
NAME_MAX = 255


def name_max(directory: Path) -> int:
    """The longest file name, in bytes, ``directory`` holds (143 on
    eCryptfs), or :data:`NAME_MAX` where it does not say (Windows, a file
    system saying no limit: -1 or 0). A directory not made yet holds what
    its nearest existing parent does, the file system it is made on."""
    for existing in (directory, *directory.parents):
        if os.path.lexists(existing):
            break
    try:
        limit = os.pathconf(existing, "PC_NAME_MAX")
    except (AttributeError, OSError, ValueError):
        return NAME_MAX
    return limit if limit > 0 else NAME_MAX


def temporary_sibling(path: Path) -> Path:
    """A uniquely named hidden file beside ``path`` for writing it atomically.

    The name ends in ``.tmp``, not the target's suffix, so a sync or a glob
    over ``*.json`` never picks up a half-written file. It begins with as
    much of ``path``'s name as fits its directory's :func:`name_max` bytes
    with the rest, cut at a character boundary: a valid long name (80
    Korean characters are 240 bytes) gets a valid temporary name.
    """
    unique = f".{uuid.uuid4().hex}.tmp"
    room = name_max(path.parent) - 1 - len(unique)
    name = path.name
    while name and len(os.fsencode(name)) > room:
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
    it); until the block ends the new text is readable by its writer only
    (the temporary file is 0600). What has no sibling to rename over it is written in place as the
    block streams: a target that is no regular file (a fifo, ``/dev/null``),
    and a descriptor's alias (``/dev/stdout``, ``/dev/fd/N``,
    ``/proc/self/fd/N``, or a link into them), which is written through the
    descriptor even when it is open on a regular file (``cmd > out.txt``):
    a rename would unlink the file the shell writes to. A file whose
    sibling cannot be made or renamed over it (a writable file in a
    read-only directory or on a read-only file system, a bind-mounted file)
    is staged in a temporary file (beside it, or in
    :func:`tempfile.gettempdir` when its directory refuses one) and copied
    into it when the block ends. The file is opened before a write is
    staged elsewhere, so one that would refuse the copy too (a read-only
    file, a read-only file system) fails before the block runs, with its
    own error and path. That opened file, not the path, is what the copy
    goes into: one another writer renamed a file over or removed while the
    block ran is no longer at the path, so the copy is refused (the text
    would land in a file no path names; checked just before the copy, so a
    rename in that instant still goes unseen). A file whose directory takes a
    sibling is not opened first (a read-only file is replaced, not
    written), so a read-only file bind-mounted into a writable directory
    fails only once the block has ended and the rename is refused. A block
    that raises still leaves the file as it was, but a reader may see the
    copy half made; a copy that is refused or fails once it has begun (a
    full disk) keeps the staged file and names it in the error, the one
    whole copy of the new text left.
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

    def created(temp: Path, private: bool) -> IO[Any]:
        fd = _created(temp, private)
        try:
            return opened(fd)
        except BaseException:  # an unknown encoding: open() closed fd
            temp.unlink(missing_ok=True)
            raise

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
    held: int | None = None  # the file itself, opened to stage a write into it
    try:
        temp = created(temp_path, current is not None)
    except OSError as error:
        if current is None or error.errno not in _STAGE_ERRORS:
            raise
        # Opened (not emptied) before the block runs, so a file that will
        # not take the write either (a read-only one, or one on a read-only
        # file system) fails now, with its own error and path, not after
        # the whole export is staged.
        held = _open_in_place(target)
        try:  # from here on ``held`` is closed on any way out
            logger.debug(
                "cannot create %s (%s); staging %s elsewhere", temp_path, error, target
            )
            temp_path = temporary_sibling(Path(tempfile.gettempdir()) / target.name)
            temp = created(temp_path, True)
        except BaseException:
            os.close(held)
            raise
        beside = False
    try:
        with temp as handle:
            yield handle
            if fsync and beside:
                handle.flush()
                os.fsync(handle.fileno())
        if beside:
            if current is not None:
                os.chmod(temp_path, stat.S_IMODE(current.st_mode))
            try:
                os.replace(temp_path, target)
            except OSError as error:
                if current is None or error.errno not in _IN_PLACE_ERRORS:
                    raise
                logger.debug(
                    "cannot rename over %s (%s); writing in place", target, error
                )
                beside = False
    except BaseException:
        temp_path.unlink(missing_ok=True)
        if held is not None:
            os.close(held)
        raise
    if not beside:  # the staged text goes into the file; _copy_in owns it now
        _copy_in(temp_path, target, fsync=fsync, held=held)
    elif fsync:
        _fsync_directory(target.parent)


def _created(temp: Path, private: bool) -> int:
    """A new file ``temp`` open to write: 0600 when it holds an existing
    file's text until it replaces it (it is given that file's mode then),
    else the mode open() gives a new file (0666 less the umask), as the
    file gets."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    return os.open(temp, flags, 0o600 if private else 0o666)


def open_temporary(path: Path) -> tuple[Path, IO[bytes]]:
    """A new :func:`temporary_sibling` of ``path`` open to write bytes, for
    a write moved into place later by :func:`replace_together` (files
    staged to be moved in together): private (0600) while
    ``path`` exists, as :func:`atomic_open`'s temporary file is."""
    temporary = temporary_sibling(path)
    fd = _created(temporary, path.exists())
    try:
        return temporary, open(fd, "wb")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def replace_together(staged: Mapping[Path, Path]) -> None:
    """Rename each of :func:`open_temporary`'s temporaries over its file
    (``staged`` maps a file to its temporary), giving it the file's
    permission bits first when the file exists (a file gone since the
    temporary was made keeps its private 0600). Every mode is set before
    the first rename, so a file whose mode cannot be read or given moves
    none in. A rename is not undone: one refused because its path changed
    meanwhile (made a folder, its folder made read-only) raises, the files
    renamed before it moved in and the rest still staged."""
    for path, temporary in staged.items():
        try:
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        except FileNotFoundError:
            pass
    for path, temporary in staged.items():
        os.replace(temporary, path)


def _open_in_place(target: Path) -> int:
    """``target`` open to be written in place, not yet emptied: its own
    error (``EROFS``, ``EACCES``) and path when it will not take a write."""
    return os.open(target, os.O_WRONLY | getattr(os, "O_BINARY", 0))


def _copy_in(staged: Path, target: Path, *, fsync: bool, held: int | None) -> None:
    """Write ``staged``'s bytes into ``target`` in place (it keeps its
    inode, mode and links) through ``held``, the descriptor it was opened
    on when the write was staged (opened now when None), flushed to disk
    with ``fsync``, and remove ``staged``: the finished text of a file that
    cannot be renamed over. The descriptor pins the file, not the path: a
    file renamed over ``target`` or removed since it was opened (checked
    just before the copy) is not written, nor one removed before it is
    opened now. That refusal, or a failure once ``target`` is emptied,
    keeps ``staged``, the one whole copy of the new text, and names it in
    the error."""
    out: IO[bytes] | None = None
    try:
        try:
            source = staged.open("rb")
        except BaseException:
            if held is not None:
                os.close(held)
            raise
        try:
            out = open(_open_in_place(target) if held is None else held, "wb")
        except FileNotFoundError:  # removed since the rename over it was refused
            source.close()
        except BaseException:
            source.close()
            raise
    except BaseException:  # the file is untouched: nothing to keep
        staged.unlink(missing_ok=True)
        raise
    try:
        if out is None:
            raise OSError(errno.ENOENT, "it was removed while its text was staged")
        with source, out:
            if not _still_named(out.fileno(), target):
                raise OSError(
                    errno.ESTALE, "it was replaced or removed while its text was staged"
                )
            out.truncate(0)
            shutil.copyfileobj(source, out)
            if fsync:
                out.flush()
                os.fsync(out.fileno())
    except OSError as error:
        raise OSError(
            error.errno,
            f"{error.strerror}; its new text is kept in {staged}",
            str(target),
        ) from error
    except BaseException:
        logger.warning("%s was cut short; its new text is kept in %s", target, staged)
        raise
    staged.unlink()


def _still_named(fd: int, path: Path) -> bool:
    """Whether ``fd`` is still open on the file at ``path``: another writer
    may have renamed a new file over it or removed it since it was opened,
    and a write through ``fd`` would then go into a file no path names."""
    held = os.fstat(fd)
    try:
        now = path.stat()
    except FileNotFoundError:
        return False
    return (held.st_dev, held.st_ino) == (now.st_dev, now.st_ino)


# A rename over an existing file that a write in place still makes: a
# bind-mounted file (EBUSY), another user's file in a sticky directory.
_IN_PLACE_ERRORS = frozenset({errno.EBUSY, errno.EPERM, errno.EACCES})
# A directory refusing a new file beside an existing one, which a staged
# write may still reach: no write permission, a read-only file system.
_STAGE_ERRORS = frozenset({errno.EACCES, errno.EPERM, errno.EROFS})
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
    if _access_mode(duplicate) == os.O_RDONLY:  # its writes would fail EBADF
        os.close(duplicate)
        raise OSError(errno.EBADF, "descriptor not open for writing", str(path))
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


def _access_mode(fd: int) -> int:
    """``fd``'s access mode (``os.O_RDONLY``, ``O_WRONLY`` or ``O_RDWR``)."""
    import fcntl  # POSIX only, as the descriptor aliases are

    return fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE


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


# What a file a user supplies (a dataset, a schema, corpus text) is read as:
# UTF-8 without a leading byte-order mark, which Excel's "CSV UTF-8" and
# Notepad write (common for Korean and Japanese files) and ``json.loads``
# refuses.
USER_TEXT_ENCODING = "utf-8-sig"


def read_user_text(path: Path | str) -> str:
    """The text of a file the user supplied, as :data:`USER_TEXT_ENCODING`."""
    return Path(path).read_text(encoding=USER_TEXT_ENCODING)


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
