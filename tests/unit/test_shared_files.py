# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``atomic_write_text``: a reader sees the old file or the new one, and a
failed write leaves neither a torn file nor a temporary one behind."""

from __future__ import annotations

import errno
import os
import subprocess  # nosec B404 - fixed argv, no shell
import sys
from pathlib import Path
from typing import Any

import pytest

from unified_kg_rag.shared.utils import files
from unified_kg_rag.shared.utils.files import atomic_write_text, temporary_sibling

pytestmark = pytest.mark.unit


def test_the_text_replaces_the_file(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("old")
    atomic_write_text(path, "새 내용 new")
    assert path.read_text(encoding="utf-8") == "새 내용 new"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"]


def test_a_failed_rename_keeps_the_old_file_and_no_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    path.write_text("old")

    def refuse(*_: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(files.os, "replace", refuse)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_text(path, "new")
    assert path.read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"]


def test_fsync_flushes_the_data_before_the_rename_and_the_directory_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(
        files.os, "fsync", lambda fd: (calls.append("fsync"), real_fsync(fd))[1]
    )
    monkeypatch.setattr(
        files.os,
        "replace",
        lambda a, b: (calls.append("replace"), real_replace(a, b))[1],
    )
    atomic_write_text(tmp_path / "a", "x", fsync=True)
    # The data before the rename; the directory after it, so the rename
    # itself survives a power loss.
    assert calls == ["fsync", "replace", "fsync"]
    calls.clear()
    atomic_write_text(tmp_path / "b", "x")
    assert calls == ["replace"]


def test_the_temporary_name_is_hidden_unique_and_not_the_target_suffix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "segment.json"
    first, second = temporary_sibling(path), temporary_sibling(path)
    assert first != second and first.parent == tmp_path
    assert first.name.startswith(".segment.json.") and first.suffix == ".tmp"


@pytest.mark.parametrize(
    "name",
    ["가" * 80, "가" * 85, "あ" * 81 + ".txt", "a" * 255],
    ids=["korean-240-bytes", "korean-255-bytes", "japanese-247-bytes", "ascii-255"],
)
def test_a_long_valid_name_is_written_atomically(tmp_path: Path, name: str) -> None:
    # A Korean or Japanese name of 80-odd characters is 240-255 bytes in
    # UTF-8: valid itself, but too long once the temporary name's 38 bytes
    # are added (ENAMETOOLONG). The temporary name cuts the target's at a
    # character boundary to stay within a name's 255 bytes.
    path = tmp_path / name
    assert len(os.fsencode(name)) <= files.NAME_MAX
    temporary = temporary_sibling(path)
    assert len(os.fsencode(temporary.name)) <= files.NAME_MAX
    assert temporary.name.startswith(".") and temporary.suffix == ".tmp"
    temporary.name.encode("utf-8")  # no half character
    path.write_text("old")
    atomic_write_text(path, "새 내용")
    assert path.read_text(encoding="utf-8") == "새 내용"
    assert sorted(p.name for p in tmp_path.iterdir()) == [name]


def _interrupted(monkeypatch: pytest.MonkeyPatch) -> None:
    def crash(*_: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(files.os, "replace", crash)


def test_saved_documents_results_and_comparisons_are_written_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A run stopped mid-write leaves the previous file, or none: never a
    # truncated one a later --from-results or a sync would read.
    from unified_kg_rag.application.cli.run_evaluation import EvaluationRunner
    from unified_kg_rag.domain.models import Document
    from unified_kg_rag.evaluation.evaluation_manager import EvaluationManager

    document, results = tmp_path / "doc.json", tmp_path / "results.json"
    document.write_text("old")
    results.write_text("old")
    _interrupted(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        Document(
            page_content="x", document_id="d", file_name="a.txt", file_path="a.txt"
        ).to_json_file(document)
    with pytest.raises(KeyboardInterrupt):
        EvaluationManager._save_json(results, {"a": 1})
    with pytest.raises(KeyboardInterrupt):
        EvaluationRunner._save_comparison({"a": 1}, tmp_path / "out")
    assert document.read_text() == results.read_text() == "old"
    assert sorted(p.name for p in tmp_path.rglob("*")) == [
        "doc.json",
        "out",
        "results.json",
    ]


def test_a_stream_replaces_the_file_whole_or_not_at_all(tmp_path: Path) -> None:
    path = tmp_path / "export.jsonl"
    path.write_text("old")
    with files.atomic_open(path, newline="\n") as out:
        out.write("a\n")
        assert path.read_text() == "old"  # not until the block ends
    assert path.read_text() == "a\n"
    with pytest.raises(RuntimeError), files.atomic_open(path, "wb") as raw:
        raw.write(b"partial")
        raise RuntimeError("the export failed")
    assert path.read_text() == "a\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["export.jsonl"]
    with pytest.raises(ValueError):
        with files.atomic_open(path, "a"):
            pass


def test_a_symlinked_target_updates_the_file_it_points_to(tmp_path: Path) -> None:
    # A config.yaml linked from elsewhere stays a link: the file it names
    # gets the text, and no copy shadows it.
    real = tmp_path / "real.yaml"
    real.write_text("a")
    link = tmp_path / "config.yaml"
    link.symlink_to(real.name)
    atomic_write_text(link, "b")
    assert link.is_symlink() and real.read_text() == "b"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.yaml", "real.yaml"]
    # A dangling link creates the file it names.
    dangling = tmp_path / "next.yaml"
    dangling.symlink_to("made.yaml")
    atomic_write_text(dangling, "c")
    assert dangling.is_symlink() and (tmp_path / "made.yaml").read_text() == "c"


def test_the_replaced_file_keeps_its_mode(tmp_path: Path) -> None:
    path = tmp_path / "secret.json"
    path.write_text("old")
    path.chmod(0o600)
    atomic_write_text(path, "new")
    assert path.stat().st_mode & 0o777 == 0o600
    # A new file gets the umask's mode, as open() gives it.
    fresh = tmp_path / "fresh.json"
    atomic_write_text(fresh, "x")
    umask = os.umask(0)
    os.umask(umask)
    assert fresh.stat().st_mode & 0o777 == 0o666 & ~umask


def test_a_fifo_or_device_target_is_written_in_place(tmp_path: Path) -> None:
    # /dev/stdout or a fifo has no sibling to rename over it (creating
    # /dev/.stdout.<uuid>.tmp fails): the stream goes to it as it is.
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    read_fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    try:
        with files.atomic_open(fifo, "wb") as out:
            out.write(b"streamed")
        assert os.read(read_fd, 64) == b"streamed"
    finally:
        os.close(read_fd)
    assert fifo.is_fifo()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pipe"]
    atomic_write_text(Path(os.devnull), "discarded")


_PRINT_AROUND_AN_EXPORT = (
    "import sys\n"
    "from pathlib import Path\n"
    "from unified_kg_rag.shared.utils.files import atomic_write_text\n"
    "print('before')\n"
    "atomic_write_text(Path(sys.argv[1]), 'EXPORT\\n')\n"
    "print('after')\n"
)


@pytest.mark.skipif(os.name != "posix", reason="shell redirects and /dev/fd")
@pytest.mark.parametrize("alias", ["/dev/stdout", "/dev/fd/1", "/proc/self/fd/1"])
@pytest.mark.parametrize("redirect", [">", ">>"])
def test_stdout_redirected_to_a_file_is_written_through_not_replaced(
    tmp_path: Path, alias: str, redirect: str
) -> None:
    # `run-describe --output-path /dev/stdout > out.txt`: /dev/stdout names
    # the regular file the shell opened. Renaming a sibling over it unlinked
    # the file the shell still writes to, so the lines printed around the
    # export (and, with >>, what the file held) were lost.
    if not Path(alias).exists():
        pytest.skip(f"no {alias} here")
    out = tmp_path / "out.txt"
    out.write_text("first\n")
    script = tmp_path / "export.py"
    script.write_text(_PRINT_AROUND_AN_EXPORT)
    # The descriptor a shell's > (truncate) or >> (append) hands the run.
    with out.open("w" if redirect == ">" else "a") as redirected:
        subprocess.run(  # nosec B603 - fixed argv, no shell
            [sys.executable, str(script), alias],
            stdout=redirected,
            check=True,
            cwd=Path(__file__).parents[2],
        )
    kept = "first\n" if redirect == ">>" else ""
    assert out.read_text() == f"{kept}before\nEXPORT\nafter\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["export.py", "out.txt"]


def test_a_symlink_to_an_fd_alias_is_written_through(tmp_path: Path) -> None:
    # A link to /dev/stdout (or into /proc/self/fd) aliases the descriptor:
    # the write goes to what the descriptor is open on, at its offset.
    out = tmp_path / "out.txt"
    with out.open("w") as held:
        held.write("held\n")
        held.flush()
        link = tmp_path / "link"
        link.symlink_to(f"/proc/self/fd/{held.fileno()}")
        if not link.exists():
            pytest.skip("no /proc/self/fd here")
        atomic_write_text(link, "through\n")
        held.write("more\n")
    assert out.read_text() == "held\nthrough\nmore\n"
    assert link.is_symlink()


_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


@pytest.mark.skipif(_ROOT or os.name != "posix", reason="root reads any directory")
def test_a_directory_it_cannot_flush_does_not_fail_a_made_write(
    tmp_path: Path,
) -> None:
    # -wx: the sibling is made and renamed, only the directory flush is
    # refused; the write is done, so the call says so.
    folder = tmp_path / "wx"
    folder.mkdir()
    folder.chmod(0o333)
    try:
        atomic_write_text(folder / "state.json", "new", fsync=True)
    finally:
        folder.chmod(0o755)
    assert (folder / "state.json").read_text() == "new"


@pytest.mark.skipif(_ROOT, reason="root writes into a read-only directory")
def test_a_writable_file_in_a_read_only_directory_is_written_in_place(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "ro"
    folder.mkdir()
    path = folder / "settings.yaml"
    path.write_text("old")
    folder.chmod(0o555)
    try:
        atomic_write_text(path, "new")
        assert path.read_text() == "new"
        assert sorted(p.name for p in folder.iterdir()) == ["settings.yaml"]
        # No file yet: nothing to write in place, the directory's refusal.
        with pytest.raises(PermissionError):
            atomic_write_text(folder / "fresh.yaml", "x")
    finally:
        folder.chmod(0o755)


def _staged_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Where a write that cannot be made beside its file stages it."""
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(files.tempfile, "tempdir", str(stage))
    return stage


def _counted_fsyncs(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    real = os.fsync
    monkeypatch.setattr(files.os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    return calls


@pytest.mark.skipif(_ROOT, reason="root writes into a read-only directory")
def test_a_read_only_directory_s_file_keeps_its_text_when_the_block_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The in-place write is staged elsewhere and copied in on success: an
    # export that fails halfway leaves the previous file, not its first half.
    stage = _staged_in(monkeypatch, tmp_path)
    folder = tmp_path / "ro"
    folder.mkdir()
    path = folder / "export.jsonl"
    path.write_text("OLD" * 1000)
    folder.chmod(0o555)
    try:
        with pytest.raises(RuntimeError), files.atomic_open(path) as out:
            out.write("NEW-partial")
            out.flush()
            raise RuntimeError("the store went away")
        assert path.read_text() == "OLD" * 1000
        fsyncs = _counted_fsyncs(monkeypatch)
        atomic_write_text(path, "new", fsync=True)
        assert path.read_text() == "new"
        assert fsyncs  # the copied-in file is flushed, as asked
        assert sorted(p.name for p in folder.iterdir()) == ["export.jsonl"]
    finally:
        folder.chmod(0o755)
    assert list(stage.iterdir()) == []


@pytest.mark.parametrize("code", [errno.EBUSY, errno.EPERM, errno.EACCES])
def test_a_file_it_cannot_rename_over_flushes_the_copy_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    path = tmp_path / "mounted.json"
    path.write_text("old")

    def refuse(*_: Any) -> None:
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(files.os, "replace", refuse)
    fsynced: list[int] = []
    real = os.fsync

    def fsync(fd: int) -> None:
        fsynced.append(os.fstat(fd).st_ino)
        real(fd)

    monkeypatch.setattr(files.os, "fsync", fsync)
    atomic_write_text(path, "new", fsync=True)
    assert path.read_text() == "new"
    assert path.stat().st_ino in fsynced  # the file itself, not just the temp


@pytest.mark.parametrize("code", [errno.EBUSY, errno.EPERM, errno.EACCES])
def test_a_file_it_cannot_rename_over_is_written_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    # A bind-mounted file (EBUSY) or another user's in a sticky directory:
    # the rename is refused, a write into the file is not.
    path = tmp_path / "mounted.json"
    path.write_text("old")

    def refuse(*_: Any) -> None:
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(files.os, "replace", refuse)
    atomic_write_text(path, "new")
    assert path.read_text() == "new"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mounted.json"]
    # A file that is not there yet has none to write into.
    with pytest.raises(OSError):
        atomic_write_text(tmp_path / "fresh.json", "x")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mounted.json"]


def test_a_descriptor_open_on_a_file_for_update_drops_the_old_tail(
    tmp_path: Path,
) -> None:
    # A file opened read-write (a shell's <>) at its start: the write goes
    # through the descriptor at its offset and ends the file there, as
    # opening the path to write would; nothing of the old text follows it.
    path = tmp_path / "held.txt"
    path.write_text("0123456789ABCDEF")
    with path.open("r+") as held:
        alias = Path(f"/proc/self/fd/{held.fileno()}")
        if not alias.exists():
            pytest.skip("no /proc/self/fd here")
        atomic_write_text(alias, "xy")
    assert path.read_text() == "xy"


def test_a_closed_descriptor_s_alias_names_the_path(tmp_path: Path) -> None:
    fd = os.open(tmp_path / "f", os.O_WRONLY | os.O_CREAT)
    os.close(fd)
    alias = Path(f"/dev/fd/{fd}")
    with pytest.raises(OSError) as raised:
        atomic_write_text(alias, "x")
    assert raised.value.filename == str(alias)


def _temporaries(folder: Path, name: str) -> list[Path]:
    return [p for p in folder.iterdir() if p.name.startswith(f".{name}.")]


@pytest.mark.parametrize("target_mode", [0o600, 0o644], ids=["0600", "0644"])
def test_the_temporary_file_is_private_while_the_block_runs(
    tmp_path: Path, target_mode: int
) -> None:
    # The new text streams into the temporary file before it is renamed over
    # the target: other users must not read it then, whatever the target's
    # mode; the target's mode is given to it only at the rename.
    path = tmp_path / "secret.json"
    path.write_text("old")
    path.chmod(target_mode)
    with files.atomic_open(path) as out:
        out.write("new")
        (temporary,) = _temporaries(tmp_path, "secret.json")
        assert temporary.stat().st_mode & 0o777 == 0o600
    assert path.read_text() == "new"
    assert path.stat().st_mode & 0o777 == target_mode


@pytest.mark.skipif(_ROOT, reason="root writes into a read-only directory")
def test_the_staged_copy_is_private_while_the_block_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The staging directory is the shared temp directory: a 0600 file's new
    # text must not be readable there while it streams.
    stage = _staged_in(monkeypatch, tmp_path)
    folder = tmp_path / "ro"
    folder.mkdir()
    path = folder / "secret.json"
    path.write_text("old")
    path.chmod(0o600)
    folder.chmod(0o555)
    try:
        with files.atomic_open(path) as out:
            out.write("new")
            (staged,) = _temporaries(stage, "secret.json")
            assert staged.stat().st_mode & 0o777 == 0o600
        assert path.read_text() == "new"
        assert path.stat().st_mode & 0o777 == 0o600
    finally:
        folder.chmod(0o755)
    assert list(stage.iterdir()) == []


@pytest.mark.parametrize("code", [errno.EROFS, errno.EACCES, errno.EPERM])
def test_a_directory_refusing_a_sibling_stages_the_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    # A bind-mounted file in a directory on a read-only file system refuses
    # the sibling with EROFS, not a PermissionError: it is staged as well.
    stage = _staged_in(monkeypatch, tmp_path)
    folder = tmp_path / "mounted"
    folder.mkdir()
    path = folder / "f.json"
    path.write_text("OLD")
    real = os.open

    def refuse(file: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if Path(file).parent == folder and Path(file).name.startswith(".f.json."):
            raise OSError(code, os.strerror(code), str(file))
        return real(file, flags, *args, **kwargs)

    monkeypatch.setattr(files.os, "open", refuse)
    atomic_write_text(path, "NEW")
    assert path.read_text() == "NEW"
    assert list(stage.iterdir()) == []
    # No file yet: nothing to write in place, the directory's refusal.
    with pytest.raises(OSError) as raised:
        atomic_write_text(folder / "f.json.new", "x")
    assert raised.value.errno == code


@pytest.mark.parametrize("refusal", ["read-only file", "read-only file system"])
def test_a_file_that_would_refuse_the_staged_copy_fails_before_the_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal: str
) -> None:
    # A read-only file system refuses the sibling with EROFS; the file
    # itself refuses the write too (a 0444 file, or the same file system):
    # it fails at once with the file's own error and path, before a byte
    # of the export is staged, not once the whole block has streamed.
    stage = _staged_in(monkeypatch, tmp_path)
    folder = tmp_path / "ro"
    folder.mkdir()
    path = folder / "out.bin"
    path.write_text("OLD")
    real = os.open
    target_code = errno.EACCES if refusal == "read-only file" else errno.EROFS

    def refuse(file: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if Path(file).parent == folder and Path(file).name.startswith(".out.bin."):
            raise OSError(errno.EROFS, os.strerror(errno.EROFS), str(file))
        if Path(file) == path and target_code == errno.EROFS:
            raise OSError(errno.EROFS, os.strerror(errno.EROFS), str(file))
        return real(file, flags, *args, **kwargs)

    monkeypatch.setattr(files.os, "open", refuse)
    if refusal == "read-only file":
        path.chmod(0o444)
    ran = False
    try:
        if refusal == "read-only file" and os.access(path, os.W_OK):
            pytest.skip("root writes a read-only file")
        with pytest.raises(OSError) as raised:
            with files.atomic_open(path, "wb") as out:
                ran = True
                out.write(b"NEW")
    finally:
        path.chmod(0o644)
    assert not ran
    assert raised.value.errno == target_code
    assert raised.value.filename == str(path)
    assert path.read_text() == "OLD"
    assert list(stage.iterdir()) == []
    assert sorted(p.name for p in folder.iterdir()) == ["out.bin"]


def test_a_staged_write_opens_the_file_once_and_empties_it_only_at_the_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The file is opened when the write is staged and written through that
    # descriptor: emptied only when the block's text goes in.
    _staged_in(monkeypatch, tmp_path)
    folder = tmp_path / "mounted"
    folder.mkdir()
    path = folder / "f.json"
    path.write_text("OLD TEXT")
    real = os.open
    opened: list[int] = []

    def refuse(file: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if Path(file).parent == folder and Path(file).name.startswith(".f.json."):
            raise OSError(errno.EROFS, os.strerror(errno.EROFS), str(file))
        if Path(file) == path:
            opened.append(flags)
        return real(file, flags, *args, **kwargs)

    monkeypatch.setattr(files.os, "open", refuse)
    with files.atomic_open(path) as out:
        assert len(opened) == 1 and not opened[0] & (os.O_TRUNC | os.O_CREAT)
        assert path.read_text() == "OLD TEXT"
        out.write("NEW")
    assert path.read_text() == "NEW" and len(opened) == 1


def _copy_fails_midway(monkeypatch: pytest.MonkeyPatch) -> None:
    def short(source: Any, out: Any, *_: Any) -> None:
        out.write(source.read(3))
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(files.shutil, "copyfileobj", short)


@pytest.mark.skipif(_ROOT, reason="root writes into a read-only directory")
def test_a_failed_copy_in_keeps_the_staged_text_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The copy into the file truncated it before it failed (a full disk):
    # the staged file is the one whole copy of the new text, so it is kept
    # and the error says where.
    stage = _staged_in(monkeypatch, tmp_path)
    folder = tmp_path / "ro"
    folder.mkdir()
    path = folder / "export.jsonl"
    path.write_text("OLD")
    folder.chmod(0o555)
    _copy_fails_midway(monkeypatch)
    try:
        with pytest.raises(OSError) as raised:
            atomic_write_text(path, "HALF-AND-REST")
    finally:
        folder.chmod(0o755)
    (staged,) = stage.iterdir()
    assert staged.read_text() == "HALF-AND-REST"
    assert raised.value.errno == errno.ENOSPC
    assert raised.value.filename == str(path) and str(staged) in str(raised.value)


@pytest.mark.skipif(_ROOT, reason="root writes into a read-only directory")
@pytest.mark.parametrize("how", ["replaced", "removed"])
def test_a_file_replaced_while_its_write_is_staged_keeps_the_staged_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    # The staged write holds the file open from before the block; another
    # writer (the directory's owner) renames a new file over it, or removes
    # it, meanwhile. Writing through the held descriptor would put the
    # export into a file no path names: it fails instead, naming the copy.
    stage = _staged_in(monkeypatch, tmp_path)
    folder = tmp_path / "ro"
    folder.mkdir()
    path = folder / "export.jsonl"
    path.write_text("OLD")
    folder.chmod(0o555)
    try:
        with pytest.raises(OSError) as raised, files.atomic_open(path) as out:
            out.write("NEW EXPORT")
            folder.chmod(0o755)
            if how == "replaced":
                (folder / "theirs").write_text("theirs")
                os.replace(folder / "theirs", path)
            else:
                path.unlink()
            folder.chmod(0o555)
    finally:
        folder.chmod(0o755)
    (staged,) = stage.iterdir()
    assert staged.read_text() == "NEW EXPORT"
    assert raised.value.filename == str(path) and str(staged) in str(raised.value)
    if how == "replaced":
        assert path.read_text() == "theirs"  # the other writer's text stands


@pytest.mark.skipif(_ROOT, reason="root writes into a read-only directory")
def test_an_interrupt_right_after_the_file_is_opened_closes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ctrl-C between opening the file to stage its write and the block:
    # the descriptor is closed, not leaked.
    _staged_in(monkeypatch, tmp_path)
    folder = tmp_path / "ro"
    folder.mkdir()
    path = folder / "export.jsonl"
    path.write_text("OLD")
    opened: list[int] = []
    real = files._open_in_place
    monkeypatch.setattr(
        files, "_open_in_place", lambda p: (opened.append(real(p)), opened[-1])[1]
    )

    def interrupt(*_: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(files.logger, "debug", interrupt)
    folder.chmod(0o555)
    try:
        with pytest.raises(KeyboardInterrupt), files.atomic_open(path):
            pytest.fail("the block runs after the interrupt")
    finally:
        folder.chmod(0o755)
    (fd,) = opened
    with pytest.raises(OSError) as closed:
        os.fstat(fd)
    assert closed.value.errno == errno.EBADF
    assert path.read_text() == "OLD"


def test_a_failed_copy_in_after_a_refused_rename_keeps_the_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "mounted.json"
    path.write_text("old")

    def refuse(*_: Any) -> None:
        raise OSError(errno.EBUSY, os.strerror(errno.EBUSY))

    monkeypatch.setattr(files.os, "replace", refuse)
    _copy_fails_midway(monkeypatch)
    with pytest.raises(OSError) as raised:
        atomic_write_text(path, "new text")
    (kept,) = _temporaries(tmp_path, "mounted.json")
    assert kept.read_text() == "new text" and str(kept) in str(raised.value)


def test_a_copy_in_that_cannot_open_the_file_leaves_no_staged_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The file was never touched: its old text stands, nothing to keep.
    path = tmp_path / "mounted.json"
    path.write_text("old")

    def refuse(*_: Any) -> None:
        raise OSError(errno.EBUSY, os.strerror(errno.EBUSY))

    monkeypatch.setattr(files.os, "replace", refuse)
    path.chmod(0o444)
    try:
        if os.access(path, os.W_OK):
            pytest.skip("root writes a read-only file")
        with pytest.raises(PermissionError):
            atomic_write_text(path, "new")
    finally:
        path.chmod(0o644)
    assert path.read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mounted.json"]


def test_the_temporary_name_fits_the_directory_s_name_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # eCryptfs allows 143 bytes: the limit is the directory's, not 255.
    monkeypatch.setattr(files.os, "pathconf", lambda *_: 143)
    temporary = temporary_sibling(tmp_path / ("a" * 140))
    assert len(os.fsencode(temporary.name)) == 143

    def unknown(*_: Any) -> int:
        raise OSError(errno.EINVAL, os.strerror(errno.EINVAL))

    monkeypatch.setattr(files.os, "pathconf", unknown)
    temporary = temporary_sibling(tmp_path / ("a" * 255))
    assert len(os.fsencode(temporary.name)) == files.NAME_MAX


@pytest.mark.parametrize("said", [-1, 0])
@pytest.mark.timeout(10)
def test_a_file_system_saying_no_name_limit_gets_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, said: int
) -> None:
    # POSIX lets pathconf return -1 for "no limit" (CPython does not raise),
    # and a file system may report 0: the default limit, not a negative room
    # the name is cut toward forever.
    monkeypatch.setattr(files.os, "pathconf", lambda *_: said)
    assert files.name_max(tmp_path) == files.NAME_MAX
    temporary = temporary_sibling(tmp_path / ("a" * 255))
    assert len(os.fsencode(temporary.name)) == files.NAME_MAX


@pytest.mark.timeout(10)
def test_a_limit_shorter_than_the_unique_part_ends_the_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(files.os, "pathconf", lambda *_: 20)
    assert temporary_sibling(tmp_path / "name.json").name.endswith(".tmp")


def test_a_directory_not_made_yet_has_its_parent_s_name_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[Path] = []

    def limit(directory: Any, _: str) -> int:
        asked.append(Path(directory))
        return 143

    monkeypatch.setattr(files.os, "pathconf", limit)
    assert files.name_max(tmp_path / "new" / "deeper") == 143
    assert asked == [tmp_path]


def test_a_read_only_descriptor_s_alias_names_the_path(tmp_path: Path) -> None:
    (tmp_path / "f").write_text("kept")
    fd = os.open(tmp_path / "f", os.O_RDONLY)
    try:
        alias = Path(f"/dev/fd/{fd}")
        if not alias.exists():
            pytest.skip("no /dev/fd here")
        with pytest.raises(OSError) as raised:
            atomic_write_text(alias, "x")
    finally:
        os.close(fd)
    assert raised.value.errno == errno.EBADF
    assert raised.value.filename == str(alias)
    assert (tmp_path / "f").read_text() == "kept"


def test_a_temporary_opened_to_move_in_later_is_private_and_keeps_the_mode(
    tmp_path: Path,
) -> None:
    # Files staged to move in together (a console upload) use the same
    # rule as atomic_open: 0600 while the file exists, its mode at the rename.
    path = tmp_path / "f.txt"
    path.write_text("old")
    path.chmod(0o640)
    temporary, out = files.open_temporary(path)
    with out:
        out.write(b"new")
    assert temporary.stat().st_mode & 0o777 == 0o600
    files.replace_together({path: temporary})
    assert path.read_text() == "new" and path.stat().st_mode & 0o777 == 0o640
    assert [p.name for p in tmp_path.iterdir()] == ["f.txt"]


def test_files_moved_in_together_move_none_when_a_mode_cannot_be_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every file's mode is set before the first rename: a failure there
    # (the second file's mode refused) leaves the first file's old text.
    paths = [tmp_path / "a.txt", tmp_path / "b.txt"]
    staged = {}
    for path in paths:
        path.write_text("old")
        temporary, out = files.open_temporary(path)
        with out:
            out.write(b"new")
        staged[path] = temporary
    real = os.chmod

    def refuse_second(target: Any, mode: int) -> None:
        if Path(target) == staged[paths[1]]:
            raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))
        real(target, mode)

    monkeypatch.setattr(files.os, "chmod", refuse_second)
    with pytest.raises(PermissionError):
        files.replace_together(staged)
    assert [p.read_text() for p in paths] == ["old", "old"]
