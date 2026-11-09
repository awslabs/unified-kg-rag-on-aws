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
