# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``atomic_write_text``: a reader sees the old file or the new one, and a
failed write leaves neither a torn file nor a temporary one behind."""

from __future__ import annotations

import os
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
