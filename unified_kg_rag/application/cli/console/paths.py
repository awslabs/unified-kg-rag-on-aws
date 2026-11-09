# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The directories the console may read and write, and the containment
check every path a client names goes through.

A client never sends a filesystem path: it names a root and a path relative
to it (``"workspace:corpus/a.txt"``). The path is resolved with symlinks
followed and must stay inside the root, so ``..``, an absolute path and a
symlink that leaves the root are refused alike. Writes go only to the roots
marked writable.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from unified_kg_rag.shared.exceptions import ConfigurationError
from unified_kg_rag.shared.utils.files import NAME_MAX, name_max

# What one uploaded path segment may hold: no separator, no control
# character, not "." or "..", not starting with "~".
_UNSAFE_SEGMENT = re.compile(r"[\x00-\x1f\x7f/\\:]")


class PathRefused(ConfigurationError):
    """A path a client named is outside the console's directories (or
    malformed): the request is refused, nothing is read or written."""


@dataclass(frozen=True)
class Root:
    """A directory the console reads (and, if ``writable``, writes)."""

    name: str
    path: Path
    writable: bool = False


@dataclass(frozen=True)
class Roots:
    """The console's directories by name."""

    roots: dict[str, Root] = field(default_factory=dict)

    @classmethod
    def build(cls, *roots: Root) -> Roots:
        return cls(
            {
                root.name: Root(root.name, root.path.resolve(), root.writable)
                for root in roots
            }
        )

    def root(self, name: str) -> Root:
        try:
            return self.roots[name]
        except KeyError:
            raise PathRefused(f"unknown directory '{name}'") from None

    def resolve(self, ref: str, *, write: bool = False) -> Path:
        """The file or directory ``ref`` (``<root>:<relative path>``) names.

        Raises:
            PathRefused: An unknown root, a write to a read-only root, an
                absolute path, or a path that resolves outside its root.
        """
        name, sep, relative = ref.partition(":")
        if not sep:
            raise PathRefused(f"'{ref}' names no directory (expected '<root>:<path>')")
        root = self.root(name)
        if write and not root.writable:
            raise PathRefused(f"the '{name}' directory is read-only")
        return contained(root.path, relative)

    def ref(self, path: Path) -> str | None:
        """``path`` as a reference, under the innermost root that holds it."""
        resolved = path.resolve()
        holding = [
            root
            for root in self.roots.values()
            if resolved == root.path or resolved.is_relative_to(root.path)
        ]
        if not holding:
            return None
        root = max(holding, key=lambda r: len(r.path.parts))
        relative = resolved.relative_to(root.path).as_posix()
        return f"{root.name}:{'' if relative == '.' else relative}"


def contained(root: Path, relative: str) -> Path:
    """``root / relative``, resolved, if it stays inside ``root``.

    Raises:
        PathRefused: ``relative`` is absolute (POSIX or Windows style) or
            resolves outside ``root`` (``..``, a symlink that leaves it).
    """
    text = relative.replace("\\", "/")
    if text.startswith("/") or re.match(r"^[A-Za-z]:", text) or "\x00" in text:
        raise PathRefused(f"'{relative}' is not a relative path")
    base = root.resolve()
    target = (base / text).resolve()
    if target != base and not target.is_relative_to(base):
        raise PathRefused(f"'{relative}' is outside the allowed directory")
    return target


def safe_relative(relative: str, *, within: Path | None = None) -> PurePosixPath:
    """An uploaded file's relative path, checked segment by segment.

    Each segment is NFC-normalized and must be a plain name (no separator,
    control character, ``..``, leading ``~``), so the written path is the
    one the user saw, and fit the name limit of the directory it is made in
    under ``within`` (143 bytes on eCryptfs; 255 when ``within`` is None).

    Raises:
        PathRefused: An empty path or an unsafe segment.
    """
    parts = [
        unicodedata.normalize("NFC", p) for p in relative.replace("\\", "/").split("/")
    ]
    parts = [p for p in parts if p not in ("", ".")]
    if not parts:
        raise PathRefused("an uploaded file has no name")
    for depth, part in enumerate(parts):
        if part == ".." or part.startswith("~") or _UNSAFE_SEGMENT.search(part):
            raise PathRefused(f"'{relative}' holds an unsafe name '{part}'")
        limit = NAME_MAX if within is None else name_max(Path(within, *parts[:depth]))
        if len(part.encode("utf-8")) > limit:
            raise PathRefused(f"'{relative}' holds a name longer than {limit} bytes")
    return PurePosixPath(*parts)


def safe_name(value: str, what: str, *, within: Path | None = None) -> str:
    """A single directory name a client chose (an upload's folder), made
    in ``within`` (:func:`safe_relative`).

    Raises:
        PathRefused: Not a plain, non-empty name.
    """
    path = safe_relative(value, within=within)
    if len(path.parts) != 1:
        raise PathRefused(f"{what} must be a single name, not a path")
    return path.parts[0]


def walk_files(directory: Path, *, limit: int) -> list[Path]:
    """Up to ``limit`` regular files under ``directory``, symlinks not
    followed, hidden entries skipped, sorted."""
    found: list[Path] = []
    for current, dirs, files in os.walk(directory, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.startswith("."):
                continue
            path = Path(current) / name
            if path.is_symlink() or not path.is_file():
                continue
            found.append(path)
            if len(found) >= limit:
                return found
    return found
