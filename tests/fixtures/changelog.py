# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CHANGELOG.md section being released, which the changelog tests read.

Between releases that is ``[Unreleased]``. A release cut renames it to
``## [X.Y.Z] - date`` and adds an empty ``[Unreleased]`` above it, so from
the cut until the first entry of the next cycle it is the newest version
section: the one the tag publishes.
"""

from __future__ import annotations

import re

UNRELEASED = "Unreleased"
_HEADING = re.compile(r"^## \[(?P<name>[^\]]+)\]", re.M)
# A section ends at the next version heading or at the link references.
_END = re.compile(r"^(?:## \[|\[[^\]]+\]:\s)", re.M)


def section(text: str, name: str) -> str:
    """The ``## [name]`` section of ``text``, its heading included."""
    match = re.search(rf"^## \[{re.escape(name)}\]", text, re.M)
    if match is None:
        raise ValueError(f"CHANGELOG.md has no `## [{name}]` section")
    start = match.start()
    end = _END.search(text, match.end())
    return text[start : end.start()] if end else text[start:]


def released_name(text: str) -> str:
    """``Unreleased`` while that section holds an entry, else the newest
    version heading's (the release just cut)."""
    unreleased = section(text, UNRELEASED)
    if unreleased.partition("\n")[2].strip():
        return UNRELEASED
    versions = [m["name"] for m in _HEADING.finditer(text) if m["name"] != UNRELEASED]
    if not versions:
        raise ValueError("CHANGELOG.md has an empty [Unreleased] and no release")
    return versions[0]
