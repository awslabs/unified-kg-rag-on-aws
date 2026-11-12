# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""How to install an optional extra (``[project.optional-dependencies]``):
the one remedy every "not installed" message names.

Two installs reach a user: a source checkout synced with ``uv``, and the
wheel a GitHub release attaches. ``uv sync`` installs exactly the extras it
is given and removes the others, so a hint must not say ``uv sync --extra
X`` alone (on a full environment it uninstalled every other extra); it says
to add the extra to the sync that was run, or ``--inexact``. The package is
not on PyPI, where anyone may register its name: a wheel install names the
wheel file, not the distribution.
"""

from __future__ import annotations

__all__ = ["extra_hint"]


def extra_hint(name: str) -> str:
    """How to install the ``name`` extra, for a source checkout and for a
    release wheel: "the 'viz' extra (...)", for a message to put after
    "install" or "needs"."""
    return (
        f"the '{name}' extra (in a source checkout, add `--extra {name}` to "
        f"the `uv sync` you ran, or `uv sync --inexact --extra {name}`; from a "
        f"release wheel, `pip install '<wheel file>[{name}]'`)"
    )
