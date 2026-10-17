# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CLIs' ``--config-path`` option and configuration loading.

A missing, unreadable or invalid configuration file is a usage error: every
CLI reports it on one line and exits 2, as argparse does for a bad option,
instead of printing "An unexpected error occurred" with a traceback.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from unified_kg_rag.application.cli.conventions import usage_error
from unified_kg_rag.shared.exceptions import ConfigurationError


def config_path_arg(value: str) -> str:
    """argparse ``type`` for ``--config-path``: the file must exist."""
    if not Path(value).is_file():
        raise argparse.ArgumentTypeError(f"configuration file not found: '{value}'")
    return value


@contextmanager
def exit_on_config_error() -> Iterator[None]:
    """Turn a bad configuration file met inside the block into a usage error.

    Wrap the ``get_config`` call: a missing file, invalid YAML or a failed
    validation prints one ``<prog>: error: ...`` line to stderr and exits 2
    (:func:`~unified_kg_rag.application.cli.conventions.usage_error`).
    """
    try:
        yield
    except (ConfigurationError, FileNotFoundError) as e:
        usage_error(str(e))
