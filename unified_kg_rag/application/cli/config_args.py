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
from unified_kg_rag.application.config import (
    ConfigLoader,
    warn_ineffective_settings,
)
from unified_kg_rag.domain.models import Config
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
        usage_error(str(e), error_type=type(e).__name__)


def add_config_arguments(
    parser: argparse.ArgumentParser | argparse._ArgumentGroup,
    help_text: str = (
        "Configuration YAML file (without one: the built-in defaults and the "
        "UKR__ environment variables)"
    ),
) -> None:
    """Add ``--config-path`` and ``--allow-unknown-keys``."""
    parser.add_argument(
        "--config-path",
        type=config_path_arg,
        default=None,
        help=f"{help_text}; a missing file is a usage error",
    )
    parser.add_argument(
        "--allow-unknown-keys",
        action="store_true",
        help=(
            "Ignore config keys (and UKR__ environment variables) the "
            "configuration does not define, with a warning each, instead of "
            "failing: the load fails on one by default, since a misspelled key "
            "would leave its setting at the default"
        ),
    )


def load_cli_config(args: argparse.Namespace) -> Config:
    """The configuration ``args`` names (``--config-path``,
    ``--allow-unknown-keys``) with its provider flags applied
    (:func:`~unified_kg_rag.application.cli.provider_args.apply_provider_arguments`);
    call it inside :func:`exit_on_config_error`.

    The settings nothing reads are warned of once the flags are applied: a
    ``base_url`` the file sets is read under ``--backend local``, not ignored.
    """
    # Imported here: the provider flags' module loads the composition root,
    # which --help and a refused command line do not need.
    from unified_kg_rag.application.cli.provider_args import apply_provider_arguments

    path = getattr(args, "config_path", None)
    loaded = ConfigLoader(
        Path(path) if path else None,
        allow_unknown_keys=getattr(args, "allow_unknown_keys", False),
        warn_ineffective=False,
    ).load_config()
    config = apply_provider_arguments(loaded, args)
    warn_ineffective_settings(config)
    return config
