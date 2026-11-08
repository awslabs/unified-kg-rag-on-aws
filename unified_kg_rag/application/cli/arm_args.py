# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``run-eval --arm NAME=overlay.yaml[,suffix=SUFFIX]``: a configuration arm.

The overlay is a YAML mapping merged over the run's configuration (as the
environment layer is over a config file), validated like a config file
(unknown keys refused unless ``--allow-unknown-keys``), its relative paths
resolved against its own directory. ``suffix=`` queries another index
namespace with the arm (``--suffix``'s, which the overlay's index settings
otherwise follow). The arm is scored exactly as the base runs are: its
``evaluation`` section is the run's.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from unified_kg_rag.application.config import (
    ConfigLoader,
    resolve_relative_paths,
    unknown_keys,
)
from unified_kg_rag.domain.models import Config, ModelRole
from unified_kg_rag.domain.models.config import merge_over
from unified_kg_rag.shared.exceptions import ConfigurationError
from unified_kg_rag.shared.utils.validation import format_validation_errors

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
# Labels a comparison already gives a meaning.
_RESERVED = frozenset({"closed_book", "baseline"})


def arm_spec(value: str) -> dict[str, Any]:
    """argparse ``type`` of ``--arm``: ``{"name", "overlay", "suffix"}``."""
    name, sep, rest = value.partition("=")
    name = name.strip()
    if not sep or not rest.strip():
        raise argparse.ArgumentTypeError(
            f"'{value}' is not NAME=overlay.yaml[,suffix=SUFFIX]"
        )
    if not _NAME.match(name) or name.lower() in _RESERVED:
        raise argparse.ArgumentTypeError(
            f"arm name '{name}' must be letters, digits, '_', '.' or '-' (not "
            f"{', '.join(sorted(_RESERVED))})"
        )
    path, *options = rest.split(",")
    suffix: str | None = None
    for option in options:
        key, eq, setting = option.partition("=")
        if key.strip() != "suffix" or not eq or not setting.strip():
            raise argparse.ArgumentTypeError(
                f"unknown --arm option '{option}' (only suffix=SUFFIX)"
            )
        suffix = setting.strip()
    overlay = Path(path.strip())
    if not overlay.is_file():
        raise argparse.ArgumentTypeError(f"overlay file not found: '{overlay}'")
    return {"name": name, "overlay": str(overlay), "suffix": suffix}


def overlay_sha256(spec: dict[str, Any]) -> str:
    """The digest of the arm's overlay file, as the run manifest records it."""
    return hashlib.sha256(Path(spec["overlay"]).read_bytes()).hexdigest()


def is_aa_overlay(spec: dict[str, Any]) -> bool:
    """Whether the arm runs the base configuration unchanged: its overlay
    sets nothing but the ``evaluation`` section ``arm_config`` ignores, and
    it queries the run's own index (no ``suffix=``). Such an A/A arm
    measures the noise floor."""
    data = ConfigLoader._read_file(Path(spec["overlay"]))
    return not spec.get("suffix") and not {k for k in data if k != "evaluation"}


def arm_config(
    base: Config, spec: dict[str, Any], *, allow_unknown_keys: bool = False
) -> tuple[Config, list[str]]:
    """The arm's configuration: ``base`` with the overlay merged over it, the
    run's ``evaluation`` section and scoring models (``models.roles``
    ``evaluation`` and ``answer_span``, resolved) kept; and the warnings (an
    overlay that set either, ignored).

    Raises:
        ConfigurationError: The overlay is no YAML mapping, sets an unknown
            key (unless ``allow_unknown_keys``) or makes the config invalid.
    """
    path = Path(spec["overlay"])
    data = ConfigLoader._read_file(path)
    unknown = unknown_keys(data, Config)
    if unknown and not allow_unknown_keys:
        raise ConfigurationError(
            f"--arm {spec['name']}: unknown config keys "
            f"{', '.join(repr(k) for k in unknown)} in '{path}' (fix or remove "
            "them, or pass --allow-unknown-keys)"
        )
    warnings = []
    if "evaluation" in data:
        warnings.append(
            f"--arm {spec['name']}: the overlay's 'evaluation' section is "
            "ignored; every arm is scored with the run's evaluation settings."
        )
        data = {k: v for k, v in data.items() if k != "evaluation"}
    try:
        config = Config.model_validate(merge_over(base.as_input(), data))
    except ValidationError as e:
        raise ConfigurationError(
            f"--arm {spec['name']}: {format_validation_errors('', e)} (in '{path}')"
        ) from e
    resolve_relative_paths(config, path)
    config.evaluation = base.evaluation.model_copy(deep=True)
    # The models that score: the run's judge and answer-span extractor, not
    # the ones the overlay's answer model or tiers would pick.
    roles = (data.get("models") or {}).get("roles") or {}
    scoring = {}
    for role in (ModelRole.EVALUATION, ModelRole.ANSWER_SPAN):
        model = base.model_id(role)
        if config.model_id(role) != model and role.value in roles:
            warnings.append(
                f"--arm {spec['name']}: the overlay's models.roles.{role.value} "
                f"is ignored; every arm is scored with the run's ({model})."
            )
        scoring[role.value] = model
    config.models = config.models.model_copy(
        update={"roles": config.models.roles.model_copy(update=scoring)}
    )
    return config, warnings
