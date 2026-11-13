# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``run-eval --arm NAME=overlay.yaml[,suffix=SUFFIX][,aa=true]``: a
configuration arm.

The overlay is a YAML mapping merged over the run's configuration (as the
environment layer is over a config file), validated like a config file
(unknown keys refused unless ``--allow-unknown-keys``), the relative paths
it sets resolved against its own directory (the others stay the run's).
``suffix=`` queries another index
namespace with the arm (``--suffix``'s, which the overlay's index settings
otherwise follow). The arm is scored exactly as the base runs are: its
``evaluation`` section is the run's. ``aa=true`` declares it an A/A arm
(``is_aa_arm``) whatever its overlay sets.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from unified_kg_rag.application.config import ConfigLoader, resolve_relative_paths
from unified_kg_rag.domain.models import Config, ModelRole
from unified_kg_rag.domain.models.config import (
    unknown_keys,
    without_keys,
)
from unified_kg_rag.shared.exceptions import ConfigurationError
from unified_kg_rag.shared.utils.validation import format_validation_errors

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
# Labels a comparison already gives a meaning.
_RESERVED = frozenset({"closed_book", "baseline"})
# Settings that size a cache and change no answer: an arm whose resolved
# configuration differs from the run's only in these is an A/A arm.
CACHE_ONLY_SETTINGS: tuple[tuple[str, ...], ...] = (
    ("search", "query_extraction_cache_size"),
    ("search", "token_manager", "token_count_cache_size"),
)
_BOOLEANS = {"true": True, "false": False}


def arm_spec(value: str) -> dict[str, Any]:
    """argparse ``type`` of ``--arm``: ``{"name", "overlay", "suffix"}``,
    and ``"aa"`` when ``aa=true|false`` is given."""
    name, sep, rest = value.partition("=")
    name = name.strip()
    if not sep or not rest.strip():
        raise argparse.ArgumentTypeError(
            f"'{value}' is not NAME=overlay.yaml[,suffix=SUFFIX][,aa=true]"
        )
    if not _NAME.match(name) or name.lower() in _RESERVED:
        raise argparse.ArgumentTypeError(
            f"arm name '{name}' must be letters, digits, '_', '.' or '-' (not "
            f"{', '.join(sorted(_RESERVED))})"
        )
    path, *options = rest.split(",")
    suffix: str | None = None
    aa: bool | None = None
    for option in options:
        key, eq, setting = (part.strip() for part in option.partition("="))
        if key == "suffix" and eq and setting:
            suffix = setting
        elif key == "aa" and setting.lower() in _BOOLEANS:
            aa = _BOOLEANS[setting.lower()]
        else:
            raise argparse.ArgumentTypeError(
                f"unknown --arm option '{option}' (only suffix=SUFFIX and "
                "aa=true|false)"
            )
    overlay = Path(path.strip())
    if not overlay.is_file():
        raise argparse.ArgumentTypeError(f"overlay file not found: '{overlay}'")
    spec: dict[str, Any] = {"name": name, "overlay": str(overlay), "suffix": suffix}
    if aa is not None:
        spec["aa"] = aa
    return spec


def overlay_sha256(spec: dict[str, Any]) -> str:
    """The digest of the arm's overlay file, as the run manifest records it."""
    return hashlib.sha256(Path(spec["overlay"]).read_bytes()).hexdigest()


def is_aa_arm(base: Config, config: Config, spec: dict[str, Any]) -> bool:
    """Whether the arm (``spec``, its configuration ``config`` by
    ``arm_config``) runs the base configuration ``base`` unchanged: its
    ``aa=`` option when given, else whether it queries the run's own index
    (no ``suffix=``) with the same resolved configuration, cache sizes
    (``CACHE_ONLY_SETTINGS``) aside. An overlay that sets only what
    ``arm_config`` ignores, or a value equal to the run's, counts. Such an
    A/A arm measures the noise floor."""
    if spec.get("aa") is not None:
        return bool(spec["aa"])
    return not spec.get("suffix") and _without_caches(config) == _without_caches(base)


# What arm_config sets to the run's own on every arm: the scoring models
# (resolved there, unset on the run's config until the judge is chosen).
_SCORING_SETTINGS = tuple(
    ("models", "roles", role.value)
    for role in (ModelRole.EVALUATION, ModelRole.ANSWER_SPAN)
)


def _without_caches(config: Config) -> dict[str, Any]:
    """``config`` as data, without its cache sizes and scoring models."""
    data = config.model_dump(mode="json")
    for *parents, key in (*CACHE_ONLY_SETTINGS, *_SCORING_SETTINGS):
        section = data
        for parent in parents:
            section = section.get(parent) or {}
        section.pop(key, None)
    return data


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
    data = without_keys(data, unknown)
    warnings = []
    if "evaluation" in data:
        warnings.append(
            f"--arm {spec['name']}: the overlay's 'evaluation' section is "
            "ignored; every arm is scored with the run's evaluation settings."
        )
        data = {k: v for k, v in data.items() if k != "evaluation"}
    try:
        config = base.updated(overlay=data)
    except ValidationError as e:
        raise ConfigurationError(
            f"--arm {spec['name']}: {format_validation_errors('', e)} (in '{path}')"
        ) from e
    # Only the overlay's own paths: the rest are the run's, as it reads them.
    resolve_relative_paths(config, path, set_in=data)
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
    # Scored as the run: its evaluation settings and scoring models.
    try:
        config = config.updated(
            {
                "evaluation": base.evaluation,
                **{f"models.roles.{role}": model for role, model in scoring.items()},
            }
        )
    except ValidationError as e:
        raise ConfigurationError(
            f"--arm {spec['name']}: {format_validation_errors('', e)} (in '{path}')"
        ) from e
    return config, warnings
