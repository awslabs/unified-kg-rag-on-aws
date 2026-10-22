# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
import types
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Union, cast, get_args, get_origin

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr, TypeAdapter, ValidationError

from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import (
    MOVED_KEYS,
    REMOVED_KEYS,
    merge_over,
    removed_keys,
)
from unified_kg_rag.domain.models.provider_config import SQLITE
from unified_kg_rag.shared.exceptions import ConfigurationError

if TYPE_CHECKING:
    from typing_extensions import TypeForm

# Stdlib logger (not get_logger): this module is imported by the logging setup,
# so reaching back through get_logger -> get_config would risk a circular import.
_logger = logging.getLogger(__name__)


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    """The BaseModel behind ``Model`` / ``Model | None`` (not dict/list values)."""
    if get_origin(annotation) not in (None, Union, types.UnionType):
        return None
    for candidate in get_args(annotation) or (annotation,):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def unknown_keys(data: Any, model: type[BaseModel], prefix: str = "") -> list[str]:
    """The config keys in ``data`` the model does not define (pydantic would
    drop them), as dotted paths. Moved keys are left out (validation refuses
    them, naming their replacement), and so are removed ones (ignored with a
    warning)."""
    if not isinstance(data, dict):
        return []
    found: list[str] = []
    for key, value in data.items():
        path = f"{prefix}{key}"
        field = model.model_fields.get(key)
        if field is None:
            if path not in MOVED_KEYS and path not in REMOVED_KEYS:
                found.append(path)
            continue
        nested = _nested_model(field.annotation)
        if nested is not None:
            found.extend(unknown_keys(value, nested, f"{path}."))
    return found


DOTENV_FILENAME = ".env"


def resolve_relative_paths(config: Config, config_path: str | Path) -> None:
    """Resolve relative SQLite store paths against ``config_path``'s directory.

    A store must be the same file on every run against it; relative to the
    working directory, a run started elsewhere would silently use an empty
    registry or store. Applies to each of the doc-status registry and the
    graph and vector stores whose kind is SQLite, so a caller that switches
    the config to it afterwards (``--backend local`` over an ``aws`` config
    file) calls this again.
    """
    providers = config.providers
    for section in (providers.doc_status, providers.graph, providers.vector):
        path = Path(section.sqlite_path)
        if (
            section.kind == SQLITE
            and section.sqlite_path != ":memory:"
            and not path.is_absolute()
        ):
            section.sqlite_path = str((Path(config_path).parent / path).resolve())


def load_cli_dotenv(directory: Path | None = None) -> Path | None:
    """Load ``.env`` from the current directory into ``os.environ`` (CLIs only).

    Only the file in ``directory`` (default: the current working directory)
    is read, never one found by walking up parent directories: a ``.env`` in
    the home directory setting ``AWS_PROFILE`` would otherwise silently switch
    accounts. A variable already set in the environment is never overridden.
    Importing the package loads no ``.env``; library users load their own.

    Returns:
        The file loaded, or ``None`` when there is none.
    """
    path = (directory or Path.cwd()) / DOTENV_FILENAME
    if not path.is_file():
        return None
    load_dotenv(path, override=False)
    _logger.debug("Loaded environment variables from '%s'", path)
    return path


# Environment overrides: UKR__<SECTION>__<KEY>... sets the config key at that
# path (double underscores between path segments, any case), e.g.
# UKR__AWS__NEPTUNE__ENDPOINT or UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO
# (a segment past a free-form mapping names one of its keys). Every key can be
# set this way; docs/operations.md lists the ones a deployment sets.
ENV_PREFIX = "UKR__"
# The AWS SDK's own variables fill the AWS keys a config file leaves unset; an
# explicit value in the file wins, as an explicit client setting wins over the
# environment in the SDKs.
AWS_ENV_FALLBACKS: dict[str, tuple[str, ...]] = {
    "AWS_PROFILE": ("aws", "profile_name"),
    "AWS_REGION": ("aws", "region_name"),
}


def env_var(path: str) -> str:
    """The environment variable that sets the config key ``path``."""
    return ENV_PREFIX + path.replace(".", "__").upper()


def _is_mapping(annotation: Any) -> bool:
    return any(
        get_origin(arg) is dict or arg is dict
        for arg in (annotation, *get_args(annotation))
    )


def _env_path(name: str) -> tuple[str, ...] | None:
    """The config path a ``UKR__`` variable names, or ``None`` if none."""
    segments = tuple(name[len(ENV_PREFIX) :].lower().split("__"))
    model: type[BaseModel] = Config
    for position, segment in enumerate(segments):
        field = model.model_fields.get(segment)
        if field is None:
            return None
        nested = _nested_model(field.annotation)
        rest = len(segments) - position - 1
        if nested is not None:
            model = nested
            continue
        if rest == 0 or (rest == 1 and _is_mapping(field.annotation)):
            return segments
        return None
    return None


def _warn_removed(where: str, path: str) -> None:
    _logger.warning(
        "Config key %s is removed and ignored: %s", where, REMOVED_KEYS[path]
    )


class ConfigLoader:
    """Loads the config from its layers: defaults < ``backend`` preset < file
    < environment (``UKR__`` variables; ``AWS_PROFILE`` / ``AWS_REGION`` only
    under a file that leaves those keys unset).

    The environment is a layer of the input, merged over the file's data before
    the one validation, not assigned to the validated model: an override is
    then part of the input the config keeps (``Config.as_input``), so a config
    rebuilt under other provider flags (``with_provider_overrides``) keeps it.
    Unknown keys, in the file or as ``UKR__`` variables, are an error unless
    ``allow_unknown_keys`` (then a WARNING each).
    """

    def __init__(
        self, config_path: Path | None = None, *, allow_unknown_keys: bool = False
    ) -> None:
        self.config_path = Path(config_path) if config_path else None
        self.allow_unknown_keys = allow_unknown_keys
        self._config: Config | None = None

        # Debug, not print: a module-level ConfigLoader() runs this on every
        # import, and a bare print() to stdout corrupts `run-rag --output-format
        # json` (machine-parseable stdout) and violates the print-vs-logging rule.
        _logger.debug(
            "Config path: '%s'",
            self.config_path.resolve() if self.config_path else None,
        )

    def load_config(self) -> Config:
        if self._config is not None:
            return self._config

        config_data = (
            self._read_file(self.config_path) if self.config_path is not None else {}
        )
        where = f" (in '{self.config_path}')" if self.config_path else ""
        unknown = unknown_keys(config_data, Config)
        for path in removed_keys(config_data):
            _warn_removed(f"'{path}'{where}", path)
        try:
            file_config = Config(**config_data)
        except ValidationError as e:
            raise ConfigurationError(
                f"Configuration validation error: {validation_summary(e)}{where}"
            ) from e
        env_data, unknown_env, env_paths = self._environment_layer(
            file_config, config_data
        )
        try:
            config = (
                Config(**_merged(config_data, env_data)) if env_data else file_config
            )
        except ValidationError as e:
            # The file alone is valid: the environment layer made it invalid.
            raise ConfigurationError(
                f"Configuration validation error: {validation_summary(e)} "
                f"(set by environment variable {_variables_at(e, env_paths)})"
            ) from e
        self._check_unknown(
            [f"'{path}'{where}" for path in unknown]
            + [f"environment variable {name}" for name in unknown_env]
        )
        for path in config.inactive_provider_settings():
            _logger.warning(
                "Config key '%s' is ignored: its provider is not the one selected "
                "(see providers.*.kind)",
                path,
            )
        if self.config_path is not None:
            resolve_relative_paths(config, self.config_path)
        self._config = config
        return config

    def _check_unknown(self, unknown: list[str]) -> None:
        """Refuse unknown keys, or warn on them with ``allow_unknown_keys``.

        A misspelled key would otherwise leave its setting at the default
        (a guardrail identifier under the wrong key leaves queries
        unscreened).
        """
        if not unknown:
            return
        if not self.allow_unknown_keys:
            raise ConfigurationError(
                "Unknown config keys: "
                + ", ".join(unknown)
                + " (fix or remove them, or pass --allow-unknown-keys to "
                "ignore them)"
            )
        for item in unknown:
            _logger.warning("Unknown config key %s is ignored", item)

    @staticmethod
    def _read_file(config_path: Path) -> dict[str, Any]:
        """The raw mapping in the config file ``config_path``.

        Raises:
            FileNotFoundError: The file does not exist.
            ConfigurationError: It cannot be read or is not a YAML mapping.
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: '{config_path}'")
        try:
            with open(config_path, encoding="utf-8") as file:
                config_data = yaml.safe_load(file) or {}
        except UnicodeDecodeError as e:
            raise ConfigurationError(
                f"Configuration file '{config_path}' is not valid UTF-8 "
                f"(byte offset {e.start})"
            ) from e
        except OSError as e:
            raise ConfigurationError(
                f"Cannot read configuration file '{config_path}': {e.strerror or e}"
            ) from e
        except yaml.YAMLError as e:
            # One line: the YAML error names the file, line and column.
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {' '.join(str(e).split())}"
            ) from e
        if not isinstance(config_data, dict):
            raise ConfigurationError(
                f"Configuration file '{config_path}' must be a YAML "
                f"mapping, got {type(config_data).__name__}"
            )
        return config_data

    @staticmethod
    def _environment_layer(
        config: Config, file_data: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str], dict[str, tuple[str, ...]]]:
        """The set environment overrides as raw config data, the ``UKR__``
        variables that name no config key, and the config path each applied
        variable sets.

        Each value is converted and checked against its field in ``config``
        (the config without them), so an invalid one is reported by variable.
        ``AWS_PROFILE`` / ``AWS_REGION`` fill only
        the keys ``file_data`` leaves unset.

        Raises:
            ConfigurationError: An environment variable holds an invalid value.
        """
        overrides: dict[str, tuple[str, ...]] = {}
        unknown: list[str] = []
        removed = {env_var(path): path for path in REMOVED_KEYS}
        for name in sorted(os.environ):
            if name.upper().startswith(ENV_PREFIX):
                path = _env_path(name)
                if name.upper() in removed:
                    _warn_removed(f"environment variable {name}", removed[name.upper()])
                elif path is None:
                    unknown.append(name)
                else:
                    overrides[name] = path
        for name, path in AWS_ENV_FALLBACKS.items():
            if (
                name in os.environ
                and path not in overrides.values()
                and not _is_set(file_data, path)
            ):
                overrides[name] = path
        layer: dict[str, Any] = {}
        for name, path in overrides.items():
            value = os.environ[name]
            if path == ("logging", "level"):
                value = value.upper()  # logging accepts any case
            try:
                _set_env_value(layer, config, path, value)
            except (ValueError, ValidationError) as e:
                # The value is left out: it may be a secret.
                reason = (
                    "; ".join(
                        detail["msg"]
                        for detail in e.errors(include_url=False, include_input=False)
                    )
                    if isinstance(e, ValidationError)
                    else "not a valid value"
                )
                raise ConfigurationError(
                    f"Invalid environment variable {name} "
                    f"(overrides {'.'.join(path)}): {reason}"
                ) from e
        return layer, unknown, overrides

    @property
    def config(self) -> Config:
        return self._config if self._config is not None else self.load_config()


def _merged(file_data: dict[str, Any], env_data: dict[str, Any]) -> dict[str, Any]:
    """``env_data`` over ``file_data``. The layer writes ``index_settings``
    whole, with the variable's spelling of a setting in place of the
    other: it replaces the file's map rather than merging into it, which
    would bring the other spelling back."""
    merged = merge_over(file_data, env_data)
    opensearch = env_data.get("indexing", {}).get("opensearch", {})
    if "index_settings" in opensearch:
        merged["indexing"]["opensearch"]["index_settings"] = opensearch[
            "index_settings"
        ]
    return merged


def _variables_at(error: ValidationError, paths: dict[str, tuple[str, ...]]) -> str:
    """The environment variables setting a key ``error`` reports (inside it,
    or a section around it); all of them when none does."""
    locations = [
        tuple(str(part) for part in detail["loc"])
        for detail in error.errors(include_url=False, include_input=False)
    ]
    named = [
        name
        for name, path in paths.items()
        if any(path[: len(loc)] == loc or loc[: len(path)] == path for loc in locations)
    ]
    return ", ".join(named or paths)


def _is_set(data: Any, path: tuple[str, ...]) -> bool:
    for key in path:
        if not isinstance(data, dict) or data.get(key) is None:
            return False
        data = data[key]
    return True


def _set_env_value(
    layer: dict[str, Any], config: Config, path_tuple: tuple[str, ...], value: str
) -> None:
    """Write the environment value ``value`` for ``path_tuple`` into ``layer``.

    ``value`` is converted to the type of the field in ``config`` and checked
    against its constraints, as the same value in the file would be. A key of a
    free-form mapping (``indexing.opensearch.index_settings``) is coerced from
    the type of the value it already holds, and the whole mapping is written
    (the file's or the default one with that key replaced), since a mapping in
    the input replaces the default rather than merging into it.

    Raises:
        ValueError: ``value`` is not a number for a numeric setting.
        ValidationError: ``value`` breaks the field's constraints.
    """
    current: Any = config
    target = layer
    for key in path_tuple[:-1]:
        current = getattr(current, key)
        if isinstance(current, dict):
            # The free-form mapping (the last step): written whole.
            target = target.setdefault(key, dict(current))
            break
        target = target.setdefault(key, {})

    final_key = path_tuple[-1]
    is_mapping = isinstance(current, dict)
    # An index setting has two spellings (with and without "index."): the
    # variable replaces the one the file or default used.
    other_spelling = (
        final_key.removeprefix("index.")
        if final_key.startswith("index.")
        else f"index.{final_key}"
    )
    spelled = is_mapping and path_tuple[-2] == "index_settings"
    current_value = (
        current.get(final_key, current.get(other_spelling) if spelled else None)
        if is_mapping
        else getattr(current, final_key)
    )

    # Coerce the string env value to the field's declared type. SecretStr is
    # handled by field annotation (current_value may be None at default) so a
    # value set from the environment stays masked and exposes
    # .get_secret_value().
    annotation = (
        None if is_mapping else type(current).model_fields[final_key].annotation
    )
    is_secret = current_value is not None and isinstance(current_value, SecretStr)
    if not is_secret and annotation is not None:
        is_secret = SecretStr in getattr(annotation, "__args__", (annotation,))

    parsed_value: bool | int | float | str | SecretStr
    if is_secret:
        parsed_value = SecretStr(value)
    elif isinstance(current_value, bool):
        parsed_value = value.lower() in ("true", "1", "yes", "on")
    elif isinstance(current_value, int):
        parsed_value = int(value)
    elif isinstance(current_value, float):
        parsed_value = float(value)
    elif isinstance(current_value, (list, dict)):
        # A list or mapping setting takes JSON: UKR__X__Y='["a", "b"]'.
        parsed_value = json.loads(value)
    else:
        parsed_value = value

    if is_mapping:
        if spelled:
            target.pop(other_spelling, None)
        target[final_key] = parsed_value
        return
    # Checked against the field's type and constraints here, so an invalid
    # value is reported by variable. The Annotated form is built at runtime
    # from the field, so a type checker cannot see it as the TypeForm that
    # TypeAdapter is typed to take (pydantic >= 2.14); it is one, hence the
    # cast.
    field = type(current).model_fields[final_key]
    field_type = cast("TypeForm[Any]", Annotated[field.annotation, field])
    target[final_key] = TypeAdapter(field_type).validate_python(parsed_value)


_config_loader = ConfigLoader()


def validation_summary(error: ValidationError) -> str:
    """One line naming each invalid key and why, without the rejected values.

    The values are left out: a misplaced secret (a password under the wrong
    key) would otherwise be echoed to the terminal and the logs.
    """
    return "; ".join(
        f"{'.'.join(str(part) for part in detail['loc']) or '<root>'}: "
        f"{detail['msg']}"
        for detail in error.errors(include_url=False, include_input=False)
    )


def get_config(
    config_path: str | Path | None = None, *, allow_unknown_keys: bool = False
) -> Config:
    """The config of ``config_path`` (and the environment); without a path,
    the defaults and the environment, loaded once.

    Raises:
        ConfigurationError: An unknown key is set (unless
            ``allow_unknown_keys``) or a value is invalid.
    """
    if config_path is not None or allow_unknown_keys:
        loader = ConfigLoader(
            Path(config_path) if config_path is not None else None,
            allow_unknown_keys=allow_unknown_keys,
        )
        return loader.load_config()
    return _config_loader.config
