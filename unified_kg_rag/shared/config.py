# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import os
import types
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Union, cast, get_args, get_origin

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr, TypeAdapter, ValidationError

from unified_kg_rag.domain.models import Config
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


def warn_unknown_keys(data: Any, model: type[BaseModel], prefix: str = "") -> None:
    """Warn on config keys the model does not define (pydantic drops them)."""
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        path = f"{prefix}{key}"
        field = model.model_fields.get(key)
        if field is None:
            _logger.warning("Unknown config key '%s' is ignored", path)
            continue
        nested = _nested_model(field.annotation)
        if nested is not None:
            warn_unknown_keys(value, nested, f"{path}.")


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


# Environment variable -> the config field it overrides. A path ending inside a
# free-form mapping (index_settings, language_analyzers) sets that one key.
ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "AWS_PROFILE": ("aws", "profile_name"),
    "AWS_REGION": ("aws", "region_name"),
    "LOG_LEVEL": ("logging", "level"),
    "LOG_FORMAT": ("logging", "log_format"),
    "LOG_TO_FILE": ("logging", "log_to_file"),
    "LOG_FILE_PATH": ("logging", "log_file_path"),
    "NEPTUNE_ENDPOINT": ("aws", "neptune", "endpoint"),
    "OPENSEARCH_ENDPOINT": ("aws", "opensearch", "endpoint"),
    "OPENSEARCH_USERNAME": ("aws", "opensearch", "username"),
    "OPENSEARCH_PASSWORD": ("aws", "opensearch", "password"),
    "BEDROCK_REGION": ("aws", "bedrock", "region_name"),
    "BEDROCK_GUARDRAIL_IDENTIFIER": ("aws", "bedrock", "guardrail", "identifier"),
    "BEDROCK_GUARDRAIL_VERSION": ("aws", "bedrock", "guardrail", "version"),
    "S3_BUCKET_NAME": ("aws", "s3", "bucket_name"),
    # Injected by the CDK compute stack from the domain's node count. Applies
    # to indices created from now on; existing indices keep theirs until
    # changed with `PUT <index>/_settings`.
    "OPENSEARCH_NUMBER_OF_REPLICAS": (
        "indexing",
        "opensearch",
        "index_settings",
        "number_of_replicas",
    ),
    # Injected by the CDK compute stack: Amazon OpenSearch Service has no nori
    # unless its optional plugin is associated, so the IaC maps Korean to an
    # analyzer the domain has (the built-in cjk).
    "OPENSEARCH_KOREAN_ANALYZER": (
        "indexing",
        "opensearch",
        "language_analyzers",
        "ko",
    ),
    # Injected by the CDK compute stack so the app uses the IaC-managed
    # doc-status table (and never auto-creates an unmanaged one).
    "GRAPHRAG_DOC_STATUS_TABLE": ("aws", "dynamodb", "table_name"),
    # Set by the container entrypoint to the S3 URI the corpus was synced
    # from, so incremental deletion is scoped to that corpus rather than the
    # shared local staging directory.
    "GRAPHRAG_SOURCE_SCOPE": ("processing", "document_parsing", "source_scope"),
    "GRAPHRAG_DOC_STATUS_CREATE_TABLE": (
        "aws",
        "dynamodb",
        "create_table_if_missing",
    ),
    # Injected by the CDK compute stack when enable_cloudwatch_tracing adds the
    # ADOT collector sidecar, so the baked-in image config can stay
    # tracing-free.
    "GRAPHRAG_OBSERVABILITY_ENABLED": ("observability", "enabled"),
    "GRAPHRAG_OBSERVABILITY_BACKEND": ("observability", "backend"),
}


def _merge_over(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    """``base`` with every leaf of ``top`` written over it (nested dicts merge)."""
    merged = dict(base)
    for key, value in top.items():
        below = merged.get(key)
        merged[key] = (
            _merge_over(below, value)
            if isinstance(value, dict) and isinstance(below, dict)
            else value
        )
    return merged


class ConfigLoader:
    """Loads the config from its layers: defaults < ``backend`` preset < file
    < environment (:data:`ENV_OVERRIDES`).

    The environment is a layer of the input, merged over the file's data before
    the one validation, not assigned to the validated model: an override is
    then part of what the config was built from (``model_fields_set``), so a
    config rebuilt from ``model_dump(exclude_unset=True)`` (the CLI provider
    flags, ``with_provider_overrides``) keeps it.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = Path(config_path) if config_path else None
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
        try:
            warn_unknown_keys(config_data, Config)
            file_config = Config(**config_data)
            env_data = self._environment_layer(file_config)
            config = (
                Config(**_merge_over(config_data, env_data))
                if env_data
                else file_config
            )
        except ValidationError as e:
            raise ConfigurationError(
                f"Configuration validation error: {validation_summary(e)}{where}"
            ) from e
        if self.config_path is not None:
            resolve_relative_paths(config, self.config_path)
        self._config = config
        return config

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
    def _environment_layer(config: Config) -> dict[str, Any]:
        """The set environment overrides as raw config data.

        Each value is converted and checked against its field in ``config``
        (the config without them), so an invalid one is reported by variable.

        Raises:
            ConfigurationError: An environment variable holds an invalid value.
        """
        layer: dict[str, Any] = {}
        for env_var, config_path in ENV_OVERRIDES.items():
            env_value = os.getenv(env_var)
            if env_value is None:
                continue
            if env_var == "LOG_LEVEL":
                env_value = env_value.upper()  # logging accepts any case
            try:
                _set_env_value(layer, config, config_path, env_value)
            except (ValueError, ValidationError) as e:
                # The value is left out: it may be a secret.
                reason = (
                    "; ".join(
                        detail["msg"]
                        for detail in e.errors(include_url=False, include_input=False)
                    )
                    if isinstance(e, ValidationError)
                    else "not a valid number"
                )
                raise ConfigurationError(
                    f"Invalid environment variable {env_var} "
                    f"(overrides {'.'.join(config_path)}): {reason}"
                ) from e
        return layer

    def reload_config(self) -> Config:
        self._config = None
        return self.load_config()

    @property
    def config(self) -> Config:
        return self._config if self._config is not None else self.load_config()


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
    current_value = (
        current.get(final_key) if is_mapping else getattr(current, final_key)
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
    else:
        parsed_value = value

    if is_mapping:
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


def get_config(config_path: str | Path | None = None) -> Config:
    if config_path is not None:
        loader = ConfigLoader(Path(config_path))
        return loader.load_config()
    return _config_loader.config


def reload_config(config_path: str | Path | None = None) -> Config:
    if config_path is not None:
        loader = ConfigLoader(Path(config_path))
        return loader.load_config()
    return _config_loader.reload_config()
