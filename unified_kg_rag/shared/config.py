# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import os
import types
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr, ValidationError

from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.provider_config import SQLITE
from unified_kg_rag.shared.exceptions import ConfigurationError

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
    """Resolve a relative SQLite registry path against ``config_path``'s directory.

    The registry must be the same file on every run against the same
    stores; relative to the working directory, a run started elsewhere would
    silently use an empty registry. Applies when the registry is SQLite, so a
    caller that switches the config to it afterwards (``--backend local``
    over an ``aws`` config file) calls this again.
    """
    doc_status = config.providers.doc_status
    path = Path(doc_status.sqlite_path)
    if (
        doc_status.kind == SQLITE
        and doc_status.sqlite_path != ":memory:"
        and not path.is_absolute()
    ):
        doc_status.sqlite_path = str((Path(config_path).parent / path).resolve())


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


class ConfigLoader:
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

        if self.config_path is None:
            self._config = Config()
            self._apply_environment_overrides()
            return self._config

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: '{self.config_path}'"
            )

        try:
            with open(self.config_path, encoding="utf-8") as file:
                config_data = yaml.safe_load(file) or {}
            if not isinstance(config_data, dict):
                raise ConfigurationError(
                    f"Configuration file '{self.config_path}' must be a YAML "
                    f"mapping, got {type(config_data).__name__}"
                )

            warn_unknown_keys(config_data, Config)
            self._config = Config(**config_data)
            self._apply_environment_overrides()
            self._resolve_relative_paths()
            return self._config

        except yaml.YAMLError as e:
            # One line: the YAML error names the file, line and column.
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {' '.join(str(e).split())}"
            ) from e
        except ValidationError as e:
            raise ConfigurationError(
                f"Configuration validation error: {validation_summary(e)} "
                f"(in '{self.config_path}')"
            ) from e

    def _resolve_relative_paths(self) -> None:
        """Resolve a relative SQLite registry path against the config file's
        directory (see :func:`resolve_relative_paths`)."""
        if self._config is not None and self.config_path is not None:
            resolve_relative_paths(self._config, self.config_path)

    def _apply_environment_overrides(self) -> None:
        if self._config is None:
            return
        env_overrides = {
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
            "BEDROCK_GUARDRAIL_IDENTIFIER": (
                "aws",
                "bedrock",
                "guardrail",
                "identifier",
            ),
            "BEDROCK_GUARDRAIL_VERSION": ("aws", "bedrock", "guardrail", "version"),
            "S3_BUCKET_NAME": ("aws", "s3", "bucket_name"),
            # Injected by the CDK compute stack from the domain's node count.
            # Applies to indices created from now on; existing indices keep
            # theirs until changed with `PUT <index>/_settings`.
            "OPENSEARCH_NUMBER_OF_REPLICAS": (
                "indexing",
                "opensearch",
                "index_settings",
                "number_of_replicas",
            ),
            # Injected by the CDK compute stack so the app uses the IaC-managed
            # doc-status table (and never auto-creates an unmanaged one).
            "GRAPHRAG_DOC_STATUS_TABLE": ("aws", "dynamodb", "table_name"),
            # Set by the container entrypoint to the S3 URI the corpus was
            # synced from, so incremental deletion is scoped to that corpus
            # rather than the shared local staging directory.
            "GRAPHRAG_SOURCE_SCOPE": (
                "processing",
                "document_parsing",
                "source_scope",
            ),
            "GRAPHRAG_DOC_STATUS_CREATE_TABLE": (
                "aws",
                "dynamodb",
                "create_table_if_missing",
            ),
        }

        for env_var, config_path in env_overrides.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_config_value(config_path, env_value)

    def _set_nested_config_value(self, path_tuple: tuple[str, ...], value: str) -> None:
        if self._config is None:
            return

        current: Any = self._config
        for key in path_tuple[:-1]:
            current = getattr(current, key)

        final_key = path_tuple[-1]
        # A free-form settings mapping (indexing.opensearch.index_settings) is
        # coerced like a field, from the type of the value it already holds.
        is_mapping = isinstance(current, dict)
        current_value = (
            current.get(final_key) if is_mapping else getattr(current, final_key)
        )

        # Coerce the string env value to the field's declared type. SecretStr is
        # handled by field annotation (current_value may be None at default) so a
        # value set from the environment stays masked and exposes
        # .get_secret_value() regardless of validate_assignment.
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
            current[final_key] = parsed_value
        else:
            setattr(current, final_key, parsed_value)

    def reload_config(self) -> Config:
        self._config = None
        return self.load_config()

    @property
    def config(self) -> Config:
        return self._config if self._config is not None else self.load_config()


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
