# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import SecretStr, ValidationError

from aws_graphrag.domain.models import Config


class ConfigLoader:
    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = Path(config_path) if config_path else None
        self._config: Config | None = None
        load_dotenv()

        print(
            f"Config path: '{self.config_path.resolve()}'"
            if self.config_path
            else "No config path provided, using default configuration"
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
                config_data = yaml.safe_load(file)

            self._config = Config(**config_data)
            self._apply_environment_overrides()
            return self._config

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}") from e
        except ValidationError as e:
            raise ValueError(f"Configuration validation error: {e}") from e

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
            "S3_BUCKET_NAME": ("aws", "s3", "bucket_name"),
        }

        for env_var, config_path in env_overrides.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_config_value(config_path, env_value)

    def _set_nested_config_value(self, path_tuple: tuple[str, ...], value: str) -> None:
        if self._config is None:
            return

        current = self._config
        for key in path_tuple[:-1]:
            current = getattr(current, key)

        final_key = path_tuple[-1]
        current_value = getattr(current, final_key)

        # Coerce the string env value to the field's declared type. SecretStr is
        # handled by field annotation (current_value may be None at default) so a
        # value set from the environment stays masked and exposes
        # .get_secret_value() regardless of validate_assignment.
        annotation = type(current).model_fields[final_key].annotation
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

        setattr(current, final_key, parsed_value)

    def reload_config(self) -> Config:
        self._config = None
        return self.load_config()

    @property
    def config(self) -> Config:
        return self._config if self._config is not None else self.load_config()


_config_loader = ConfigLoader()


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
