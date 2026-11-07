# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Settings page's server side: the config file, validated as a run
loads it, diffed before it is written, secrets never shown or written.

Secrets are the ``SecretStr`` fields of :class:`Config` (found from its JSON
schema: ``format: password``). Every response masks them; a file or an edit
that sets one is refused, naming the ``UKR__`` environment variable that
should hold it instead, so the console never writes a credential to disk.
"""

from __future__ import annotations

import difflib
import hashlib
import importlib.util
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from unified_kg_rag.application.config import ConfigLoader, env_var, unknown_keys
from unified_kg_rag.domain.models import Config, Methodology
from unified_kg_rag.shared.exceptions import ConfigurationError

MASK = "********"
# The tracing variables the observability panel reports as set or not.
TRACING_VARIABLES = (
    "LANGFUSE_HOST",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "PHOENIX_COLLECTOR_ENDPOINT",
    "PHOENIX_API_KEY",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_SERVICE_NAME",
    "OTEL_SDK_DISABLED",
)


# The keys that name where a provider sends its requests (and with them the
# credentials the environment holds for it: an API key, a store password,
# signed AWS requests). Changing one is confirmed on Save.
ENDPOINT_KEYS = frozenset({"base_url", "endpoint", "host", "port", "url"})
ENDPOINT_SECTIONS = ("providers", "aws")


class EndpointNotConfirmed(ConfigurationError):
    """A save that changes a provider endpoint without confirming it."""

    def __init__(self, paths: list[str]) -> None:
        super().__init__(
            "the change sends requests to another address ("
            + ", ".join(paths)
            + "), with the credentials the environment holds for it: confirm "
            "the endpoints to save it"
        )
        self.paths = paths


def _flat(data: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(data, dict):
        return {prefix: data}
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flat.update(_flat(value, path) if isinstance(value, dict) else {path: value})
    return flat


def endpoint_changes(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    """The provider endpoints ``new`` changes from ``old``: where requests,
    and the credentials sent with them, would go."""
    before, after = _flat(old), _flat(new)
    changes = []
    for path in sorted(set(before) | set(after)):
        if path.split(".")[0] not in ENDPOINT_SECTIONS:
            continue
        if path.rsplit(".", 1)[-1] not in ENDPOINT_KEYS:
            continue
        if before.get(path) != after.get(path):
            changes.append(
                {"path": path, "from": before.get(path), "to": after.get(path)}
            )
    return changes


class SecretRefused(ConfigurationError):
    """A config file or edit sets a secret: not written, not shown."""


def schema() -> dict[str, Any]:
    """The JSON schema of :class:`Config`: types, defaults and each field's
    description, which the page shows as help."""
    return Config.model_json_schema()


def _resolve(schema_doc: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return dict(schema_doc["$defs"][ref.removeprefix("#/$defs/")])
    return node


def _secret_paths(
    schema_doc: dict[str, Any],
    node: dict[str, Any],
    prefix: tuple[str, ...],
    depth: int = 0,
) -> Iterator[tuple[str, ...]]:
    node = _resolve(schema_doc, node)
    if depth > 12:
        return
    if node.get("format") == "password":
        yield prefix
        return
    for option in node.get("anyOf") or node.get("allOf") or []:
        yield from _secret_paths(schema_doc, option, prefix, depth + 1)
    for name, child in (node.get("properties") or {}).items():
        yield from _secret_paths(schema_doc, child, (*prefix, name), depth + 1)
    if isinstance(node.get("items"), dict):
        yield from _secret_paths(schema_doc, node["items"], (*prefix, "*"), depth + 1)
    if isinstance(node.get("additionalProperties"), dict):
        yield from _secret_paths(
            schema_doc, node["additionalProperties"], (*prefix, "*"), depth + 1
        )


def secret_paths() -> list[tuple[str, ...]]:
    """The paths of every secret field (``*`` for a list item or map key)."""
    doc = schema()
    return sorted(set(_secret_paths(doc, doc, ())))


def _matches(
    data: Any, pattern: tuple[str, ...], at: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], Any]]:
    """The values of ``data`` at ``pattern`` (``*`` matches any key/index)."""
    if not pattern:
        yield at, data
        return
    head, rest = pattern[0], pattern[1:]
    if isinstance(data, dict):
        keys = list(data) if head == "*" else [head] if head in data else []
        for key in keys:
            yield from _matches(data[key], rest, (*at, str(key)))
    elif isinstance(data, list) and head == "*":
        for index, item in enumerate(data):
            yield from _matches(item, rest, (*at, str(index)))


def set_secrets(data: Any) -> list[str]:
    """The dotted paths of the secrets ``data`` (a config mapping) sets."""
    found = []
    for pattern in secret_paths():
        for path, value in _matches(data, pattern):
            if value not in (None, ""):
                found.append(".".join(path))
    return sorted(found)


def masked(data: Any) -> Any:
    """``data`` with every secret replaced by :data:`MASK`."""
    if not isinstance(data, dict):
        return data
    copy = yaml.safe_load(yaml.safe_dump(data)) or {}
    for pattern in secret_paths():
        for path, value in list(_matches(copy, pattern)):
            if value in (None, ""):
                continue
            target = copy
            for key in path[:-1]:
                target = target[int(key)] if isinstance(target, list) else target[key]
            last = path[-1]
            if isinstance(target, list):
                target[int(last)] = MASK
            else:
                target[last] = MASK
    return copy


def _secret_error(paths: list[str]) -> SecretRefused:
    variables = ", ".join(f"{p} ({env_var(p)})" for p in paths)
    return SecretRefused(
        f"the config sets a secret: {variables}. The console does not show or "
        "write secrets: remove it from the file and set the environment "
        "variable instead, then restart the console"
    )


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse(text: str) -> dict[str, Any]:
    """The mapping ``text`` holds.

    Raises:
        ConfigurationError: Not YAML, or not a mapping.
    """
    try:
        data = yaml.safe_load(text) if text.strip() else {}
    except yaml.YAMLError as e:
        raise ConfigurationError(f"not valid YAML: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError("the config file must hold a mapping")
    return data


def _load(data: dict[str, Any], config_path: Path | None) -> Config:
    """``data`` loaded as the file at ``config_path`` would be (relative
    paths resolve against its directory), with the same layers, checks and
    errors as a run: a loader whose file read returns ``data``."""

    class _DataLoader(ConfigLoader):
        @staticmethod
        def _read_file(config_path: Path) -> dict[str, Any]:  # noqa: ARG004
            return data

    # Without a path the base loader reads no file: name one in the working
    # directory, as a run without --config-path resolves against it.
    return _DataLoader(config_path or Path("console-config.yaml")).load_config()


def validate(data: dict[str, Any], config_path: Path | None) -> dict[str, Any]:
    """Whether ``data`` loads as a run would load it (unknown keys and the
    ``UKR__`` environment included), with the errors and warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    unknown = unknown_keys(data, Config)
    try:
        config = _load(data, config_path)
    except ConfigurationError as e:
        errors.append(str(e))
    else:
        warnings.extend(
            f"'{p}' is ignored: its provider is not the one selected"
            for p in config.inactive_provider_settings()
        )
        warnings.extend(config.ineffective_settings())
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "unknown_keys": unknown,
    }


def apply_changes(
    data: dict[str, Any], changes: list[dict[str, Any]]
) -> dict[str, Any]:
    """``data`` with each change applied: ``{"path": "a.b", "value": v}``
    sets a key, ``"value": None`` with ``"unset": true`` removes it.

    Raises:
        ConfigurationError: A malformed path.
    """
    result = yaml.safe_load(yaml.safe_dump(data)) or {}
    for change in changes:
        path = str(change.get("path") or "")
        keys = [k for k in path.split(".") if k]
        if not keys or any(not k.replace("_", "").isalnum() for k in keys):
            raise ConfigurationError(f"'{path}' is not a config key")
        target = result
        for key in keys[:-1]:
            child = target.get(key)
            if not isinstance(child, dict):
                child = {}
                target[key] = child
            target = child
        if change.get("unset"):
            target.pop(keys[-1], None)
        else:
            target[keys[-1]] = change.get("value")
    pruned: dict[str, Any] = _prune(result)
    return pruned


def _prune(data: Any) -> Any:
    """``data`` without the empty mappings unsetting keys left behind."""
    if isinstance(data, dict):
        pruned = {k: _prune(v) for k, v in data.items()}
        return {k: v for k, v in pruned.items() if v != {}}
    return data


def dump(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True) if data else ""


class SettingsFile:
    """The config file the console was started with (or none)."""

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def read(self) -> str:
        if self.path is None or not self.path.exists():
            return ""
        return self.path.read_text(encoding="utf-8")

    def view(self) -> dict[str, Any]:
        """The file (masked) and its digest, the effective config (masked),
        the methodologies and the tracing environment."""
        text = self.read()
        secrets: list[str] = []
        data: dict[str, Any] = {}
        parse_error = None
        try:
            data = parse(text)
            secrets = set_secrets(data)
        except ConfigurationError as e:
            parse_error = str(e)
        effective: dict[str, Any] | None = None
        load_error = None
        try:
            effective = _load(data, self.path).model_dump(mode="json")
        except ConfigurationError as e:
            load_error = str(e)
        return {
            "path": str(self.path) if self.path else None,
            "writable": self.path is not None,
            "sha": digest(text),
            # The text itself only when it holds no secret: masking YAML
            # text in place could not be trusted to catch every form.
            "text": text if not secrets else None,
            "data": masked(data),
            "secrets_in_file": secrets,
            "parse_error": parse_error,
            "load_error": load_error,
            "effective": masked(effective) if effective else None,
            "methodologies": [m.value for m in Methodology],
            "tracing_environment": {
                name: name in os.environ for name in TRACING_VARIABLES
            },
            "otel_installed": importlib.util.find_spec("opentelemetry") is not None,
        }

    def preview(
        self, *, text: str | None, changes: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        """The file as it would be written, its diff and its validation.

        Raises:
            SecretRefused: The file or the result sets a secret.
            ConfigurationError: The text is not a YAML mapping.
        """
        current = self.read()
        base = parse(current)
        if text is not None:
            new_data = parse(text)
            new_text = text
        else:
            if set_secrets(base):
                raise _secret_error(set_secrets(base))
            new_data = apply_changes(base, changes or [])
            new_text = dump(new_data)
        if secrets := set_secrets(new_data):
            raise _secret_error(secrets)
        diff = "".join(
            difflib.unified_diff(
                current.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"{self.path.name if self.path else 'config'} (on disk)",
                tofile=f"{self.path.name if self.path else 'config'} (edited)",
            )
        )
        comments_dropped = text is None and any(
            line.lstrip().startswith("#") for line in current.splitlines()
        )
        return {
            "text": new_text,
            "diff": diff,
            "endpoint_changes": endpoint_changes(base, new_data),
            "changed": new_text != current,
            "comments_dropped": comments_dropped,
            **validate(new_data, self.path),
        }

    def save(
        self, *, text: str, base_sha: str, confirm_endpoints: list[str] | None = None
    ) -> dict[str, Any]:
        """Write ``text`` to the file if it validates, the file did not
        change since ``base_sha`` was read, and every provider endpoint it
        changes is in ``confirm_endpoints``.

        Raises:
            SecretRefused: ``text`` sets a secret.
            ConfigurationError: No file to write, a changed file, or a
                config that does not validate.
        """
        if self.path is None:
            raise ConfigurationError(
                "the console was started without --config-path: there is no file to save to"
            )
        current = self.read()
        if digest(current) != base_sha:
            raise ConfigurationError(
                f"'{self.path.name}' changed on disk since it was loaded; reload it and redo the change"
            )
        data = parse(text)
        if secrets := set_secrets(data):
            raise _secret_error(secrets)
        result = validate(data, self.path)
        if not result["valid"]:
            raise ConfigurationError("; ".join(result["errors"]))
        changed = [c["path"] for c in endpoint_changes(parse(current), data)]
        if changed and sorted(confirm_endpoints or []) != changed:
            raise EndpointNotConfirmed(changed)
        # Written whole or not at all: a temporary file beside it, renamed
        # over it.
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        return {"sha": digest(text), **result}
