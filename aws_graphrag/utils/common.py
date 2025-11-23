# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import hashlib
import re
import uuid
from typing import Any

RE_INVALID_CHARS = re.compile(r"[^a-z0-9\s]")
RE_EXTRA_SPACES = re.compile(r"\s+")


def compute_hash(data: str, algorithm: str = "sha256", length: int = 16) -> str:
    encoded_data = data.encode("utf-8")

    algorithm = algorithm.lower()
    if algorithm == "md5":
        hash_obj = hashlib.md5(encoded_data, usedforsecurity=False)
    elif algorithm == "sha256":
        hash_obj = hashlib.sha256(encoded_data)
    else:
        raise ValueError(f"Unsupported algorithm: '{algorithm}'")

    return hash_obj.hexdigest()[:length]


def ensure_list(data: Any, inner_key: str | None = None) -> list[Any]:
    if isinstance(data, dict):
        data = data.get(inner_key, []) if inner_key else data

    if not isinstance(data, list):
        return [data] if data else []

    return data


def generate_stable_id(content: str, namespace_key: str = "aws-graphrag") -> str:
    namespace = uuid.uuid5(uuid.NAMESPACE_DNS, namespace_key)
    return str(uuid.uuid5(namespace, content))


def normalize_name(name: str | None) -> str:
    if not name:
        return ""

    normalized = name.lower()

    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = RE_INVALID_CHARS.sub(" ", normalized)
    normalized = RE_EXTRA_SPACES.sub(" ", normalized)

    return normalized.strip()


def safe_float_parse(value: Any, default_value: float | None = None) -> float | None:
    if value is None:
        return default_value

    try:
        return float(value)
    except (ValueError, TypeError):
        return default_value
