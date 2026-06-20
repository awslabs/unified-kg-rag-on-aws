# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import hashlib
import re
import uuid
from typing import Any

RE_INVALID_CHARS = re.compile(r"[^a-z0-9\s]")
RE_EXTRA_SPACES = re.compile(r"\s+")


def compute_hash(data: str, algorithm: str = "sha256", length: int = 16) -> str:
    """Compute a truncated content hash for dedup / cache-key / id purposes.

    SHA-256 is used for all hashing (no MD5) — these are content fingerprints,
    not security digests, but standardizing on SHA-256 avoids weak-algorithm
    findings and keeps one code path. ``algorithm`` is accepted for backward
    compatibility; only "sha256" is supported.
    """
    if algorithm.lower() != "sha256":
        raise ValueError(f"Unsupported algorithm: '{algorithm}'")
    hash_obj = hashlib.sha256(data.encode("utf-8"))
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
