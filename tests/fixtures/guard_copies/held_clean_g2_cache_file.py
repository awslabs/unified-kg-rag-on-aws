# guard: G2
# as: application/ingestion/cache_manager.py
# expect: flagged: json.loads cannot tell a cache file from a model answer; allowlist it
import json
from pathlib import Path
from typing import Any


def _mut(cache_dir: Path, key: str) -> dict[str, Any] | None:
    """Read a stage cache manifest this pipeline wrote itself (not a model
    answer)."""
    manifest = cache_dir / f"{key}.json"
    if not manifest.exists():
        return None
    return json.loads(manifest.read_text(encoding="utf-8"))
