# guard: G5
# as: adapters/ingestion/loader.py
# expect: clean
from pathlib import Path


def _mut(root: Path) -> list[tuple[str, str]]:
    """List the corpus files on disk: their names and paths under ``root``."""
    found = []
    for path in sorted(root.rglob("*.txt")):
        found.append((path.name, path.relative_to(root).as_posix()))
    return found
