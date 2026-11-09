# guard: G7
# as: adapters/ingestion/file_loaders.py
# expect: clean
from pathlib import Path


def _mut(path: Path) -> tuple[str, bytes]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    with path.open("rb") as raw:
        head = raw.read(16)
    return text + path.read_text(encoding="utf-8")[:0], head
