# guard: G7
# as: evaluation/report.py
# expect: clean
import json
from pathlib import Path
from typing import Any

from unified_kg_rag.shared.utils.files import atomic_open, atomic_write_text


def _mut(out_dir: Path, summary: str, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(out_dir / "report.md", summary)
    with atomic_open(out_dir / "rows.jsonl") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
