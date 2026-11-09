# guard: G7
# as: application/cli/run_visualization.py
# expect: copy
import json
from pathlib import Path
from typing import Any


def _mut(out_dir: Path, data: dict[str, Any]) -> Path:
    target = out_dir / "graph.json"
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return target
