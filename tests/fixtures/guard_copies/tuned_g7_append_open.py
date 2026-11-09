# guard: G7
# as: application/ingestion/pipeline_metrics.py
# expect: copy
import json
from pathlib import Path
from typing import Any


def _mut(log_path: Path, record: dict[str, Any]) -> None:
    with log_path.open(mode="a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
