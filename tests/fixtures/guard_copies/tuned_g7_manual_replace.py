# guard: G7
# as: application/ingestion/resume.py
# expect: copy
import json
import os
from pathlib import Path
from typing import Any


def _mut(state_file: Path, state: dict[str, Any]) -> None:
    tmp = state_file.with_suffix(state_file.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, state_file)
