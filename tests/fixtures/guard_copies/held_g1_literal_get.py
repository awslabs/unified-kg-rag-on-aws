# guard: G1
# as: application/cli/display.py
# expect: copy
from typing import Any


def _mut(metadata: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for entry in metadata.get("degraded") or []:
        lines.append(f"skipped {entry.get('step')}: {entry.get('reason')}")
    if metadata.get("degraded_error"):
        lines.append("first error: %s" % metadata["degraded_error"])
    return lines
