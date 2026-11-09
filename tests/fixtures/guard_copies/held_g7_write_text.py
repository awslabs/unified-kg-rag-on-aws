# guard: G7
# as: evaluation/report.py
# expect: copy
from pathlib import Path


def _mut(markdown: str, output: str) -> None:
    Path(output).write_text(markdown, encoding="utf-8")
