# guard: G2
# as: evaluation/pairwise.py
# expect: copy
from json import loads as _loads
from typing import Any


def _mut(judge_output: str) -> dict[str, Any]:
    start = judge_output.find("{")
    end = judge_output.rfind("}") + 1
    payload = judge_output[start:end] if start != -1 and end > start else "{}"
    verdict = _loads(payload)
    return {"winner": verdict.get("winner"), "reason": verdict.get("reason", "")}
