# guard: G2
# as: adapters/llm/langchain.py
# expect: copy
import json
from typing import Any

_DECODER = json.JSONDecoder()


def _mut(generation: str) -> Any:
    idx = generation.index("{")
    obj, _end = _DECODER.raw_decode(generation, idx)
    return obj
