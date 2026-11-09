# guard: G6
# as: evaluation/retrieval_evaluator.py
# expect: missed: the type test is one side of an 'and', and the wrap falls through after it
from collections.abc import Iterable
from typing import Any


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _mut(result_metadata: dict[str, Any]) -> set[str]:
    return set(_as_list(result_metadata.get("community_ids")))
