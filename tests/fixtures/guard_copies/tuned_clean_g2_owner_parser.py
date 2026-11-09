# guard: G2
# as: adapters/search_strategies/drift_search.py
# expect: clean
from typing import Any

from pydantic import BaseModel

from ...shared.utils.model_output import parse_model_object


class _Primer(BaseModel):
    intermediate_answer: str = ""
    follow_up_queries: list[str] = []


def _mut(response: Any) -> _Primer:
    return parse_model_object(response.content, _Primer)
