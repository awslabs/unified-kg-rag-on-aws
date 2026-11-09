# guard: G3
# as: application/cli/mcp_tools.py
# expect: clean
from typing import Any

from unified_kg_rag.application.presentation.answer_view import QueryOutcome


def _mut(output: Any, elapsed: float) -> dict[str, Any]:
    outcome = QueryOutcome.of_output(output)
    return {"elapsed_seconds": round(elapsed, 3), **outcome.outcome_fields()}
