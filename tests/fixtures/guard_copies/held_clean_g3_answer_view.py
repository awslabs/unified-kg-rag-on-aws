# guard: G3
# as: application/cli/run_rag_chain.py
# expect: clean
from typing import Any

from unified_kg_rag.application.presentation.answer_view import AnswerView


def _mut(output: Any) -> None:
    view = AnswerView.from_output(output)
    print(view.answer)
    for source in view.sources:
        print(f"[{source.rank}] {source.kind}")
    if view.partial:
        print("answered without: " + ", ".join(d.step for d in view.degraded))
