# guard: G3
# as: application/cli/run_rag_chain.py
# expect: copy
from typing import Any


def _mut(output: Any) -> None:
    print(output.answer)
    for rank, source in enumerate(output.sources, start=1):
        meta = source.get("metadata") or {}
        print(f"[{rank}] {meta.get('type', 'text')} score={source.get('score')}")
