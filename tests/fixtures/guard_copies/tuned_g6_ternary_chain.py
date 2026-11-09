# guard: G6
# as: application/retrieval/graph_lookup.py
# expect: copy
from typing import Any


def _mut(props: dict[str, Any]) -> list[str]:
    raw = props.get("document_ids")
    return (
        []
        if raw is None
        else (
            [str(v) for v in raw] if isinstance(raw, (list, tuple, set)) else [str(raw)]
        )
    )
