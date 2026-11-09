# guard: G1
# as: application/retrieval/citations.py
# expect: copy
from typing import Any


def _mut(output_metadata: dict[str, Any], exc: BaseException) -> None:
    bucket = output_metadata.setdefault("citations_unresolved", [])
    bucket.append(
        {
            "step": "citation_lineage",
            "reason": "lineage read failed",
            "error_type": exc.__class__.__name__,
        }
    )
