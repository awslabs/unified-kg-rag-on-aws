# guard: G11
# as: application/retrieval/citations.py
# expect: missed: the lineages joined by set().union over a generator bound to a name
from typing import Any


def _mut(sources: list[Any]) -> list[str]:
    chunk_sets = (set(s.metadata.get("chunk_ids", [])) for s in sources)
    return sorted(set().union(*chunk_sets))
