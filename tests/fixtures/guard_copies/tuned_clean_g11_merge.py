# guard: G11
# as: adapters/search_strategies/local_search.py
# expect: clean
from typing import Any

from ...domain.retrieval import lineage
from ...domain.retrieval.source_fields import listed


def _mut(entities: list[Any]) -> list[str]:
    return lineage.merge(
        [listed(e.metadata, "text_unit_ids") for e in entities],
        order="support",
        scores=[e.score for e in entities],
    )
