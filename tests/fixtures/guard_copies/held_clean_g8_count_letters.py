# guard: G8
# as: application/ingestion/source_parsing.py
# expect: flagged: count_letters is the detector's internal; a caller outside domain.languages is reported
from collections.abc import Iterable

from unified_kg_rag.shared.utils.scripts import count_letters


def _mut(texts: Iterable[str]) -> str | None:
    return count_letters(texts, max_texts=50, max_chars=4000).dominant()
