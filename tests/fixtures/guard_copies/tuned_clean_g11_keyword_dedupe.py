# guard: G11
# as: adapters/search_strategies/lightrag_search.py
# expect: clean


def _mut(high: list[str], low: list[str]) -> list[str]:
    """The query's keywords, each once (no lineage involved)."""
    keywords = [k.strip() for k in high + low if k.strip()]
    return list(dict.fromkeys(k.lower() for k in keywords))
