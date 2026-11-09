# guard: G6
# as: application/retrieval/rag_chain.py
# expect: clean
from collections.abc import Iterable


def _mut(namespaces: list[str], extra: Iterable[str]) -> list[str]:
    """The namespaces to query: the configured ones, then the requested."""
    out = list(namespaces)
    out.extend(extra)
    return out
