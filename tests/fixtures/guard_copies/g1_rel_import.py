# guard: G1
# as: domain/retrieval/citations.py
# expect: copy
from .degradation import DEGRADED_KEY as _DK


def _mut(m):
    return m.get(_DK)
