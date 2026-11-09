# guard: G1
# as: evaluation/report.py
# expect: copy
from ..domain.retrieval import degradation as _deg


def _mut(m):
    return m.get(_deg.DEGRADED_ERROR_KEY)
