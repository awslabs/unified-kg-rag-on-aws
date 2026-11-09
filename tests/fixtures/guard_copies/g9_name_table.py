# guard: G9
# as: evaluation/report.py
# expect: copy
import operator

_NAMES = {"gt": "gt", "gte": "ge", "lt": "lt", "lte": "le"}


def _mut(op, a, b):
    return getattr(operator, _NAMES[op])(a, b)
