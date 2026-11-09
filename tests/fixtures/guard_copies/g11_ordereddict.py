# guard: G11
# as: evaluation/report.py
# expect: copy
from collections import OrderedDict


def _mut(units):
    return list(OrderedDict.fromkeys(units))
