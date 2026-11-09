# guard: G9
# as: evaluation/report.py
# expect: copy
def _mut(op, a, b):
    if op in ("gt", "gte"):
        return a > b if op[2:] else a >= b
    return a < b
