# guard: G9
# as: evaluation/report.py
# expect: copy
def _mut(op, a, b):
    if op in ("gt", "gte"):
        return a > b or (op == "gte" and a == b)
    return a < b
