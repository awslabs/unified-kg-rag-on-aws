# guard: G6
# as: evaluation/report.py
# expect: copy
def _mut(v):
    return list(v) if isinstance(v, list | tuple) else [v]
