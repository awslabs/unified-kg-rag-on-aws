# guard: G6
# as: evaluation/report.py
# expect: copy
def _mut(v):
    return [v] if type(v) is str else list(v)
