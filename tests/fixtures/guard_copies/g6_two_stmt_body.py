# guard: G6
# as: evaluation/report.py
# expect: copy
def _mut(v):
    if isinstance(v, str):
        v = v.strip()
        v = [v]
    return v
