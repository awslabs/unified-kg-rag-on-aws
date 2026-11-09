# guard: G1
# as: evaluation/report.py
# expect: copy
def _mut(m):
    match m:
        case {"degraded": True}:
            return 1
    return 0
