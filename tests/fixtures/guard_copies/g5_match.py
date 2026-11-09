# guard: G5
# as: evaluation/report.py
# expect: copy
def _mut(m):
    match m:
        case {"file_name": str(name)}:
            return name
    return None
