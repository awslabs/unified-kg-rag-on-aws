# guard: G11
# as: evaluation/report.py
# expect: copy
def _mut(rs):
    return sorted({u for r in rs for u in r.text_unit_ids})
