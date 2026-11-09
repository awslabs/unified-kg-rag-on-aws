# guard: G11
# as: evaluation/report.py
# expect: copy
def _mut(rs):
    return list({u: None for r in rs for u in r.text_unit_ids})
