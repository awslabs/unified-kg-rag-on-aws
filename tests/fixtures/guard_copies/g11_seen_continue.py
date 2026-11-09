# guard: G11
# as: evaluation/report.py
# expect: copy
def _mut(rs):
    seen, out = set(), []
    for r in rs:
        for u in r.text_unit_ids:
            if u in seen:
                continue
            seen.add(u)
            out.append(u)
    return out
