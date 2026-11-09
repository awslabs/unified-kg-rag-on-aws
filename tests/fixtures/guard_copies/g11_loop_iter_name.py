# guard: G11
# as: evaluation/report.py
# expect: copy
def _mut(rs):
    out = []
    for r in rs:
        for cid in r.metadata.get("source_chunks", []):
            if cid not in out:
                out.append(cid)
    return out
