# guard: G7
# as: evaluation/report.py
# expect: copy
def _mut(p, t):
    with p.open("r+") as f:
        f.seek(0)
        f.write(t)
        f.truncate()
