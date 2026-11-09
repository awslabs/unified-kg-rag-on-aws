# guard: G7
# as: evaluation/report.py
# expect: copy
def _mut(p, t, append=False):
    mode = "a" if append else "w"
    with open(p, mode) as f:
        f.write(t)
