# guard: G5
# as: application/cli/run_describe.py
# expect: flagged: getattr on a CLI namespace reads like a stored field; allowlist it
def _f(args):
    return getattr(args, "file_path", None)
