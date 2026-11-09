# guard: G3
# as: application/cli/mcp_tools.py
# expect: copy
def _mut(output):
    meta = output.metadata
    return meta.get("error")
