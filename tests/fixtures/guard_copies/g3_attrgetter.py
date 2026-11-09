# guard: G3
# as: application/cli/mcp_tools.py
# expect: copy
import operator


def _mut(output):
    return operator.attrgetter("sources")(output)
