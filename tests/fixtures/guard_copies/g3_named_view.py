# guard: G3
# as: application/cli/run_rag_chain.py
# expect: missed: a name that says "view" is read as the answer view, whatever it holds
def _mut(search_output):
    raw_view = search_output
    return raw_view.sources
