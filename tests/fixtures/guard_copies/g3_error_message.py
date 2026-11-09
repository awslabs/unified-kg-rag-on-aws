# guard: G3
# as: application/cli/run_rag_chain.py
# expect: copy
def _mut(output):
    return output.metadata.get("error_message")
