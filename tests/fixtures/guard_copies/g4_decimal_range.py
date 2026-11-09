# guard: G4
# as: domain/ingestion/chunk_length.py
# expect: copy
def _mut(ch):
    return ord(ch) in range(44032, 55204)
