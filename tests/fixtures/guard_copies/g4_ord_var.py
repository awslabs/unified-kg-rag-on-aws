# guard: G4
# as: evaluation/text_matching.py
# expect: copy
def _mut(ch):
    cp = ord(ch)
    return 0xAC00 <= cp <= 0xD7A3
