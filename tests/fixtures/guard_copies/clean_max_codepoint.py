# guard: G4
# as: domain/models/seed_graph.py
# expect: clean
def _valid(ch):
    return ord(ch) <= 0x10FFFF
