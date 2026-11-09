# guard: G4
# as: application/storage/inspection.py
# expect: clean
def _xml_legal(ch):
    return (
        ord(ch) in (0x9, 0xA, 0xD)
        or 0x20 <= ord(ch) <= 0xD7FF
        or 0xE000 <= ord(ch) <= 0xFFFD
    )
