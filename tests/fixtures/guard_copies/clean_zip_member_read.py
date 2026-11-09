# guard: G7
# as: adapters/ingestion/document_converter.py
# expect: clean
import zipfile


def _f(p):
    with zipfile.ZipFile(p) as z:
        return z.open("word/document.xml").read()
