# guard: G7
# as: adapters/ingestion/document_converter.py
# expect: copy
import io
from pathlib import Path

_MODE = "wb"


def _mut(pdf_bytes: bytes, dest: Path) -> None:
    handle = io.open(dest, _MODE)
    try:
        handle.write(pdf_bytes)
    finally:
        handle.close()
