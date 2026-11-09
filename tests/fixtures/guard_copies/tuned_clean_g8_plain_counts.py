# guard: G8
# as: adapters/ingestion/chunker.py
# expect: clean


def _mut(text: str) -> dict[str, int]:
    """Size statistics of a chunk, not by script."""
    return {
        "chars": len(text),
        "spaces": sum(1 for ch in text if ch.isspace()),
        "digits": sum(ch.isdigit() for ch in text),
        "words": len(text.split()),
    }
