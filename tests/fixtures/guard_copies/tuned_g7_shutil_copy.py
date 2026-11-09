# guard: G7
# as: application/cli/console/store.py
# expect: copy
import shutil
from pathlib import Path


def _mut(uploaded: Path, store_dir: Path) -> Path:
    dest = store_dir / uploaded.name
    shutil.copyfile(uploaded, dest)
    return dest
