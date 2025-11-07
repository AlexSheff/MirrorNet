import sys
from pathlib import Path


def pytest_configure():
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    paths = [str(root), str(src_path)]
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)