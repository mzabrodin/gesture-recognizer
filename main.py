from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    """
    Thin wrapper that runs the real application from src/main.py.

    Allows you to keep using:
        python main.py
    from the project root.
    """

    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "2")

    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    app_entry = src_dir / "main.py"
    if not app_entry.exists():
        raise SystemExit(f"Entry script not found: {app_entry}")

    runpy.run_path(str(app_entry), run_name="__main__")


if __name__ == "__main__":
    main()
