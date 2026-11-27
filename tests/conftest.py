"""Pytest 路径引导，确保根目录、父目录与 tests 目录均在模块搜索路径中。"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
_PROJECT_PARENT = _PROJECT_ROOT.parent

for _path in (_PROJECT_PARENT, _PROJECT_ROOT, _TESTS_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
