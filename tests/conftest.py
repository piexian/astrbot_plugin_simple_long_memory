"""Pytest 环境引导：隔离 AstrBot 数据目录并确保测试模块搜索路径可用。"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

if not os.environ.get("ASTRBOT_ROOT"):
    _tmp_astrbot_root = tempfile.mkdtemp(prefix="astrbot-test-")
    os.environ["ASTRBOT_ROOT"] = _tmp_astrbot_root
    atexit.register(shutil.rmtree, _tmp_astrbot_root, True)

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
_PROJECT_PARENT = _PROJECT_ROOT.parent

for _path in (_PROJECT_PARENT, _PROJECT_ROOT, _TESTS_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
