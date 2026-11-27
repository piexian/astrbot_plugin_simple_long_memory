"""Pytest 环境引导：始终将 AstrBot 数据目录隔离到临时目录并设置搜索路径。"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

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
