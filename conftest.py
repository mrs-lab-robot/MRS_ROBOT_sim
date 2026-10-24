"""pytest 引导：统一交给 tools/bootstrap.py（见该文件说明）。"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[0] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import bootstrap  # noqa: E402  （导入即完成 sys.path / 环境变量设置）

collect_ignore = bootstrap.collect_ignore()
