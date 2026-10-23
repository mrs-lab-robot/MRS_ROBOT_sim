"""pytest 引导：把仓库里的源码目录挂进 sys.path。

之前 `pytest tests/unit` 会直接 collection error（`No module named
openflex_isaac_contract`），因为仓库既没有 conftest、也没把 arena_benchmark/src、
isaaclab_ext/src、sim_runtime 下的包装进 venv —— 每个脚本各自
`_configure_project_paths()` 往 sys.path 里塞（全仓 50+ 处），测试却没人管。

这里统一一次，测试和脚本走同一套路径约定；只在已存在时插入，不覆盖环境。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOTS = (
    _REPO_ROOT / "arena_benchmark" / "src",
    _REPO_ROOT / "isaaclab_ext" / "src",
    _REPO_ROOT / "sim_runtime" / "teleoperation" / "src",
    _REPO_ROOT / "sim_runtime" / "ros2" / "openflex_isaac_sim" / "openflex_isaac_contract",
)

os.environ.setdefault("MRS_ROBOT_SIM_ROOT", str(_REPO_ROOT))

for _root in _SOURCE_ROOTS:
    _entry = str(_root)
    if _root.exists() and _entry not in sys.path:
        sys.path.insert(0, _entry)
