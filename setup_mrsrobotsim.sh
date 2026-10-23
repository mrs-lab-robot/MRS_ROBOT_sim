#!/usr/bin/env bash
# setup_mrsrobotsim.sh — MRS_ROBOT_sim 仿真仓库一键配置 + 自检
#
# 用法:
#   bash setup_mrsrobotsim.sh                 # 克隆(或更新) + 建 Python 环境 + 自检
#   bash setup_mrsrobotsim.sh --verify-only   # 只做环境自检（不克隆/不建环境）
#   bash setup_mrsrobotsim.sh --skip-sim      # 不跑模拟 VR E2E
#   bash setup_mrsrobotsim.sh --with-vr       # 配置完成后跑真实 VR E2E（需 Pico 头显）
#
# 前置:
#   - 已跑 setup_mrsrobot.sh（主仓库），或已设 MRS_ROBOT 变量
#   - 磁盘充足（仿真环境含 Isaac Sim 依赖约 20-30G）
#   - 能访问 github.com（含子模块，可能需要 SSH/HTTPS）
#
# 环境变量可覆盖:
#   MRS_ROBOT_SIM_ROOT   仿真仓库根（默认 ${PWD}，即脚本所在目录的父/自身）
#   MRS_ROBOT_ROOT       主仓库根（默认 ${PWD}/../MRS_ROBOT）
#   UV_CACHE_DIR         uv 缓存目录（默认 ${PWD}/.cache/uv）
#   UV_PYTHON_INSTALL_DIR uv Python 安装目录（默认 ${PWD}/.uv-python）

set -eo pipefail

GREEN=$'\e[32m'; YELLOW=$'\e[33m'; RED=$'\e[31m'; NC=$'\e[0m'
ok()   { echo "  ${GREEN}[OK]${NC} $*"; }
warn() { echo "  ${YELLOW}[WARN]${NC} $*"; }
fail() { echo "  ${RED}[FAIL]${NC} $*"; }

VERIFY_ONLY=0; SKIP_SIM=0; WITH_VR=0
for arg in "$@"; do
  case "$arg" in
    --verify-only) VERIFY_ONLY=1 ;;
    --skip-sim)    SKIP_SIM=1 ;;
    --with-vr)     WITH_VR=1 ;;
    *) echo "未知参数: $arg" >&2; exit 2 ;;
  esac
done

REPO_URL="git@github.com:mrs-lab-robot/MRS_ROBOT_sim.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="${MRS_ROBOT_SIM_ROOT:-$SCRIPT_DIR}"
MRS_ROBOT_ROOT="${MRS_ROBOT_ROOT:-$(dirname "$SIM_ROOT")/MRS_ROBOT}"
ARENA_ROOT="$SIM_ROOT/arena_benchmark"
VENV="$ARENA_ROOT/.venv"
UV_CACHE_DIR="${UV_CACHE_DIR:-$SIM_ROOT/.cache/uv}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SIM_ROOT/.uv-python}"

echo "== MRS_ROBOT_sim 仿真仓库配置 =="
echo "  仿真仓库: $SIM_ROOT"
echo "  Arena:    $ARENA_ROOT"

# ---- 前置 ----
echo; echo "== [L0] 前置检查 =="
command -v git >/dev/null || { fail "缺少 git"; exit 1; }
ok "git 存在"
command -v uv >/dev/null 2>&1 || command -v "$HOME/.local/bin/uv" >/dev/null 2>&1 || {
  fail "缺少 uv。安装: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
}
ok "uv 存在"

# ---- 克隆/更新 ----
echo; echo "== 克隆 / 更新 =="
if [ -d "$SIM_ROOT/.git" ]; then
  echo "  已存在 $SIM_ROOT，git pull + 子模块更新..."
  (cd "$SIM_ROOT" && git pull --ff-only origin main)
  (cd "$SIM_ROOT" && git submodule update --init --depth 1 third_party/IsaacLab-Arena)
  (cd "$SIM_ROOT/third_party/IsaacLab-Arena" && git submodule update --init --depth 1 submodules/IsaacLab)
  ok "仿真仓库已更新"
else
  echo "  克隆 $REPO_URL ..."
  git clone "$REPO_URL" "$SIM_ROOT"
  (cd "$SIM_ROOT" && git submodule update --init --depth 1 third_party/IsaacLab-Arena)
  (cd "$SIM_ROOT/third_party/IsaacLab-Arena" && git submodule update --init --depth 1 submodules/IsaacLab)
  ok "仿真仓库已克隆"
fi

[ -d "$ARENA_ROOT" ] || { fail "克隆后未找到 $ARENA_ROOT"; exit 1; }

if [ "$VERIFY_ONLY" = "1" ]; then
  echo; echo "== 仅验证 =="
  if [ -x "$VENV/bin/python" ]; then
    "$VENV/bin/python" - <<'PY'
import isaacsim, mrs_teleoperation, mrs_robot_lab, mrs_arena
print("  [OK] isaacsim:", isaacsim.__file__)
print("  [OK] mrs_teleoperation:", mrs_teleoperation.__file__)
print("  [OK] mrs_robot_lab:", mrs_robot_lab.__file__)
print("  [OK] mrs_arena:", mrs_arena.__file__)
PY
  else
    fail "未找到 $VENV/bin/python（先跑一次完整配置）"
    exit 1
  fi
  exit 0
fi

# ---- Python 环境 ----
echo; echo "== Python 环境 (.venv) =="
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR
if [ -x "$VENV/bin/python" ]; then
  echo "  复用已有 $VENV"
  # 用共享/本机 Python 3.12 指回，避免指向已删除的用户 home
  PY312=$(command -v python3.12 || echo "")
  if [ -n "$PY312" ]; then
    ln -sfn "$PY312" "$VENV/bin/python" 2>/dev/null || true
  fi
  ok "venv 已存在"
else
  echo "  创建 venv（uv sync，下载 Isaac Sim 依赖，可能需 20-40 分钟）..."
  uv python install 3.12
  PY312="$UV_PYTHON_INSTALL_DIR"/cpython-3.12*/bin/python3.12
  mkdir -p "$VENV"
  UV_PROJECT_ENVIRONMENT="$VENV" uv sync \
    --project "$SIM_ROOT/third_party/IsaacLab-Arena" \
    --no-default-groups --group isaaclab-from-wheel
  uv pip install --python "$VENV/bin/python" --no-deps --editable "$SIM_ROOT/third_party/IsaacLab-Arena"
  uv pip install --python "$VENV/bin/python" --no-deps --editable "$SIM_ROOT/sim_runtime/teleoperation"
  uv pip install --python "$VENV/bin/python" --no-deps --editable "$SIM_ROOT/isaaclab_ext"
  uv pip install --python "$VENV/bin/python" --no-deps -e "$ARENA_ROOT"
  ok "venv 创建完成"
fi

# ---- 自检 ----
echo; echo "== [V] 仿真环境自检 =="
"$VENV/bin/python" - <<'PY'
import isaacsim, mrs_teleoperation, mrs_robot_lab, mrs_arena
print("  [OK] isaacsim import")
print("  [OK] mrs_teleoperation import")
print("  [OK] mrs_robot_lab import")
print("  [OK] mrs_arena import")
PY
ok "仿真 Python 环境导入正常"

# ---- E2E ----
if [ "$SKIP_SIM" != "1" ]; then
  echo; echo "== [E2E] 模拟 VR 采集验证 =="
  if [ -d "$MRS_ROBOT_ROOT" ] && [ -x "$MRS_ROBOT_ROOT/scripts/test/run_e2e.sh" ]; then
    echo "  将调用 $MRS_ROBOT_ROOT/scripts/test/run_e2e.sh sim"
    MRS_ROBOT_SIM_ROOT="$SIM_ROOT" MRS_ROBOT_ROOT="$MRS_ROBOT_ROOT" \
      bash "$MRS_ROBOT_ROOT/scripts/test/run_e2e.sh" sim \
      || warn "模拟 VR E2E 未通过（见上方输出）"
  else
    warn "主仓库 E2E 脚本不可用，跳过仿真 E2E"
  fi
fi

echo; echo "== 配置完成 =="
ok "MRS_ROBOT_sim 仿真仓库已就绪: $SIM_ROOT"
echo "  启动仿真（GUI 侧机器选 local，输入源选 模拟VR 或 真实VR）"

exit 0
