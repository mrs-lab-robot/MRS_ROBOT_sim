#!/bin/bash
# Arena PPO 检查点评估快速启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA_ROOT="$SCRIPT_DIR"

echo "=========================================="
echo "Arena PPO 检查点评估器"
echo "=========================================="
echo ""

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <checkpoint_path> [options]"
    echo ""
    echo "示例:"
    echo "  $0 /path/to/run/checkpoints/model_5000.pt --num-episodes 10"
    echo "  $0 /path/to/run/checkpoints/model_5000.pt --num-episodes 50 --seed-start 2000"
    echo ""
    echo "选项:"
    echo "  --num-episodes N    评估的 episode 数量 (默认: 10)"
    echo "  --seed-start N      起始随机种子 (默认: 1000)"
    echo "  --headless          无头模式运行 (推荐)"
    echo "  --device DEVICE     计算设备，如 cuda:0"
    echo ""
    exit 1
fi

CHECKPOINT="$1"
shift

# 验证检查点文件存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: 检查点文件不存在: $CHECKPOINT"
    exit 1
fi

echo "检查点: $CHECKPOINT"
echo "Arena 根目录: $ARENA_ROOT"
echo ""

# 运行评估
cd "$ARENA_ROOT"
exec ./scripts/evaluate_arena_navigation_ppo.py \
    --checkpoint "$CHECKPOINT" \
    "$@"
