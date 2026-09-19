#!/usr/bin/env bash
# train_safe.sh - 跑 UniLab 训练，但先确认 CUDA 可见，并在训练期间禁止系统休眠。
#
# 为什么需要它
# ------------
# 2026-09-18 的事故：
#   01:07:28  系统进 s2idle 休眠，三条并行训练被杀
#   01:20:13  唤醒
#   01:20:16  NVIDIA 报 Xid 31 MMU Fault
#   01:24:25  新训练启动 -> torch.cuda.is_available() 返回 False
#             -> src/unilab/utils/device.py:18 get_default_device() 静默返回 "cpu"
#             -> 整场 3000 轮跑在 CPU 上，55 分钟而不是 24 分钟，零报错
#
# 根因不是 CUDA，是 training.device 没写死时用运行时探测，探测失败就静默降级。
# resolve_torch_device_alias() 里那个"绝不静默降级"的校验函数，在训练路径上零调用。
#
# 这个脚本做两件事：
#   1. 启动前校验 CUDA；不可见就直接退出，不跑
#   2. 用 systemd-inhibit 挡住训练期间的休眠
#
# 用法：
#   ./train_safe.sh --algo ppo --task dm10_joystick_flat --sim mujoco algo.max_iterations=3000
#
# 环境变量：
#   UNILAB_ROOT  UniLab 检出目录，默认 $HOME/UniLab

set -euo pipefail

UNILAB_ROOT="${UNILAB_ROOT:-$HOME/UniLab}"
if [[ ! -d "$UNILAB_ROOT" ]]; then
    echo "找不到 UniLab 目录：$UNILAB_ROOT（用 UNILAB_ROOT=... 指定）" >&2
    exit 1
fi
cd "$UNILAB_ROOT"

if ! uv run python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    cat >&2 <<'MSG'
CUDA 不可见。训练会静默降级到 CPU（实测慢 2.3 倍），已中止。

  常见原因：刚从系统休眠唤醒，NVIDIA 驱动没恢复（dmesg 里会有 Xid 31）。
  处理：
    1) 先看 nvidia-smi 能否列出显卡
    2) 重载驱动：sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia
    3) 最稳妥：重启

  参考：全历史 60+ 条 run 里唯一一条跑在 CPU 上的，就是休眠唤醒后 4 分钟启动的那条。
MSG
    exit 1
fi

echo "CUDA 可见，训练期间禁止休眠"
exec systemd-inhibit \
    --what=idle:sleep \
    --why="UniLab training in progress (train_safe.sh)" \
    --mode=block \
    uv run train "$@"
