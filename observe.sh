#!/usr/bin/env bash
# observe.sh - 跑一轮训练，同时采硬件数据，输出一份画像。
#
# 输出到 $REPORTS_ROOT/<时间戳>_<tag>/：
#   summary.md    分相预算 + 每档核利用率 + 温度 + GPU
#   phases.json   机读，用于历次 diff
#   percore.txt   每核明细（P / E / LP-E 分档）、每线程 schedstat、温度、降频增量
#   gpu.csv       GPU 利用率/显存/功耗/温度/降频原因采样
#   train.log     原始训练日志
#
# 用法：
#   ./observe.sh --tag baseline
#   ./observe.sh --tag compile-on --train-args "algo.algorithm.enable_compile=true"
#   ./observe.sh --diff baseline compile-on
#
# 环境变量：
#   UNILAB_ROOT   UniLab 检出目录，默认 $HOME/UniLab
#   REPORTS_ROOT  输出根目录，默认 $HOME/reports/dm10-observe
#
# 注意：
#   - 训练走 train_safe.sh，所以会先校验 CUDA 并禁止休眠
#   - 采样窗口默认从训练启动后 20s 开始（跳过 import 和 chunk_tuner 冷启动）
#   - 采样器只读 /proc，对训练几乎无扰动，但仍会量一次插桩开销（见 S3 说明）

set -euo pipefail

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
UNILAB_ROOT="${UNILAB_ROOT:-$HOME/UniLab}"
REPORTS_ROOT="${REPORTS_ROOT:-$HOME/reports/dm10-observe}"

ITERS=250
TAG="run"
TRAIN_ARGS="--algo ppo --task dm10_joystick_flat --sim mujoco"
SAMPLE_INTERVAL=0.1
SAMPLE_DURATION=45
WARMUP=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iters)      ITERS="$2"; shift 2 ;;
        --tag)        TAG="$2"; shift 2 ;;
        --train-args) TRAIN_ARGS="$2"; shift 2 ;;
        --interval)   SAMPLE_INTERVAL="$2"; shift 2 ;;
        --duration)   SAMPLE_DURATION="$2"; shift 2 ;;
        --warmup)     WARMUP="$2"; shift 2 ;;
        --diff)       DIFF_A="$2"; DIFF_B="$3"; shift 3 ;;
        -h|--help)    sed -n '2,25p' "$0"; exit 0 ;;
        *)            echo "未知参数: $1" >&2; exit 2 ;;
    esac
done

# ---- diff 模式：比两次观测的 phases.json ----
if [[ -n "${DIFF_A:-}" ]]; then
    A=$(ls -d "$REPORTS_ROOT"/*_"$DIFF_A" 2>/dev/null | tail -1)
    B=$(ls -d "$REPORTS_ROOT"/*_"$DIFF_B" 2>/dev/null | tail -1)
    [[ -n "$A" && -n "$B" ]] || { echo "找不到 tag：$DIFF_A 或 $DIFF_B" >&2; exit 1; }
    echo "$(basename "$A")  vs  $(basename "$B")"
    python3 - "$A/phases.json" "$B/phases.json" <<'PYEOF'
import json, sys

def flat(d):
    out = {}
    for k, v in (d.get("scalars") or {}).items():
        if isinstance(v, dict) and "mean" in v:
            out[k] = v["mean"]
    for k, v in (d.get("summary") or {}).items():
        if isinstance(v, (int, float)):
            out[f"summary/{k}"] = float(v)
    return out

fa, fb = flat(json.load(open(sys.argv[1]))), flat(json.load(open(sys.argv[2])))
print(f"{'metric':<40s} {'A':>12s} {'B':>12s} {'delta':>10s}")
print("-" * 78)
for k in sorted(set(fa) & set(fb)):
    x, y = fa[k], fb[k]
    pct = (y / x - 1) * 100 if x else float("nan")
    mark = "  <<<" if abs(pct) > 1.2 else ""
    print(f"{k:<40s} {x:>12.4f} {y:>12.4f} {pct:>9.1f}%{mark}")
print()
print("噪声底约 1.2%（同配置重跑实测）。小于它的差异不下结论。")
PYEOF
    exit 0
fi

# ---- 观测模式 ----
TS=$(date +%Y-%m-%d_%H%M%S)
DIR="$REPORTS_ROOT/${TS}_${TAG}"
mkdir -p "$DIR"
echo "输出目录: $DIR"

cd "$UNILAB_ROOT"

"$SELF_DIR/train_safe.sh" $TRAIN_ARGS algo.max_iterations="$ITERS" training.no_play=true \
    > "$DIR/train.log" 2>&1 &
TRAIN_BG=$!

sleep "$WARMUP"
if ! kill -0 "$TRAIN_BG" 2>/dev/null; then
    echo "训练在 ${WARMUP}s 内退出，看 $DIR/train.log" >&2
    tail -20 "$DIR/train.log" >&2
    exit 1
fi

TPID=$(ps -eo pid,args 2>/dev/null | grep -F 'train_rsl_rl.py' | grep -v grep | awk '{print $1}' | head -1)
echo "训练进程 PID = ${TPID:-未找到}"

( while kill -0 "$TRAIN_BG" 2>/dev/null; do
      nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu,temperature.memory,clocks_throttle_reasons.active \
          --format=csv,noheader,nounits 2>/dev/null
      sleep 0.5
  done > "$DIR/gpu.csv" ) &
GPU_BG=$!

UNILAB_ROOT="$UNILAB_ROOT" uv run --project "$UNILAB_ROOT" python "$SELF_DIR/scripts/percore.py" \
    --duration "$SAMPLE_DURATION" --interval "$SAMPLE_INTERVAL" --timeseries \
    ${TPID:+--pid "$TPID"} --out "$DIR/percore.txt" 2>&1 | tail -1

kill "$GPU_BG" 2>/dev/null || true
wait "$TRAIN_BG" 2>/dev/null; TRAIN_RC=$?
echo "训练退出码=$TRAIN_RC"

NEW_RUN=$(ls -dt logs/rsl_rl_ppo/*/*/ 2>/dev/null | head -1)
uv run --project "$UNILAB_ROOT" python "$SELF_DIR/scripts/collect_perf.py" "$NEW_RUN" \
    --out "$DIR/phases.json" 2>&1 | tail -1

REPORTS_ROOT="$REPORTS_ROOT" python3 "$SELF_DIR/scripts/render_summary.py" "$DIR" "$TAG" "$TRAIN_RC"

echo
echo "完成 -> $DIR"
