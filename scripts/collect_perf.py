#!/usr/bin/env python3
"""从一次训练 run 的产物里提取可对比的性能画像 → JSON。

读什么
------
- `<run>/run_config.json`  → device / num_envs / max_iterations
- `<run>/run_summary.json` → 吞吐 / 墙钟
- `<run>/events.out.tfevents*` → 全部 scalar（`Perf/*` 为主，也收 `timing/*`）

为什么单独成文件
----------------
1. 读取要用 `.venv` 的 python（tensorboard 在里面），不能靠系统 python
2. `Perf/collection_time` 等的**口径**必须在一个地方写清楚，避免各工具各读一套
   （本会话已三次因读数口径不一得出假结论）
3. 输出固定 schema ⇒ 历次结果可直接 diff

用法
----
    uv run python scripts/observe/collect_perf.py <run_dir> [--out phases.json]
    uv run python scripts/observe/collect_perf.py --latest [--task DM10JoystickFlat]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("UNILAB_ROOT", Path.home() / "UniLab")).resolve()
# ^ UniLab 检出目录。脚本用 uv run --project $UNILAB_ROOT 跑，所以 import 得到 unilab/uni_rl。

# 这些 prefix 的 scalar 收进 JSON；其余忽略（reward 明细等噪音太大）
KEEP_PREFIXES = ("Perf/", "timing/", "Loss/", "Train/mean_reward", "Episode_Termination/")


def latest_run(task: str = "DM10JoystickFlat", root: str = "logs/rsl_rl_ppo") -> str | None:
    base = REPO / root / task
    if not base.is_dir():
        return None
    cands = [p for p in base.iterdir() if p.is_dir()]
    if not cands:
        return None
    return str(max(cands, key=lambda p: p.stat().st_mtime))


def read_scalars(run_dir: str) -> dict[str, dict]:
    """{tag: {n, mean, min, max, last}}。用 tensorboard 的 EventAccumulator。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = glob.glob(os.path.join(run_dir, "events.out.tfevents*"))
    if not files:
        return {}
    acc = EventAccumulator(sorted(files)[-1])
    acc.Reload()
    out: dict[str, dict] = {}
    for tag in acc.Tags().get("scalars", []):
        if not any(tag.startswith(p) for p in KEEP_PREFIXES):
            continue
        vals = [e.value for e in acc.Scalars(tag)]
        if not vals:
            continue
        out[tag] = {
            "n": len(vals),
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
            "last": vals[-1],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--latest", action="store_true", help="用最新的 run（而不是给路径）")
    ap.add_argument("--task", default="DM10JoystickFlat")
    ap.add_argument("--out", default=None, help="输出 JSON 路径（默认 print）")
    args = ap.parse_args(argv)

    run_dir = args.run_dir
    if args.latest or run_dir is None:
        run_dir = latest_run(args.task)
        if run_dir is None:
            print("找不到 run", file=sys.stderr)
            return 1
    run_dir = str(Path(run_dir).resolve())

    result: dict = {
        "run_dir": run_dir,
        "run_name": Path(run_dir).name,
        "config": {},
        "summary": {},
        "scalars": read_scalars(run_dir),
    }

    cfg_path = os.path.join(run_dir, "run_config.json")
    if os.path.isfile(cfg_path):
        raw = json.load(open(cfg_path))
        c = raw.get("config", raw)
        result["config"] = {
            "device": (raw.get("run") or {}).get("device"),
            "num_envs": (c.get("algo") or {}).get("num_envs"),
            "max_iterations": (c.get("algo") or {}).get("max_iterations"),
            "num_steps_per_env": (c.get("algo") or {}).get("num_steps_per_env"),
            "sim_backend": (c.get("training") or {}).get("sim_backend"),
        }

    sum_path = os.path.join(run_dir, "run_summary.json")
    if os.path.isfile(sum_path):
        s = json.load(open(sum_path))
        result["summary"] = {
            k: s.get(k)
            for k in (
                "status", "completed_iterations", "total_env_steps",
                "training_throughput_env_steps_per_sec", "training_wall_time_sec",
                "final_mean_reward", "last_checkpoint",
            )
            if k in s
        }

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"已写入 {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
