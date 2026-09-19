#!/usr/bin/env python3
"""把 `update_state` 的耗时拆到各个 manager —— 判断并行化值不值得做。

为什么需要它
------------
`update_state_ms ≈ 2.0 ms/步`（占一个控制步 13.4%），是**最大的单块串行**。
但它是六个 manager 的串联：
    termination → reward → metrics → event(DR) → command → observation → _map_observations
**只有知道哪一块是大头，才知道该不该动、动哪一块。**

⚠️ 不是所有 manager 都能并行：
   - `event_manager`（DR 随机化）和 `command_manager` **会写 sim state**
     （代码里紧跟 `scene._invalidate_state_reads()`），并行会引入依赖
   - `observation_manager.compute(update_history=True)` 有历史状态
   ⇒ 能做的是"测出每一块多大"，再判断有没有值得动的。

现成工具的局限
--------------
`scripts/benchmark/env/benchmark_env_overhead_parallel.py` 是**合成形状**
（8192 envs / obs_dim 98 / 29 奖励项），与 dm10（512 / 39 / 16）差很远，
它的绝对数字不能搬。本脚本在**真实 dm10 env** 上测。

用法
----
    uv run python scripts/observe/update_state_breakdown.py --num-envs 512 --steps 150
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = str(Path(os.environ.get("UNILAB_ROOT", Path.home() / "UniLab")).resolve())
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
# 模型路径在配置里是相对的（src/unilab/assets/...），所以必须切到检出目录
os.chdir(REPO_ROOT)


def build_env(num_envs: int, task: str, config_group: str):
    import hydra
    from omegaconf import OmegaConf

    from unilab.base.config_adapter import BackendAdapter, create_env
    from unilab.training import ensure_registries

    ensure_registries()
    with hydra.initialize_config_dir(
        version_base="1.3",
        config_dir=os.path.join(REPO_ROOT, "src", "unilab", "conf", config_group),
    ):
        cfg = hydra.compose(
            config_name="config",
            overrides=[f"task={task}", f"algo.num_envs={num_envs}"],
        )
    OmegaConf.resolve(cfg)
    override = BackendAdapter(
        cfg, root_dir=REPO_ROOT, algo_name=str(cfg.algo.algo)
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=num_envs, env_cfg_override=override)
    if env.state is None:
        env.init_state()
    return env


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--task", default="dm10_joystick_flat/mujoco")
    ap.add_argument("--config-group", default="ppo")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args(argv)

    env = build_env(args.num_envs, args.task, args.config_group)
    import numpy as np

    acc: dict[str, list[float]] = defaultdict(list)

    def wrap(obj, name: str, label: str):
        fn = getattr(obj, name)
        def wrapped(*a, **kw):
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            acc[label].append((time.perf_counter() - t0) * 1000.0)
            return out
        setattr(obj, name, wrapped)

    # 逐个 manager 打点
    wrap(env.termination_manager, "compute", "termination.compute")
    wrap(env.reward_manager, "compute", "reward.compute")
    if hasattr(env, "metrics_manager"):
        wrap(env.metrics_manager, "compute", "metrics.compute")
        if hasattr(env.metrics_manager, "compute_substep"):
            wrap(env.metrics_manager, "compute_substep", "metrics.compute_substep")
    if hasattr(env, "event_manager"):
        wrap(env.event_manager, "apply", "event.apply")
    if hasattr(env, "command_manager"):
        wrap(env.command_manager, "compute", "command.compute")
        if hasattr(env.command_manager, "post_compute"):
            wrap(env.command_manager, "post_compute", "command.post_compute")
    wrap(env.observation_manager, "compute", "observation.compute")
    wrap(env, "_map_observations", "_map_observations")

    # 整块 update_state 的总时长（含未打点的部分）
    wrap(env, "update_state", "UPDATE_STATE_TOTAL")

    rng = np.random.default_rng(0)
    adim = env.action_space.shape[-1]

    for _ in range(args.warmup):
        env.step(rng.uniform(-0.2, 0.2, size=(args.num_envs, adim)).astype(np.float32))

    for k in acc:
        acc[k].clear()
    acts = [rng.uniform(-0.2, 0.2, size=(args.num_envs, adim)).astype(np.float32)
            for _ in range(args.steps)]
    t0 = time.perf_counter()
    for a in acts:
        env.step(a)
    wall_total = (time.perf_counter() - t0) / args.steps * 1000.0

    total = sum(acc["UPDATE_STATE_TOTAL"]) / len(acc["UPDATE_STATE_TOTAL"])
    print(f"\nenv={args.task} num_envs={args.num_envs} steps={args.steps}")
    print(f"整步 wall = {wall_total:.3f} ms    update_state = {total:.3f} ms "
          f"({total/wall_total*100:.1f}% of step)\n")
    print(f"  {'子阶段':<26s} {'ms/次':>8s} {'占 update_state':>15s} {'占整步':>9s}")
    print("  " + "-" * 64)
    rows = []
    for k, v in acc.items():
        if k == "UPDATE_STATE_TOTAL" or not v:
            continue
        m = sum(v) / len(v)
        rows.append((m, k, len(v)))
    rows.sort(reverse=True)
    for m, k, n in rows:
        print(f"  {k:<26s} {m:>8.4f} {m/total*100:>14.1f}% {m/wall_total*100:>8.1f}%")
    accounted = sum(r[0] for r in rows)
    print("  " + "-" * 64)
    print(f"  {'已归因小计':<26s} {accounted:>8.4f} {accounted/total*100:>14.1f}%")
    print(f"  {'未归因（控制流/赋值等）':<26s} {total-accounted:>8.4f} "
          f"{(total-accounted)/total*100:>14.1f}%")
    print()
    print("⚠️ 判读纪律：")
    print("  • 并行化的上限 = 各块耗时中【可并行部分】的占比；串行控制流部分无法摊掉")
    print("  • event/command 会写 sim state，并行需先解决依赖，不一定可行")
    print("  • 本机 ms/step 噪声底约 4-13%，单次测量不下结论")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
