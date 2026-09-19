#!/usr/bin/env python3
"""nan_guard 代价的隔离测量（A/B 交替，固定随机动作）。

为什么不能用训练 run 做 A/B
---------------------------
实测发现：`Perf/collection_time` 的时序趋势**主要受【策略行为】驱动**，不是硬件。
（前 26 轮策略还近似随机 vs 后期学到的策略 —— 物理接触状态不同，代价就不同。
 实测温度只解释 2.7%（r=0.163），而分段趋势显示 +10.3%。）
⇒ 要测硬件/插桩的代价，**负载行为必须恒定**。

本脚本用**固定随机动作**跑同一个 env，交替开/关 nan_guard，从而把它的代价隔离出来。

为什么要"交替"而不是"先关后开"
-----------------------------
机器会热、缓存会暖、chunk_size 会自适应 —— 单向前后对比会把这些漂移混进来。
A/B/A/B 交替可让漂移在两组间大致均摊。

已核实的机制（读源码得出，非转述）
--------------------------------
- 唯一挂载点：`scripts/train_rsl_rl.py:558` → `training/run.py:91 apply_env_nan_guard`
- `resolve_nan_guard_cfg`（run.py:76-89）在配置 `enabled=False` 时返回 None ⇒ **不挂载**
- `np_env.py:251-261` 的 `capture(...)` 形参**提前求值** ⇒ 只要挂了 guard，
  每步都付一次 `get_physics_state_snapshot()`（全状态 detach 拷贝）
- `nan_guard.py:52/70` 的 `enabled` 检查只挡 check/check_ctrl 的 isfinite 扫描

用法
----
    uv run python scripts/observe/nan_guard_ab.py --num-envs 512 --rounds 4 --steps 80
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = str(Path(os.environ.get("UNILAB_ROOT", Path.home() / "UniLab")).resolve())
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


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
    env_cfg_override = BackendAdapter(
        cfg, root_dir=REPO_ROOT, algo_name=str(cfg.algo.algo)
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=num_envs, env_cfg_override=env_cfg_override)
    if env.state is None:
        env.init_state()
    return env


def make_guard(env, buffer_size: int):
    from unilab.utils.nan_guard import NanGuard, NanGuardCfg

    return NanGuard(
        NanGuardCfg(enabled=True, buffer_size=buffer_size, max_envs_to_dump=5, output_dir=None),
        num_envs=env.num_envs,
        supports_state_playback=env.play_capabilities.supports_physics_state_playback,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--task", default="dm10_joystick_flat/mujoco")
    ap.add_argument("--config-group", default="ppo")
    ap.add_argument("--rounds", type=int, default=4, help="A/B 交替轮数")
    ap.add_argument("--steps", type=int, default=80, help="每轮步数")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--buffer-size", type=int, default=100)
    args = ap.parse_args(argv)

    env = build_env(args.num_envs, args.task, args.config_group)
    import numpy as np

    rng = np.random.default_rng(0)
    adim = env.action_space.shape[-1]

    def rand_actions():
        return rng.uniform(-0.2, 0.2, size=(args.num_envs, adim)).astype(np.float32)

    # 预热
    for _ in range(args.warmup):
        env.step(rand_actions())

    # 单独量一次快照拷贝的代价（这是 nan_guard 每步的固定开销）
    snap_ms = None
    if env.play_capabilities.supports_physics_state_playback:
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            env.get_physics_state_snapshot()
            ts.append((time.perf_counter() - t0) * 1000.0)
        snap_ms = sum(ts) / len(ts)

    guard = make_guard(env, args.buffer_size)

    res: dict[str, list[float]] = {"off": [], "on": []}
    print(f"env={args.task} num_envs={args.num_envs}  steps/round={args.steps} rounds={args.rounds}")
    if snap_ms is not None:
        print(f"get_physics_state_snapshot() 单次 ≈ {snap_ms:.3f} ms"
              f"   （每步一次 ⇒ {snap_ms/15.1*100:.1f}% of a 15.1ms step）")
    print()

    for r in range(args.rounds):
        for mode in ("off", "on"):
            env.set_nan_guard(None if mode == "off" else guard)
            acts = [rand_actions() for _ in range(args.steps)]  # 预生成，避免计入
            t0 = time.perf_counter()
            for a in acts:
                env.step(a)
            dt = (time.perf_counter() - t0) / args.steps * 1000.0
            res[mode].append(dt)
            print(f"  轮{r+1} nan_guard={mode:3s}: {dt:7.3f} ms/步")

    off = sum(res["off"]) / len(res["off"])
    on = sum(res["on"]) / len(res["on"])
    print()
    print("═" * 62)
    print(f"  nan_guard OFF : {off:7.3f} ms/步   ({[f'{x:.2f}' for x in res['off']]})")
    print(f"  nan_guard ON  : {on:7.3f} ms/步   ({[f'{x:.2f}' for x in res['on']]})")
    print(f"  代价          : {on-off:+.3f} ms/步   ({(on/off-1)*100:+.2f}%)")
    if off > 0:
        print(f"  ⇒ 关掉它可提速采集约 {(1-off/on)*100:.1f}%")
    print("═" * 62)
    print("⚠️ 噪声底：本 harness 的 ms/step 单轮波动约 4-13%（见 memory）；")
    print("   若 |代价| 小于该量级，则不可下结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
