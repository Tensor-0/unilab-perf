#!/usr/bin/env python3
"""把 `reward.compute`（update_state 的 57.7%）拆到 16 个奖励项。

结论来自 `update_state_breakdown.py`：
    update_state = 1.915 ms/步（占整步 13.1%）
      └─ reward.compute = 1.106 ms ← 占 57.7%，绝对大头
         其余 manager 加起来才 0.95 ms

⇒ 要动 update_state，就必须先看清 reward 里哪一项贵。

关键性质（决定并行化可行性）
--------------------------
`reward` 的每一项都是**纯读**：`term_cfg.func(env, **params)` 读 state 算标量，
**不写 sim state**（对比 `event`/`command` 会写）⇒ **理论上可并行**。
但 `compute()` 里有个共享 scratch 和 `self._reward_buf += scratch` 的累加，
并行要先把这两处改成 per-shard。

用法
----
    uv run python scripts/observe/reward_breakdown.py --num-envs 512 --steps 200
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
        cfg = hydra.compose(config_name="config",
                            overrides=[f"task={task}", f"algo.num_envs={num_envs}"])
    OmegaConf.resolve(cfg)
    override = BackendAdapter(cfg, root_dir=REPO_ROOT,
                              algo_name=str(cfg.algo.algo)).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=num_envs, env_cfg_override=override)
    if env.state is None:
        env.init_state()
    return env


class TermProxy:
    """透明代理 reward term cfg，只把 `.func` 换成计时版。"""

    def __init__(self, cfg, sink: list[float]):
        object.__setattr__(self, "_cfg", cfg)
        object.__setattr__(self, "_sink", sink)

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_cfg"), k)

    @property
    def func(self):
        cfg = object.__getattribute__(self, "_cfg")
        sink = object.__getattribute__(self, "_sink")

        def wrapped(env, **kw):
            t0 = time.perf_counter()
            try:
                return cfg.func(env, **kw)
            finally:
                sink.append((time.perf_counter() - t0) * 1000.0)

        return wrapped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--task", default="dm10_joystick_flat/mujoco")
    ap.add_argument("--config-group", default="ppo")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args(argv)

    env = build_env(args.num_envs, args.task, args.config_group)
    rm = env.reward_manager
    names = list(rm._term_names)
    sink: dict[str, list[float]] = defaultdict(list)
    rm._term_cfgs = [TermProxy(c, sink[n]) for n, c in zip(names, rm._term_cfgs, strict=False)]

    # reward.compute 整体（用于核对分项之和）
    total_sink: list[float] = []
    _orig_compute = rm.compute

    def compute_timed(dt):
        t0 = time.perf_counter()
        try:
            return _orig_compute(dt)
        finally:
            total_sink.append((time.perf_counter() - t0) * 1000.0)

    rm.compute = compute_timed

    import numpy as np
    rng = np.random.default_rng(0)
    adim = env.action_space.shape[-1]
    for _ in range(args.warmup):
        env.step(rng.uniform(-0.2, 0.2, size=(args.num_envs, adim)).astype(np.float32))
    for v in sink.values():
        v.clear()
    total_sink.clear()

    acts = [rng.uniform(-0.2, 0.2, size=(args.num_envs, adim)).astype(np.float32)
            for _ in range(args.steps)]
    t0 = time.perf_counter()
    for a in acts:
        env.step(a)
    wall = (time.perf_counter() - t0) / args.steps * 1000.0

    tot = sum(total_sink) / len(total_sink)
    print(f"\nenv={args.task} num_envs={args.num_envs} steps={args.steps}")
    print(f"整步 wall={wall:.3f} ms   reward.compute={tot:.4f} ms "
          f"({tot/wall*100:.1f}% of step)\n")
    print(f"  {'奖励项':<26s} {'ms/次':>9s} {'占 reward':>11s} {'占整步':>9s}")
    print("  " + "-" * 60)
    rows = sorted(((sum(v)/len(v), n) for n, v in sink.items() if v), reverse=True)
    for m, n in rows:
        print(f"  {n:<26s} {m:>9.4f} {m/tot*100:>10.1f}% {m/wall*100:>8.2f}%")
    s = sum(r[0] for r in rows)
    print("  " + "-" * 60)
    print(f"  {'分项之和':<26s} {s:>9.4f} {s/tot*100:>10.1f}%")
    print(f"  {'未归因（循环/scratch/log）':<26s} {tot-s:>9.4f} {(tot-s)/tot*100:>10.1f}%")
    print()
    top = rows[:5]
    print(f"  Top5 占比 = {sum(r[0] for r in top)/tot*100:.1f}%")
    print(f"  并行化上限（若全部可并行、用 N 核）= 最多把 reward 的 {100.0:.0f}% 摊到 N 核")
    print("  ⚠️ 但 compute() 里的 `self._reward_buf += scratch` 是共享累加，并行前必须先改掉")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
