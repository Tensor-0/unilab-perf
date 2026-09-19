#!/usr/bin/env python3
"""拆 mjwarp reset 里那 5.7 ms 的域随机化到底花在哪。

不修改任何文件：用 monkeypatch 在运行时给 MjwarpBackend 的关键方法套计时器。
（unisim-core 是 PyPI 装的，site-packages 里的文件硬链接到 uv 缓存，
 直接改会写穿硬链接污染缓存，所以这里只做运行时插桩。）

用法：
    UNILAB_ROOT=$HOME/UniLab uv run --project $UNILAB_ROOT \
        python scripts/mjwarp_reset_probe.py --num-envs 512 --steps 200
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

UNILAB_ROOT = str(Path(os.environ.get("UNILAB_ROOT", Path.home() / "UniLab")).resolve())
sys.path.insert(0, os.path.join(UNILAB_ROOT, "src"))
os.chdir(UNILAB_ROOT)

ACC: dict[str, list[float]] = defaultdict(list)


def _instrument(cls, name: str) -> None:
    fn = getattr(cls, name, None)
    if fn is None:
        print(f"  [warn] {cls.__name__}.{name} 不存在，跳过", file=sys.stderr)
        return

    def wrapped(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(self, *a, **kw)
        finally:
            ACC[name].append((time.perf_counter() - t0) * 1000.0)

    setattr(cls, name, wrapped)


def build_env(num_envs: int, task: str):
    import hydra
    from omegaconf import OmegaConf

    from unilab.base.config_adapter import BackendAdapter, create_env
    from unilab.training import ensure_registries

    ensure_registries()
    with hydra.initialize_config_dir(
        version_base="1.3",
        config_dir=os.path.join(UNILAB_ROOT, "src", "unilab", "conf", "ppo"),
    ):
        cfg = hydra.compose(config_name="config",
                            overrides=[f"task={task}", f"algo.num_envs={num_envs}"])
    OmegaConf.resolve(cfg)
    override = BackendAdapter(cfg, root_dir=UNILAB_ROOT,
                              algo_name=str(cfg.algo.algo)).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=num_envs, env_cfg_override=override)
    if env.state is None:
        env.init_state()
    return env


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--task", default="dm10_joystick_flat/mjwarp")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args(argv)

    env = build_env(args.num_envs, args.task)

    from unisim.backend.mjwarp.backend import MjwarpBackend
    import mujoco_warp

    for nm in ("_apply_reset_randomization", "_upload", "_download", "_synchronize",
               "_refresh_host_cache", "_execute_device_forward", "_execute_device_reset",
               "_execute_device_steps", "_execute_reset_scratch_forward",
               "_refresh_reset_scratch_cache"):
        _instrument(MjwarpBackend, nm)
    # set_const 系列是 warp kernel，单独包一层（模块级函数）
    for nm in ("set_const", "set_const_0"):
        _instrument(mujoco_warp, nm)

    backend = env._backend
    print(f"\ntask={args.task} num_envs={args.num_envs}")
    print(f"reset_scratch_capacity = {backend._reset_scratch_capacity}")
    print(f"cuda_graph_enabled     = {backend._cuda_graph_enabled}")

    import numpy as np
    rng = np.random.default_rng(0)
    adim = env.action_space.shape[-1]

    def acts():
        return rng.uniform(-0.2, 0.2, size=(args.num_envs, adim)).astype(np.float32)

    for _ in range(args.warmup):
        env.step(acts())
    for v in ACC.values():
        v.clear()

    n_reset = 0
    reset_steps = 0
    t0 = time.perf_counter()
    for _ in range(args.steps):
        st = env.step(acts())
        n = int(np.count_nonzero(st.terminated | st.truncated))
        if n:
            reset_steps += 1
        n_reset += n
    wall = (time.perf_counter() - t0) * 1000.0

    print(f"\n{args.steps} 步 / {wall/args.steps:.2f} ms per step / "
          f"{n_reset} 次 reset（{reset_steps} 步里有 reset，"
          f"平均每次 reset 事件 {n_reset/max(reset_steps,1):.2f} 个 env）\n")
    print(f"  {'method':<34s} {'总 ms':>10s} {'每步 ms':>9s} {'调用次数':>8s} {'单次 µs':>9s}")
    print("  " + "-" * 76)
    for k, v in sorted(ACC.items(), key=lambda kv: -sum(kv[1])):
        tot = sum(v)
        if tot <= 0:
            continue
        print(f"  {k:<34s} {tot:>10.2f} {tot/args.steps:>9.3f} {len(v):>8d} "
              f"{tot/len(v)*1000:>9.1f}")
    print("  " + "-" * 76)
    print(f"  {'（整步 wall）':<34s} {'':>10s} {wall/args.steps:>9.3f}")
    print("\n  注：嵌套方法（_apply_reset_randomization 内部会调 _upload）会重复计时，")
    print("      看「总 ms」列时按调用层级理解，不要直接相加。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
