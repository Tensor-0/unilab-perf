#!/usr/bin/env python3
"""给上游 issue 用的证据：`mujoco_warp.set_const` 的发射开销拆解。

背景：mjwarp 后端把 step / forward / reset 都抓成了 CUDA graph
（`unisim/backend/mjwarp/backend.py:697-725`），但 reset 途中做模型域随机化时
直接裸调 `mujoco_warp.set_const()`（同文件 1546 行），没有 graph 分支。
实测这一次调用恒定 ~5.4 ms，且与 nworld 无关（64 和 512 都是 5.3 ms）。

2026-09-20 实测结论（nworld=512，dm10: nv=16 nbody=13 ngeom=25）：
    墙钟 5.89 ms，其中 host 侧发射 5.39 ms（92%），399 次调用、平均 13.5 µs/次
    主要来源是 io.py:3405 的 dof 循环 和 io.py:3425-3451 的 body×row 嵌套循环
    （`_compute_body_jac_row` / `_compute_body_A_diag_entry` 各 72 次，
      `solve_m`→`_tile_cholesky_solve_block` 98 次）
    ⇒ 瓶颈是逐 kernel 的派发，不是 GPU 算不动；GPU 侧几乎是空的。

本脚本回答三个问题：
  1. 一次 set_const 到底发多少次 kernel / 多少次内存操作
  2. 时间花在「host 侧发射」还是「GPU 执行」
  3. 哪些 kernel 贡献最多

用法：
    UNILAB_ROOT=$HOME/UniLab uv run --project $UNILAB_ROOT \
        python scripts/mjwarp_setconst_issue_evidence.py --num-envs 512
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

REC: dict[str, list[float]] = defaultdict(list)


def _name(obj) -> str:
    for attr in ("key", "__name__", "name"):
        v = getattr(obj, attr, None)
        if isinstance(v, str) and v:
            return v
    return type(obj).__name__


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
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    import warp
    import mujoco_warp

    env = build_env(args.num_envs, args.task)
    b = env._backend
    m, d = b._device_model, b._device_data

    rng = np.random.default_rng(0)
    a = rng.uniform(-0.2, 0.2, size=(args.num_envs, env.action_space.shape[-1])).astype(np.float32)
    for _ in range(5):
        env.step(a)
    torch.cuda.synchronize()

    # ---- 打点：wp.launch / wp.copy / 分配 ----
    patches = []
    for mod, fname in ((warp, "launch"), (warp, "launch_tiled"), (warp, "copy")):
        if not hasattr(mod, fname):
            continue
        orig = getattr(mod, fname)
        tag = f"warp.{fname}"

        def mk(orig=orig, tag=tag):
            def w(*a_, **kw):
                nm = _name(a_[0]) if a_ else "?"
                t0 = time.perf_counter()
                r = orig(*a_, **kw)
                REC[f"{tag}::{nm}"].append((time.perf_counter() - t0) * 1000.0)
                return r
            return w
        setattr(mod, fname, mk())
        patches.append((mod, fname, orig))

    allocs = []
    for mod, fname in ((warp, "zeros"), (warp, "empty"), (warp, "clone")):
        if not hasattr(mod, fname):
            continue
        orig = getattr(mod, fname)
        tag = f"warp.{fname}"

        def mk2(orig=orig, tag=tag):
            def w(*a_, **kw):
                t0 = time.perf_counter()
                r = orig(*a_, **kw)
                allocs.append((tag, (time.perf_counter() - t0) * 1000.0))
                return r
            return w
        setattr(mod, fname, mk2())
        patches.append((mod, fname, orig))

    # ---- 测 set_const ----
    walls = []
    for _ in range(args.reps):
        REC.clear()
        allocs.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mujoco_warp.set_const(m, d)
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t0) * 1000.0)

    for mod, fname, orig in patches:
        setattr(mod, fname, orig)

    # ---- 对照组：把同样的 work 塞进 CUDA graph ----
    # 背靠背发 N 次、只在末尾 sync 一次，这样量到的是 max(host 派发, GPU 执行)。
    # eager 5.33 ms > replay 2.14 ms ⇒ 说明 GPU 侧只要 2.1 ms，瓶颈在 host 派发。
    graph_ms = float("nan")
    try:
        with warp.ScopedCapture() as cap:
            mujoco_warp.set_const(m, d)
        g = cap.graph

        def _bench(fn, reps=20):
            fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / reps * 1000.0

        eager_ms = _bench(lambda: mujoco_warp.set_const(m, d))
        graph_ms = _bench(lambda: warp.capture_launch(g))
    except Exception as exc:  # noqa: BLE001 - 捕获失败本身就是结论
        eager_ms = float("nan")
        print(f"\n  [graph 对照] 捕获失败: {type(exc).__name__}: {str(exc)[:200]}")

    n_calls = sum(len(v) for v in REC.values())
    host_ms = sum(sum(v) for v in REC.values())
    wall = sum(walls) / len(walls)

    print(f"\nnworld={args.num_envs}  nv={m.nv}  nbody={m.nbody}  ngeom={m.ngeom}")
    print(f"set_const 墙钟          = {wall:.3f} ms  (min {min(walls):.3f} / max {max(walls):.3f}，{args.reps} 次)")
    print(f"总 launch/copy 调用数   = {n_calls}")
    print(f"其中 host 侧发射耗时    = {host_ms:.3f} ms  -> 占墙钟 {host_ms/wall*100:.0f}%")
    print(f"单次平均发射耗时        = {host_ms/n_calls*1000:.1f} µs")
    print(f"分配调用（zeros/empty/clone） = {len(allocs)} 次, "
          f"合计 {sum(t for _, t in allocs):.3f} ms")

    print(f"\n  {'kernel':<52s} {'次数':>6s} {'host ms':>9s} {'µs/次':>7s}")
    print("  " + "-" * 78)
    for k, v in sorted(REC.items(), key=lambda kv: -sum(kv[1]))[:18]:
        print(f"  {k[:52]:<52s} {len(v):>6d} {sum(v):>9.3f} {sum(v)/len(v)*1000:>7.1f}")
    print("  " + "-" * 78)
    print(f"  {'合计':<52s} {n_calls:>6d} {host_ms:>9.3f}")
    if graph_ms == graph_ms:  # not NaN
        print(f"\n  对照组（背靠背，只在末尾 sync 一次）：")
        print(f"    eager 逐 kernel 发射（现状）  {eager_ms:7.3f} ms")
        print(f"    同样 work 走 CUDA graph       {graph_ms:7.3f} ms   -> {eager_ms/graph_ms:.2f}x")
        print(f"    差值 {eager_ms - graph_ms:.3f} ms 就是逐 kernel 的 host 派发开销")
    print("\n  读法：eager 明显慢于 graph ⇒ 瓶颈是逐 kernel 的发射/派发，不是 GPU 算不动。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
