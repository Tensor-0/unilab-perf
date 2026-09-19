#!/usr/bin/env python3
"""把 `mujoco_warp.set_const` 录进 CUDA graph 后，replay 出来的数还对吗？

为什么必须做这个
----------------
mjlab 文档说明 event manager 逻辑「runs as regular Python between graph
replays, so it will not break graph capture」—— 即【不捕获是有意设计】。
所以「没捕获」不是 bug，不能按 bug 报。

但 `set_const` 单次 5.4 ms、与 nworld 无关（399 次 kernel 发射）是实测事实，
值得作为「成本归因更正 + 可优化证据」报上去。**前提是捕获后数值正确** ——
否则这个建议本身就是错的。

三个具体的失效模式，逐个测：

  测 1  立即 replay            —— 基础正确性
  测 2  中间跑 env.step 制造内存 churn 后再 replay
                              —— 验「捕获期分配的临时数组被 GC ⇒ 内存被复用 ⇒ replay 写坏别人的缓冲」
  测 3  同一状态连续 replay 两次
                              —— 验「wp.zeros 的归零有没有被录进 graph」。
                                 若没录，`_*_accumulate` 这类累加 kernel 会在上一次的
                                 残留值上继续累加 ⇒ 第二次 replay 结果不同

判据：与 eager 路径逐元素比。测 3 若第二次 ≠ 第一次，就是**实锤不能直接捕获**。

用法：
    UNILAB_ROOT=$HOME/UniLab uv run --project $UNILAB_ROOT \
        python scripts/mjwarp_setconst_capture_correctness.py --num-envs 512
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

UNILAB_ROOT = str(Path(os.environ.get("UNILAB_ROOT", Path.home() / "UniLab")).resolve())
sys.path.insert(0, os.path.join(UNILAB_ROOT, "src"))
os.chdir(UNILAB_ROOT)

# 由 set_const* 重算的派生量（randomization.py 的 DERIVED_FIELDS + 相机/光源）
CANDIDATE_FIELDS = (
    "body_subtreemass",
    "dof_invweight0",
    "body_invweight0",
    "tendon_length0",
    "tendon_invweight0",
    "actuator_acc0",
    "cam_pos0",
    "light_pos0",
)


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


def snapshot(m, names):
    import numpy as np
    out = {}
    for n in names:
        arr = getattr(m, n, None)
        if arr is None:
            continue
        a = np.array(arr.numpy(), copy=True)
        if a.size == 0:  # dm10 无腱 ⇒ tendon_length0 是空数组，比不了
            continue
        out[n] = a
    return out


def compare(ref, got, label):
    import numpy as np
    print(f"\n  {label}")
    print(f"  {'field':<22s} {'shape':>14s} {'max|Δ|':>13s} {'相对':>11s} {'判定':>6s}")
    print("  " + "-" * 72)
    worst = 0.0
    for n in ref:
        a, b = ref[n], got[n]
        if a.shape != b.shape:
            print(f"  {n:<22s} {str(a.shape):>14s}  形状不一致!")
            worst = float("inf")
            continue
        d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        scale = float(np.max(np.abs(a.astype(np.float64)))) or 1.0
        rel = d / scale
        worst = max(worst, rel)
        flag = "OK" if d == 0.0 else ("≈0" if rel < 1e-12 else "差异!")
        print(f"  {n:<22s} {str(a.shape):>14s} {d:>13.3e} {rel:>11.2e} {flag:>6s}")
    print("  " + "-" * 72)
    return worst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--task", default="dm10_joystick_flat/mjwarp")
    ap.add_argument("--churn-steps", type=int, default=60)
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    import warp
    import mujoco_warp

    env = build_env(args.num_envs, args.task)
    b = env._backend
    m, d = b._device_model, b._device_data

    rng = np.random.default_rng(0)
    ad = env.action_space.shape[-1]
    acts = rng.uniform(-0.2, 0.2, size=(args.num_envs, ad)).astype(np.float32)
    for _ in range(10):
        env.step(acts)
    torch.cuda.synchronize()

    names = [n for n in CANDIDATE_FIELDS if getattr(m, n, None) is not None]
    print(f"\nnworld={args.num_envs}   会被比较的派生量: {', '.join(names) or '(一个都没有!)'}")
    if not names:
        print("  ⚠️ 一个派生字段都没找到 —— 字段名可能变了，本实验作废")
        return 1

    bm = getattr(m, "body_mass", None)
    print(f"body_mass shape = {None if bm is None else bm.shape}   (需要 per-world 才能扰动)")

    def perturb(seed: int):
        """给 body_mass 灌一组可复现的随机值，逼派生量真的变。"""
        base = np.asarray(b._default_body_mass, dtype=np.float32)
        r = np.random.default_rng(seed).uniform(0.8, 1.2, size=(args.num_envs, base.shape[0]))
        bm.assign((base[None, :] * r).astype(np.float32))
        torch.cuda.synchronize()

    # ---- 参照：eager 路径 ----
    def eager(seed: int):
        perturb(seed)
        mujoco_warp.set_const(m, d)
        torch.cuda.synchronize()
        return snapshot(m, names)

    # ⚠️ 参照必须是【当场的】eager 调用，不能是程序开头存的快照。
    #    原因：环境自己的 reset 期 DR 会改 body_ipos / dof_armature / body_mass，
    #    而 body_invweight0 依赖 body_ipos、dof_invweight0 依赖 dof_armature。
    #    拿旧快照当参照 = 在比两个不同的模型状态，会凭空造出「差异」。
    def trial(seed: int, use_graph: bool, label: str, graph=None):
        perturb(seed)
        mujoco_warp.set_const(m, d)          # 当场 eager 参照
        torch.cuda.synchronize()
        ref_now = snapshot(m, names)
        perturb(seed)                         # 完全相同的输入
        if use_graph:
            warp.capture_launch(graph if graph is not None else g)
        else:
            mujoco_warp.set_const(m, d)       # 自对照：eager vs eager
        torch.cuda.synchronize()
        return compare(ref_now, snapshot(m, names), label)

    # ---- 正对照：换一个扰动，派生量必须真的变 ----
    # 没有这一步，「全部 0 差异」可能只是因为这个比较根本没有分辨力。
    perturb(1234)
    mujoco_warp.set_const(m, d)
    torch.cuda.synchronize()
    ref = snapshot(m, names)
    perturb(999)
    mujoco_warp.set_const(m, d)
    torch.cuda.synchronize()
    ctrl = compare(ref, snapshot(m, names),
                   "正对照｜不同扰动(1234 vs 999) —— 必须【不一致】，否则本比较无分辨力")
    if ctrl == 0.0:
        print("\n  ⚠️ 正对照失败：换了扰动派生量却没变 ⇒ 本实验测不出任何东西，作废")
        return 1
    print(f"     正对照通过（最大相对差 {ctrl:.3e}）⇒ 下面的 0 差异是真的 0")

    # ---- 测 0：自对照，eager vs eager（验证 trial 这个脚手架本身没问题）----
    w0 = trial(1234, use_graph=False, label="测 0｜自对照 eager vs eager（必须全 0）")

    # ---- 捕获 ----
    print("\n捕获 set_const 进 CUDA graph ...")
    try:
        with warp.ScopedCapture() as cap:
            mujoco_warp.set_const(m, d)
        g = cap.graph
        print("  捕获成功")
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ 捕获失败: {type(exc).__name__}: {str(exc)[:300]}")
        print("\n  ⇒ 结论：捕获路径根本走不通，上游素材里不能建议「加个 graph 分支」。")
        return 0

    # ---- 测 1：立即 replay（参照是当场的 eager）----
    w1 = trial(1234, use_graph=True, label="测 1｜立即 replay vs 当场 eager")

    # ---- 测 3：同一输入连续 replay 两次（验归零是否被录进 graph）----
    perturb(1234)
    warp.capture_launch(g)
    torch.cuda.synchronize()
    first = snapshot(m, names)
    warp.capture_launch(g)
    torch.cuda.synchronize()
    w3 = compare(first, snapshot(m, names),
                 "测 3｜同一输入连续 replay 两次（第2次 vs 第1次）")
    print("        ↑ 若有差异 ⇒ 归零没被录进 graph，累加型 kernel 在残留值上继续加")

    # ---- 测 2：中间跑 env.step 制造内存 churn ----
    print(f"\n中间跑 {args.churn_steps} 步 env.step（制造分配/复用）...")
    for _ in range(args.churn_steps):
        env.step(acts)
    torch.cuda.synchronize()
    w2 = trial(1234, use_graph=True, label="测 2｜churn 之后 replay vs 当场 eager")

    # ---- 测 4：churn 之后【重新捕获】再 replay ----
    # 若这次正确 ⇒ 实锤是「捕获时刻的指针失效」，不是算法本身不可捕获。
    print("\n重新捕获一次（churn 之后）...")
    try:
        with warp.ScopedCapture() as cap2:
            mujoco_warp.set_const(m, d)
        g2 = cap2.graph
        w4 = trial(1234, use_graph=True, label="测 4｜churn 之后【重新捕获】再 replay vs 当场 eager",
                   graph=g2)
    except Exception as exc:  # noqa: BLE001
        print(f"  重新捕获失败: {type(exc).__name__}: {str(exc)[:200]}")
        w4 = float("nan")

    # ---- 结论 ----
    print("\n" + "=" * 74)
    print(f"  测0 自对照(eager vs eager) = {w0:.3e}   <- 脚手架本身必须是 0")
    print(f"  测1 立即 replay            = {w1:.3e}")
    print(f"  测2 churn 后 replay        = {w2:.3e}")
    print(f"  测3 连续 replay 两次        = {w3:.3e}")
    print(f"  测4 churn 后重新捕获         = {w4:.3e}")
    print("-" * 74)
    if w0 != 0.0:
        print("结论：自对照都不为 0 ⇒ 本实验的脚手架有问题，所有结果作废。")
    elif w1 == 0.0 and w2 == 0.0 and w3 == 0.0 and w4 == 0.0:
        print("结论：全部逐位相同 ⇒ 直接套 ScopedCapture 在数值上是安全的，")
        print("      「把 set_const 加进 CUDA graph」这个建议成立，约 2.9x。")
    else:
        print("结论：存在差异 ⇒ 不能直接套 ScopedCapture。")
        if w1 == 0.0 and w2 != 0.0:
            print("      测1 对 / 测2 错 ⇒ 捕获期分配的临时数组在内存 churn 后失效。")
            print("      修法前提：先把临时数组提到调用方持有，再捕获。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
