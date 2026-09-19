#!/usr/bin/env python3
"""拆解 reset_done（占一个控制步约 13.9%，是最后一块没查过的）。

不需要新加插桩
--------------
`src/unilab/base/np_env.py` 已经在 `_reset_done_envs()` 里写了 43 个细节计时 key
（`reset_done_*` / `dr_reset_*` / `set_state_*`），只是没有消费者。
本脚本直接读 `state.info["timing"]` 汇总。

它会做三件事：
  1. 用固定随机动作跑，行为恒定（不受策略漂移影响）
  2. 只统计【真的发生了 reset】的那些步（`reset_done_count > 0`），其余步这些 key 是 0
  3. 按 key 汇总 mean/max，并算出占 reset_done 的比例

用法：
    UNILAB_ROOT=$HOME/UniLab uv run --project $UNILAB_ROOT python scripts/reset_breakdown.py \
        --num-envs 512 --steps 300
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

UNILAB_ROOT = str(Path(os.environ.get("UNILAB_ROOT", Path.home() / "UniLab")).resolve())
sys.path.insert(0, os.path.join(UNILAB_ROOT, "src"))
# 模型路径在配置里是相对的（src/unilab/assets/...），所以必须切到检出目录
os.chdir(UNILAB_ROOT)

# reset_done_ms 之外，np_env 写的细节 key 都在这几个前缀下
PREFIXES = ("reset_done_", "dr_reset_", "set_state_")
# 这些是计数/标志，不是毫秒，不能参与求和
NON_MS_KEYS = ("reset_done_count",)


def build_env(num_envs: int, task: str, config_group: str):
    import hydra
    from omegaconf import OmegaConf

    from unilab.base.config_adapter import BackendAdapter, create_env
    from unilab.training import ensure_registries

    ensure_registries()
    with hydra.initialize_config_dir(
        version_base="1.3",
        config_dir=os.path.join(UNILAB_ROOT, "src", "unilab", "conf", config_group),
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
    ap.add_argument("--task", default="dm10_joystick_flat/mujoco")
    ap.add_argument("--config-group", default="ppo")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args(argv)

    env = build_env(args.num_envs, args.task, args.config_group)
    import numpy as np

    rng = np.random.default_rng(0)
    adim = env.action_space.shape[-1]

    def acts():
        return rng.uniform(-0.2, 0.2, size=(args.num_envs, adim)).astype(np.float32)

    for _ in range(args.warmup):
        env.step(acts())

    acc: dict[str, list[float]] = defaultdict(list)
    total_steps = 0
    reset_steps = 0
    reset_counts: list[int] = []
    step_total_ms: list[float] = []
    # 按 reset 个数分箱，用来拟合「固定开销 + 每个 env 的边际开销」
    by_count: dict[int, list[float]] = defaultdict(list)

    for _ in range(args.steps):
        state = env.step(acts())
        timing = (state.info or {}).get("timing") or {}
        total_steps += 1
        if "env_step_total_ms" in timing:
            step_total_ms.append(float(timing["env_step_total_ms"]))
        n = timing.get("reset_done_count", 0)
        try:
            n = int(n)
        except (TypeError, ValueError):
            n = 0
        rd = timing.get("reset_done_ms")
        if rd is not None:
            try:
                by_count[n].append(float(rd))
            except (TypeError, ValueError):
                pass
        if n <= 0:
            continue
        reset_steps += 1
        reset_counts.append(n)
        for k, v in timing.items():
            if k in NON_MS_KEYS or not k.startswith(PREFIXES):
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv > 0:
                acc[k].append(fv)

    mean_step = sum(step_total_ms) / len(step_total_ms) if step_total_ms else float("nan")
    print(f"\nenv={args.task} num_envs={args.num_envs} steps={total_steps}")
    print(f"其中发生 reset 的步：{reset_steps}（{reset_steps/total_steps*100:.1f}%）"
          f"，平均每步 reset {np.mean(reset_counts) if reset_counts else 0:.1f} 个 env")
    print(f"整步 wall（env_step_total_ms 均值）= {mean_step:.3f} ms\n")

    if not acc:
        print("  没有采到 reset 细节计时 —— 可能是这批 step 里没触发 reset，或 key 名变了")
        return 0

    rd = acc.get("reset_done_ms") or []
    base = sum(rd) / len(rd) if rd else float("nan")
    print(f"  reset_done_ms（只含有 reset 的步）= {base:.4f} ms")
    print()
    print(f"  {'key':<34s} {'ms':>9s} {'占 reset_done':>14s} {'占整步':>9s}")
    print("  " + "-" * 70)
    rows = sorted(((sum(v) / len(v), k) for k, v in acc.items() if k != "reset_done_ms"),
                  reverse=True)
    for m, k in rows:
        pct_base = f"{m/base*100:.1f}%" if base == base else "-"
        print(f"  {k:<34s} {m:>9.4f} {pct_base:>14s} {m/mean_step*100:>8.2f}%")
    s = sum(r[0] for r in rows)
    print("  " + "-" * 70)
    print(f"  {'已归因小计':<34s} {s:>9.4f} {s/base*100:>13.1f}% {s/mean_step*100:>8.2f}%")
    print(f"  {'未归因':<34s} {base-s:>9.4f} {(base-s)/base*100:>13.1f}%")

    print()
    print("  提示：reset 只在部分步发生，所以「占整步」这一列是按【全部步】算的，")
    print("        反映的是它摊到每一步的平均成本。")

    # ---- reset 个数 vs 成本：分离固定开销与边际开销 ----
    pts = [(n, sum(v) / len(v), len(v)) for n, v in sorted(by_count.items()) if v]
    if len(pts) >= 3:
        print()
        print("  ── reset_done_ms 随 reset 个数的变化 ──")
        print(f"  {'reset 个数':>10s} {'步数':>6s} {'reset_done_ms':>14s}")
        for n, m, c in pts:
            print(f"  {n:>10d} {c:>6d} {m:>14.4f}")
        # 最小二乘拟合 y = a + b*n
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        ws = np.array([p[2] for p in pts], dtype=float)
        b, a = np.polyfit(xs, ys, 1, w=np.sqrt(ws))
        print()
        print(f"  拟合：reset_done_ms ≈ {a:.4f} + {b:.4f} × (reset 个数)")
        print(f"        固定开销 {a:.4f} ms，每个 env 边际 {b:.4f} ms")
        for ref, tag in ((2.87, "09-18 那条 3000 轮（稳态）"), (7.8, "本次随机动作测试")):
            print(f"        按 {tag} 的 {ref} env/步 → {a + b*ref:.4f} ms"
                  f"  （占 {mean_step:.2f}ms 整步的 {(a+b*ref)/mean_step*100:.1f}%）")
        print()
        print("  ⚠️ 本次用随机动作，摔倒远多于训练。换算到真实训练要用上面的拟合式，")
        print("     不能直接拿测出来的绝对值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
