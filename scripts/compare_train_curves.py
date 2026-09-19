#!/usr/bin/env python3
"""对比两次训练的 reward 曲线，出十分位表。

为什么单独写一个
----------------
`run_summary.json` 里的 `best_mean_reward` 是派生量，不知道它的窗口定义就下结论会翻车：
同一批数据里 `best_mean_reward` 给出 0.981（像"打平"），而 `Train/mean_reward` 的
逐迭代原始序列给出 0.85、十分位从第 600 轮起稳定 0.73-0.83。
**取 max / 取 best 的指标天生偏乐观** —— 判训练质量要看原始曲线。

数据来源：训练日志里每个 iteration 打一行 `Mean reward: <值>`，所以 3000 轮 = 3000 个点，
不需要装 tensorboard 也能还原整条曲线。

用法：
    python3 scripts/compare_train_curves.py A.log B.log [--label-a startup] [--label-b reset]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_PATTERN = re.compile(r"Mean reward:\s*([-\d.]+)")


def series(path: str | Path) -> np.ndarray:
    text = Path(path).read_text(errors="ignore")
    return np.array([float(x) for x in _PATTERN.findall(text)], dtype=float)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log_a")
    ap.add_argument("log_b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--deciles", type=int, default=10)
    args = ap.parse_args(argv)

    a, b = series(args.log_a), series(args.log_b)
    if a.size == 0 or b.size == 0:
        print("有一个日志里没抓到 `Mean reward:` —— 训练没跑到，或者日志格式变了")
        return 1
    print(f"{args.label_a}: {a.size} 个点   {args.label_b}: {b.size} 个点")

    print(f"\n  {'指标':<28s} {args.label_a:>10s} {args.label_b:>10s} {'A/B':>7s}")
    print("  " + "-" * 60)
    for name, x, y in (
        ("max", a.max(), b.max()),
        ("末值", a[-1], b[-1]),
        ("全程均值", a.mean(), b.mean()),
        ("后 25% 均值", a[int(len(a) * 0.75):].mean(), b[int(len(b) * 0.75):].mean()),
    ):
        print(f"  {name:<28s} {x:>10.2f} {y:>10.2f} {x / y:>7.3f}")

    n = min(len(a), len(b))
    k = args.deciles
    print(f"\n  十分位（每段 {(n + k - 1) // k} 轮）")
    print(f"  {'迭代段':<14s} {args.label_a:>8s} {args.label_b:>8s} {'A/B':>7s}   {'':<20s}")
    for i in range(k):
        sa = a[i * n // k:(i + 1) * n // k]
        sb = b[i * n // k:(i + 1) * n // k]
        ma, mb = sa.mean(), sb.mean()
        r = ma / mb if mb else float("nan")
        bar = "#" * int(max(0.0, min(20.0, (r - 0.5) * 20)))
        print(f"  {i * n // k:>5d}-{(i + 1) * n // k:>5d}   {ma:>8.2f} {mb:>8.2f} {r:>7.3f}   {bar}")

    below = sum(
        1 for i in range(k)
        if a[i * n // k:(i + 1) * n // k].mean() < b[i * n // k:(i + 1) * n // k].mean()
    )
    print(f"\n  {args.label_a} 落后的十分位：{below}/{k}")
    print("  ⚠️ n=1 seed 时这只能说明「有差距的形状」，不能说明差距是机制还是种子噪声。")
    print("     跨 seed 方差通常远大于此 —— 要下结论必须每臂多跑几个 seed。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
