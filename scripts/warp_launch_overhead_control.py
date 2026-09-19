#!/usr/bin/env python3
"""对照实验：一个「零成本」的 warp kernel，逐次发射要多少 µs？

用途：验证 set_const 那 399 次调用里 13.5 µs/次 到底是「发射开销」还是
「GPU 反压」。如果连一个什么都不干的 kernel 也是 ~13 µs/次，
那 13.5 µs 就是纯发射成本，与 kernel 内容无关。

（必须在文件里定义 kernel —— warp 不支持 exec() 定义的代码。）
"""
from __future__ import annotations

import time

import numpy as np
import torch
import warp as wp


@wp.kernel
def _noop(out: wp.array1d[float]):
    i = wp.tid()
    if i == 0:
        out[0] = out[0] + 0.0


def bench(n: int, dim: int, reps: int = 3) -> float:
    dev = "cuda:0"
    out = wp.zeros(8, dtype=float, device=dev)
    for _ in range(20):  # warm
        wp.launch(_noop, dim=dim, inputs=[out], device=dev)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            wp.launch(_noop, dim=dim, inputs=[out], device=dev)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / n * 1000.0)
    return float(np.median(ts))


def main() -> int:
    print("\n对照：零成本 noop kernel 的逐次发射成本（背靠背 N 次，末尾 sync 一次）\n")
    print(f"  {'N':>6s} {'dim':>7s} {'ms/次':>10s} {'µs/次':>8s}")
    print("  " + "-" * 36)
    for dim in (64, 512):
        for n in (1, 10, 100, 400):
            ms = bench(n, dim)
            print(f"  {n:>6d} {dim:>7d} {ms:>10.4f} {ms*1000:>8.1f}")
    print()
    single = bench(1, 512)
    many = bench(400, 512)
    print(f"  N=1 时 {single*1000:.1f} µs —— 含首次发射的固定成本（寻址/校验/PDL 等）")
    print(f"  N=400 摊薄后 {many*1000:.1f} µs/次 —— 这才是「多一次发射」的边际成本")
    print()
    print(f"  => set_const 实测 399 次调用、host 侧 13.5 µs/次")
    print(f"     若与 noop 的边际成本同量级 ⇒ 那 5.3 ms 确实是发射开销，与 kernel 内容无关")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
