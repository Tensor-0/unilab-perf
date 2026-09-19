#!/usr/bin/env python3
"""把一次观测的产物渲染成 summary.md，并追加一行到台账。

输入：观测目录（含 phases.json / percore.txt / gpu.csv）
输出：<dir>/summary.md，以及追加到 $REPORTS_ROOT/INDEX.md
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


def parse_gpu(path: Path) -> dict[str, list[float]]:
    """gpu.csv 的列：util, mem_used, sm_clk, power, temp_gpu, temp_mem, throttle"""
    cols: dict[str, list[float]] = {k: [] for k in
                                    ("util", "mem", "clk", "power", "temp", "temp_mem")}
    throttles: set[str] = set()
    if not path.exists():
        return cols
    for line in path.read_text().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 7:
            continue
        for key, i in (("util", 0), ("mem", 1), ("clk", 2),
                       ("power", 3), ("temp", 4), ("temp_mem", 5)):
            if p[i] not in ("", "N/A", "[N/A]"):
                try:
                    cols[key].append(float(p[i]))
                except ValueError:
                    pass
        if p[6] not in ("", "0x0000000000000000", "Not Active"):
            throttles.add(p[6])
    cols["_throttle"] = list(throttles)  # type: ignore[assignment]
    return cols


def main() -> int:
    d = Path(sys.argv[1])
    tag = sys.argv[2]
    rc = sys.argv[3]
    root = Path(os.environ.get("REPORTS_ROOT", str(d.parent)))

    ph = json.loads((d / "phases.json").read_text()) if (d / "phases.json").exists() else {}
    sc = ph.get("scalars", {})
    cfg = ph.get("config", {})
    sm = ph.get("summary", {})

    def g(key: str) -> float | None:
        v = sc.get(key, {}).get("mean")
        return v if isinstance(v, (int, float)) else None

    L: list[str] = []
    L.append(f"# {tag}\n")
    L.append("| | |")
    L.append("|---|---|")
    L.append(f"| 时间 | {d.name} |")
    L.append(f"| run | `{ph.get('run_name', '?')}` |")
    L.append(f"| device | **{cfg.get('device')}** |")
    L.append(f"| num_envs / max_iter | {cfg.get('num_envs')} / {cfg.get('max_iterations')} |")
    L.append(f"| 退出码 | {rc} |")
    tp = sm.get("training_throughput_env_steps_per_sec")
    if tp:
        L.append(f"| 吞吐 | {tp:.0f} env-steps/s |")
    wall = sm.get("training_wall_time_sec")
    if wall:
        L.append(f"| 墙钟 | {wall:.1f} s |")
    L.append("")

    c, lr, fps = g("Perf/collection_time"), g("Perf/learning_time"), g("Perf/total_fps")
    if c and lr:
        tot = c + lr
        L.append("## 分相预算\n")
        L.append("| 阶段 | 秒/轮 | 占比 |")
        L.append("|---|---|---|")
        L.append(f"| collection | {c:.4f} | {c / tot * 100:.1f}% |")
        L.append(f"| learning | {lr:.4f} | {lr / tot * 100:.1f}% |")
        L.append(f"| 合计 | {tot:.4f} | 100% |")
        if fps:
            L.append(f"\n`Perf/total_fps` = {fps:.0f}")
        L.append("")

    pc = d / "percore.txt"
    t = pc.read_text() if pc.exists() else ""
    for title, pat in (("每档核利用率", r"=== 每档利用率.*?===\n(.*?)\n\n"),
                       ("CPU 温度", r"=== CPU 温度 ===\n(.*?)\n\n"),
                       ("降频计数（窗口增量）", r"=== 降频计数.*?===\n(.*?)\n\n"),
                       ("每线程 CPU", r"=== 进程 .*?每线程 CPU.*?===\n(.*?)\n\n")):
        m = re.search(pat, t, re.S)
        if m:
            L.append(f"## {title}\n")
            L.append("```")
            L.append(m.group(1).strip()[:3000])
            L.append("```\n")

    m = re.search(r"最小 ([\d.]+) / 最大 ([\d.]+) / 均值 ([\d.]+) / 极差 ([\d.]+)", t)
    if m:
        L.append(f"等价满载核数：min {m.group(1)} / max {m.group(2)} / "
                 f"mean {m.group(3)} / 极差 {m.group(4)}\n")

    gc = parse_gpu(d / "gpu.csv")
    if gc["util"]:
        L.append("## GPU\n")
        L.append(f"- 利用率：min {min(gc['util']):.0f}% / mean {sum(gc['util']) / len(gc['util']):.0f}% "
                 f"/ max {max(gc['util']):.0f}%")
        L.append(f"- 显存峰值：{max(gc['mem']):.0f} MiB")
        if gc["power"]:
            L.append(f"- 功耗：mean {sum(gc['power']) / len(gc['power']):.1f} W / max {max(gc['power']):.1f} W")
        if gc["temp"]:
            L.append(f"- 温度：core mean {sum(gc['temp']) / len(gc['temp']):.1f}C / max {max(gc['temp']):.0f}C")
        thr = gc.get("_throttle") or []
        L.append(f"- 降频原因：{'; '.join(sorted(thr)) if thr else '无'}\n")

    (d / "summary.md").write_text("\n".join(L) + "\n")
    print(f"写入 {d / 'summary.md'}")

    # 台账
    idx = root / "INDEX.md"
    root.mkdir(parents=True, exist_ok=True)
    if not idx.exists():
        idx.write_text(
            "# 观测台账\n\n"
            "| 时间 | tag | device | collection(s) | learning(s) | fps | CPU均值核 | CPU峰值C | GPU均值% | GPU峰值C | 备注 |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n"
        )
    mp = re.search(r"^\s*(Package id 0|Tctl)\s+\S+\s+\S+\s+(\S+)", t, re.M)
    ctemp = mp.group(2).rstrip("°") if mp else "-"
    meq = re.search(r"均值 ([\d.]+)", t)
    gmean = f"{sum(gc['util']) / len(gc['util']):.0f}" if gc["util"] else "-"
    gmax = f"{max(gc['temp']):.0f}" if gc["temp"] else "-"
    stamp = " ".join(d.name.split("_")[:2])
    with idx.open("a") as fh:
        fh.write(
            f"| {stamp} | {tag} | {cfg.get('device')} | "
            f"{c:.4f} | {lr:.4f} | {fps:.0f} | {meq.group(1) if meq else '-'} | {ctemp} | "
            f"{gmean} | {gmax} |  |\n"
        )
    print(f"追加台账 {idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
