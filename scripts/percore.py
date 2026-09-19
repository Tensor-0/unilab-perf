#!/usr/bin/env python3
"""每核利用率采样器 —— 区分 P 核 / E 核 / LP-E 核。

为什么需要它
------------
本机是 Intel Meteor Lake 混合架构（Core Ultra 9 185H）：
    6× P核  → 12 逻辑核 (cpu 0-11)   @ 4.8-5.1 GHz，有超线程
    8× E核  →  8 逻辑核 (cpu 12-19)  @ 3.8 GHz，无超线程
    2× LP-E →  2 逻辑核 (cpu 20-21)  @ 2.5 GHz，无超线程
Linux 只暴露逻辑核编号，不告诉你是哪一档（没有 sysfs 接口，Intel Thread Director
本体不可见）。而 MuJoCo 池会起 min(num_envs, 22) 个 worker 撒在全部逻辑核上 ——
所以"16 物理核到底用到了什么程度"必须自己分档统计才知道。

数据源：/proc/stat 的每核 jiffies（CLK_TCK=100 ⇒ 10 ms 分辨率）。
零权限、无依赖。

用法
----
    # 采样 30 秒，每 0.5 秒一次，同时记录某个进程的每线程 schedstat
    python3 percore.py --duration 30 --interval 0.5 --pid 12345 --out percore.txt

    # 只打印当前瞬间
    python3 percore.py --duration 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ---- 拓扑：从 sysfs 自己推，不硬编 ----
# 判定依据（本机验证过）：cpuinfo_max_freq + 是否超线程(core_id 重复)
# 本机结果：cpu 0-11 = P核(4800/5100 MHz, 有 SMT)；12-19 = E核(3800)；20-21 = LP-E(2500)
SYS_CPU = Path("/sys/devices/system/cpu")

CLK_TCK = os.sysconf("SC_CLK_TCK")


def read_topology() -> tuple[dict[int, str], dict[int, float]]:
    """返回 ({cpu: 类别}, {cpu: 当前频率 MHz})，类别从实读的拓扑推。"""
    n = os.cpu_count() or 0
    maxfreq: dict[int, float] = {}
    for c in range(n):
        try:
            maxfreq[c] = float((SYS_CPU / f"cpu{c}/cpufreq/cpuinfo_max_freq").read_text()) / 1000.0
        except Exception:
            maxfreq[c] = 0.0

    # 按 max_freq 分档；同档内再看 core_id 是否重复（超线程）区分 P / E
    core_id: dict[int, int] = {}
    for c in range(n):
        try:
            core_id[c] = int((SYS_CPU / f"cpu{c}/topology/core_id").read_text())
        except Exception:
            core_id[c] = c
    smt: dict[int, bool] = {}
    for c in range(n):
        try:
            sibs = (SYS_CPU / f"cpu{c}/topology/thread_siblings_list").read_text().strip()
            smt[c] = "," in sibs or "-" in sibs
        except Exception:
            smt[c] = False

    freqs = sorted({f for f in maxfreq.values() if f > 0}, reverse=True)
    # 最高档 = P；中间档 = E；最低档 = LP-E。带超线程的必定是 P。
    tier_of: dict[float, str] = {}
    if freqs:
        for i, f in enumerate(freqs):
            if i == 0:
                tier_of[f] = "P"
            elif i == len(freqs) - 1 and len(freqs) > 2:
                tier_of[f] = "LP-E"
            else:
                tier_of[f] = "E"
    cls: dict[int, str] = {}
    for c in range(n):
        t = tier_of.get(maxfreq[c], "?")
        if smt.get(c):
            t = "P"  # 超线程 ⇒ 一定是 P 核
        cls[c] = t
    return cls, maxfreq


def read_proc_stat() -> dict[int, tuple[int, int]]:
    """返回 {cpu: (busy_jiffies, total_jiffies)}。busy 含 user+nice+system+irq+softirq+steal。"""
    out: dict[int, tuple[int, int]] = {}
    with open("/proc/stat") as fh:
        for line in fh:
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            parts = line.split()
            try:
                idx = int(parts[0][3:])
            except ValueError:
                continue
            v = [int(x) for x in parts[1:]]
            while len(v) < 8:
                v.append(0)
            user, nice, system, idle, iowait, irq, softirq, steal = v[:8]
            busy = user + nice + system + irq + softirq + steal
            total = busy + idle + iowait
            out[idx] = (busy, total)
    return out


def read_cur_freq() -> dict[int, float]:
    out: dict[int, float] = {}
    for p in SYS_CPU.glob("cpu[0-9]*/cpufreq/scaling_cur_freq"):
        try:
            idx = int(p.parent.parent.name[3:])
            out[idx] = float(p.read_text()) / 1000.0
        except Exception:
            pass
    return out


def read_throttle_counts() -> dict[str, int]:
    """累积降频计数。⚠️ 必须前后取差才有意义 —— 它是自开机以来的累计值。"""
    out: dict[str, int] = {}
    for p in sorted(SYS_CPU.glob("cpu[0-9]*/thermal_throttle/core_throttle_count")):
        try:
            out[p.parent.parent.name] = int(p.read_text())
        except Exception:
            pass
    try:
        out["package"] = int((SYS_CPU / "cpu0/thermal_throttle/package_throttle_count").read_text())
    except Exception:
        pass
    return out


def read_thread_schedstat(pid: int) -> dict[int, tuple[int, int, int]]:
    """每线程 (on_cpu_ns, runqueue_wait_ns, timeslices)。零权限可读。"""
    out: dict[int, tuple[int, int, int]] = {}
    base = Path(f"/proc/{pid}/task")
    if not base.is_dir():
        return out
    for d in base.iterdir():
        try:
            f = d / "schedstat"
            if not f.exists():
                continue
            a, b, c = f.read_text().split()[:3]
            out[int(d.name)] = (int(a), int(b), int(c))
        except Exception:
            continue
    return out


# ---- 温度 ----
HWMON = Path("/sys/class/hwmon")


def find_cpu_hwmon() -> Path | None:
    """找 coretemp / k10temp 那类 CPU 温度传感器。"""
    if not HWMON.is_dir():
        return None
    for d in sorted(HWMON.iterdir()):
        try:
            name = (d / "name").read_text().strip()
        except Exception:
            continue
        if name in ("coretemp", "k10temp", "zenpower", "cpu_thermal"):
            return d
    return None


def read_cpu_temps() -> dict[str, float]:
    """{'package': °C, 'core0': °C, ...}。标签取 tempN_label，缺省用 tempN。"""
    hw = find_cpu_hwmon()
    if hw is None:
        return {}
    out: dict[str, float] = {}
    for p in sorted(hw.glob("temp*_input")):
        base = p.name[:-6]  # 去掉 _input
        try:
            v = int(p.read_text()) / 1000.0
        except Exception:
            continue
        lbl = None
        lp = hw / f"{base}_label"
        if lp.exists():
            try:
                lbl = lp.read_text().strip()
            except Exception:
                lbl = None
        out[lbl or base] = v
    return out


def read_thermal_zones() -> dict[str, float]:
    """ACPI thermal zones（备选，标签较粗）。"""
    out: dict[str, float] = {}
    for z in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            typ = (z / "type").read_text().strip()
            v = int((z / "temp").read_text()) / 1000.0
            out[f"{z.name}:{typ}"] = v
        except Exception:
            continue
    return out


def thread_name(tid: int) -> str:
    try:
        return Path(f"/proc/self/task").exists() and Path(f"/proc/{os.getpid()}/task/{tid}/comm").read_text().strip()
    except Exception:
        return "?"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--duration", type=float, default=5.0, help="采样总时长（秒）")
    ap.add_argument("--interval", type=float, default=0.5, help="采样间隔（秒）")
    ap.add_argument("--pid", type=int, default=None, help="同时采集该进程的每线程 schedstat")
    ap.add_argument("--out", default=None, help="输出文件（默认 stdout）")
    ap.add_argument("--timeseries", action="store_true",
                    help="额外输出每个采样点的「等价满载核数」时间序列（看振荡）")
    ap.add_argument("--tsv", default=None,
                    help="把每采样点写成 TSV（unixtime/等价核/温度/分档频率），"
                         "用于和 tensorboard 的 wall_time 对齐做相关性分析")
    args = ap.parse_args(argv)

    cls, maxfreq = read_topology()
    tiers: dict[str, list[int]] = {}
    for c, t in cls.items():
        tiers.setdefault(t, []).append(c)
    for t in tiers:
        tiers[t].sort()

    lines: list[str] = []
    def emit(s: str = "") -> None:
        lines.append(s)

    emit("=== 拓扑（自行从 sysfs 推出）===")
    for t in ("P", "E", "LP-E", "?"):
        if t not in tiers:
            continue
        cpus = tiers[t]
        mf = max((maxfreq.get(c, 0) for c in cpus), default=0)
        emit(f"  {t:5s} 逻辑核 {len(cpus):2d} 个: {cpus}   最高 {mf:.0f} MHz")
    emit()

    prev = read_proc_stat()
    prev_t = time.perf_counter()
    thr_before = read_throttle_counts()
    thr_prev: dict[int, tuple[int, int, int]] = {}
    if args.pid:
        thr_prev = read_thread_schedstat(args.pid)

    tier_busy_sum = {t: 0.0 for t in tiers}
    tier_tot_sum = {t: 0.0 for t in tiers}
    per_core_last: dict[int, float] = {}
    freq_acc: dict[int, list[float]] = {}
    n_samples = 0
    ts: list[float] = []          # 每个采样点的等价满载核数
    temp_hist: dict[str, list[float]] = {}
    tsv_rows: list[tuple] | None = [] if args.tsv else None

    t_end = prev_t + args.duration
    while time.perf_counter() < t_end:
        time.sleep(args.interval)
        now = read_proc_stat()
        now_t = time.perf_counter()
        dt = now_t - prev_t
        if dt <= 0:
            continue
        cur_f = read_cur_freq()
        sample_equiv = 0.0
        for c, (b, tt) in now.items():
            if c not in prev:
                continue
            db = b - prev[c][0]
            dtt = tt - prev[c][1]
            if dtt <= 0:
                continue
            frac = max(0.0, min(1.0, db / dtt))
            per_core_last[c] = frac
            sample_equiv += frac
            tier = cls.get(c, "?")
            if tier in tier_busy_sum:
                tier_busy_sum[tier] += db / CLK_TCK
                tier_tot_sum[tier] += dtt / CLK_TCK
            freq_acc.setdefault(c, []).append(cur_f.get(c, 0.0))
        ts.append(sample_equiv)
        # 温度：每采样点记一次（读取极轻）
        cur_t = read_cpu_temps()
        for k, v in cur_t.items():
            temp_hist.setdefault(k, []).append(v)
        # TSV 行：用于与 tensorboard wall_time 对齐
        if tsv_rows is not None:
            pkg = next((v for k, v in cur_t.items() if "ackage" in k or "Tctl" in k),
                       max(cur_t.values()) if cur_t else 0.0)
            mx = max(cur_t.values()) if cur_t else 0.0
            def tier_freq(t):
                cs = tiers.get(t) or []
                vs = [cur_f.get(c, 0.0) for c in cs if cur_f.get(c, 0.0) > 0]
                return sum(vs) / len(vs) if vs else 0.0
            tsv_rows.append((time.time(), sample_equiv, pkg, mx,
                             tier_freq("P"), tier_freq("E"), tier_freq("LP-E"),
                             per_core_last.get(0, 0.0)))
        prev, prev_t = now, now_t
        n_samples += 1

    emit(f"=== 每档利用率（{n_samples} 个采样点，{args.duration:.1f}s）===")
    emit(f"  {'档':6s} {'核数':>4s} {'平均利用率':>10s} {'等价满载核数':>12s}")
    total_equiv = 0.0
    for t in ("P", "E", "LP-E", "?"):
        if t not in tiers or tier_tot_sum.get(t, 0) <= 0:
            continue
        frac = tier_busy_sum[t] / tier_tot_sum[t]
        equiv = frac * len(tiers[t])
        total_equiv += equiv
        emit(f"  {t:6s} {len(tiers[t]):>4d} {frac*100:>9.1f}% {equiv:>12.2f}")
    emit(f"  {'合计':6s} {len(cls):>4d} {'':>10s} {total_equiv:>12.2f}  / {len(cls)} 逻辑核")
    emit()

    if args.timeseries and ts:
        emit("=== 时间序列：等价满载核数（看采集/学习是否振荡）===")
        emit(f"  {'采样#':>6s} {'等价核':>8s}   {'':<40s}")
        lo = min(ts)
        hi = max(ts)
        span = max(hi - lo, 1e-9)
        for i, v in enumerate(ts):
            bar = "#" * int((v - lo) / span * 40)
            emit(f"  {i:>6d} {v:>8.2f}   |{bar:<40s}|")
        emit(f"  最小 {lo:.2f} / 最大 {hi:.2f} / 均值 {sum(ts)/len(ts):.2f} / 极差 {hi-lo:.2f}")
        emit()

    emit("=== 每核明细 ===")
    for t in ("P", "E", "LP-E", "?"):
        if t not in tiers:
            continue
        for c in tiers[t]:
            f = per_core_last.get(c)
            if f is None:
                continue
            fa = freq_acc.get(c) or [0.0]
            emit(f"  cpu{c:<3d} [{t:4s}] 利用率 {f*100:5.1f}%   平均频率 {sum(fa)/len(fa):6.0f} MHz")
    emit()

    # ---- 温度 ----
    if temp_hist:
        emit("=== CPU 温度 ===")
        emit(f"  {'传感器':<24s} {'起始':>7s} {'均值':>7s} {'最高':>7s} {'末值':>7s}")
        pkg = None
        for k in sorted(temp_hist, key=lambda x: (0 if "ackage" in x or "Tctl" in x else 1, x)):
            h = temp_hist[k]
            if not h:
                continue
            if pkg is None and ("ackage" in k or "Tctl" in k):
                pkg = h
            emit(f"  {k:<24s} {h[0]:>6.1f}° {sum(h)/len(h):>6.1f}° {max(h):>6.1f}° {h[-1]:>6.1f}°")
            if len(h) > 1:
                emit(f"  {'':<24s} 温升 {h[-1]-h[0]:+.1f}°   （最高-最低 {max(h)-min(h):.1f}°）")
        emit()

    # ---- 降频计数（**取增量** —— 累积值本身没有意义）----
    thr_after = read_throttle_counts()
    emit("=== 降频计数（本采样窗口的【增量】）===")
    emit(f"  窗口时长 {args.duration:.1f}s")
    pkg_d = thr_after.get("package", 0) - thr_before.get("package", 0)
    emit(f"  package 增量 = {pkg_d}   （累积值 {thr_after.get('package', '?')}，仅供参考）")
    deltas = []
    for k in sorted(thr_after):
        if k == "package":
            continue
        d_ = thr_after[k] - thr_before.get(k, thr_after[k])
        if d_:
            deltas.append(f"{k}={d_}")
    emit(f"  有增量的核: {', '.join(deltas) if deltas else '**无**（窗口内未降频）'}")
    emit()

    if args.pid and thr_prev:
        thr_now = read_thread_schedstat(args.pid)
        emit(f"=== 进程 {args.pid} 的每线程 CPU（schedstat，纳秒）===")
        rows = []
        for tid, (a, b, c) in thr_now.items():
            pa, pb, pc = thr_prev.get(tid, (a, b, c))
            on_cpu = (a - pa) / 1e9
            wait = (b - pb) / 1e9
            rows.append((on_cpu, wait, tid))
        rows.sort(reverse=True)
        emit(f"  {'tid':>8s} {'on-CPU(s)':>10s} {'等调度(s)':>10s} {'CPU占比':>8s}")
        tot = sum(r[0] for r in rows) or 1.0
        for on_cpu, wait, tid in rows[:25]:
            emit(f"  {tid:>8d} {on_cpu:>10.3f} {wait:>10.3f} {on_cpu/tot*100:>7.1f}%")
        emit(f"  （共 {len(rows)} 个线程；CPU 总时间 {tot:.2f}s）")
        emit()

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"已写入 {args.out}", file=sys.stderr)
    else:
        print(text)

    if args.tsv and tsv_rows:
        hdr = "unixtime,equiv_cores,pkg_temp_c,max_core_temp_c,freq_p_mhz,freq_e_mhz,freq_lpe_mhz,util_cpu0"
        body = "\n".join(",".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r)
                         for r in tsv_rows)
        Path(args.tsv).write_text(hdr + "\n" + body + "\n")
        print(f"已写入 {args.tsv}（{len(tsv_rows)} 行）", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
