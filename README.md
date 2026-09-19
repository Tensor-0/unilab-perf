# unilab-perf

给 UniLab 训练做硬件画像的一套脚本。跑一轮训练，同时采 CPU 分档利用率、
每线程调度、温度、降频、GPU 状态，输出一份可 diff 的报告。

在 DM10（10 自由度双足，MuJoCo 物理跑 CPU，PPO）上开发和验证。
测出来的结论在 [docs/findings.md](docs/findings.md)。

## 依赖

- Linux，`/proc` 和 `/sys` 可读（不需要 root）
- `nvidia-smi`（只有 NVIDIA 卡才需要）
- 被观测的 UniLab 检出目录，用 `UNILAB_ROOT` 指定，默认 `$HOME/UniLab`
- `uv`（训练本身通过 `uv run train` 启动）

`percore.py` 只用标准库。其余脚本通过 `uv run --project $UNILAB_ROOT` 跑，
所以能 import 到 UniLab 的依赖（tensorboard 等）。

## 用法

```bash
# 跑一轮 250 iter 的训练 + 采样，打上 tag
./observe.sh --tag baseline

# 改训练参数
./observe.sh --tag compile-on --train-args "algo.algorithm.enable_compile=true"

# 比两次观测
./observe.sh --diff baseline compile-on
```

输出到 `$REPORTS_ROOT`（默认 `$HOME/reports/dm10-observe`）：

```
<时间戳>_<tag>/
  summary.md    分相预算、每档核利用率、温度、降频增量、每线程 CPU、GPU
  phases.json   机读，给 --diff 用
  percore.txt   原始采样
  gpu.csv       GPU 采样
  train.log     训练日志
```

## 脚本

| 文件 | 干什么 |
|---|---|
| `observe.sh` | 入口。起训练、并行采样、收尾出报告 |
| `train_safe.sh` | 校验 CUDA 可见 + 训练期间禁止系统休眠，然后 `uv run train` |
| `scripts/percore.py` | 采样器。从 sysfs 推 P/E/LP-E 分档，读 `/proc/stat` 和每线程 `schedstat`、温度、降频增量 |
| `scripts/collect_perf.py` | 从 run 目录抽 Perf 指标，输出固定 schema 的 JSON |
| `scripts/render_summary.py` | 渲染 summary.md 并追加台账 |
| `scripts/nan_guard_ab.py` | nan_guard 开/关的 A/B，固定随机动作 |
| `scripts/update_state_breakdown.py` | 把 `update_state` 拆到各个 manager |
| `scripts/reward_breakdown.py` | 把 `reward.compute` 拆到各个奖励项 |

## 为什么要 train_safe.sh

2026-09-18 踩过一次：系统进 s2idle 休眠，把训练进程杀了；唤醒后 NVIDIA 报 Xid 31，
`torch.cuda.is_available()` 返回 False。而 UniLab 在 `training.device` 没写死时靠运行时探测，
探测失败就静默走 CPU——整场 3000 轮慢 2.3 倍，零报错。

`train_safe.sh` 干两件事：启动前确认 CUDA 可见（不可见就退出），
以及用 `systemd-inhibit` 挡住训练期间的休眠。

如果你想彻底关掉这个静默降级，在 task 的 owner yaml 里写死 `training.device: cuda`
——这样 CUDA 不可用时会抛异常而不是默默跑 CPU。

## 测量上的注意事项

**别用时序趋势推断硬件效应。** `collection_time` 会随训练变化，但那主要是策略行为在变，
不是硬件。实测温度和采集耗时的相关系数只有 0.163（温度解释 2.7%），
而分段趋势显示涨了 10.3%。要测硬件，负载行为必须恒定——用固定动作的 benchmark。

**累积计数器要取增量。** `thermal_throttle/*_count` 是自开机累积的，
直接读没有意义，必须前后取差。

**先量噪声底。** 这个 harness 的训练指标噪声约 1.2%，env benchmark 单轮 4-13%。
小于噪声底的差异不下结论。

## 数据

`data/2026-09-19/` 是跑出 [findings.md](docs/findings.md) 那批结论的原始数据。
路径里的 `$UNILAB_ROOT` 是打码后的，原值是本机检出目录。

跑的时候机器状态：`platform_profile=performance`，22 核 EPP 全 `performance`，
`intel_pstate/no_turbo=0`，`max_perf_pct=100`，GPU power limit 55W（上限 115W）。
