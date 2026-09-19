# unilab-perf

跑一轮 UniLab 训练，同时采硬件数据，出一份可 diff 的报告。
只读 `/proc` 和 `/sys`，不需要 root，不改被观测的代码。

**它回答**：训练时那 22 个核在干什么、GPU 为什么闲着、时间花在哪一段。

**已经用它得到的结论**（DM10 双足 / MuJoCo / PPO / 512 envs）：

- 物理仿真占一个控制步 **71.3%**，已跑 17.96 核，受限于内存带宽 —— 加核、提频都没用
- 串行段 28.7%（`reset_done` 13.9% / `update_state` 13.4% / glue 1.4%）
- 推翻了四个听起来合理的方向：钉 P 核、16 物理核甜点、热降频、`nan_guard` 拷贝
- **换求解器** `PGS → Newton`，真实训练快 **1.64 倍**，改一个 XML 属性
- **`mjwarp` 不要用**：物理快 1.77 倍，但 reset 期模型 DR 调的
  `mujoco_warp.set_const` 每次固定 5.4 ms，全吃回去（根因是 399 次 kernel 发射，
  **与 env 数无关**）
- 给 UniLab 补了 `startup` 模式把上面那 5.4 ms 降到 0，mjwarp 从 0.86× 变 **1.30×**；
  但 3000 轮 A/B 里 startup 组 reward 落后 ~20%（一个 seed，未定论）
  ⇒ **代码保留、默认值撤回**
- **升级 `unisim-core` 1.1.4 → 1.7.2：+14.2%**（env 步进）。
  纯粹是版本落后 6 个小版本，没有一行代码优化。但代价是一次**原生层迁移**
  （批处理引擎换成 `mjbatch-uni`、4 处 API 断裂），不是改版本号
- 三个方向**动手前先查了社区，结果是别做**：多 run 并发、APPO overlap（天花板 1.19×）、
  `torch.compile`（rsl_rl 原话「MLP-only policies see no benefit」）

细节和全部数据在 [docs/findings.md](docs/findings.md)。

## 用

```bash
export UNILAB_ROOT=$HOME/UniLab
./observe.sh --tag baseline
```

跑 250 iter 训练，采 45 秒，出到 `$HOME/reports/dm10-observe/<时间戳>_baseline/`。

改训练参数、比两次：

```bash
./observe.sh --tag compile-on --train-args "algo.algorithm.enable_compile=true"
./observe.sh --diff baseline compile-on
```

## 出什么

```
<时间戳>_<tag>/
  summary.md    分相预算 · 每档核利用率 · 温度 · 降频增量 · 每线程 CPU · GPU
  phases.json   机读，给 --diff 用
  percore.txt   原始采样
  gpu.csv       GPU 采样
  train.log     训练日志
```

`phases.json` 是固定 schema，所以不同配置、不同时间的观测能直接 diff。

## 脚本

| 文件 | 干什么 |
|---|---|
| `observe.sh` | 入口。起训练、并行采样、收尾出报告 |
| `train_safe.sh` | 校验 CUDA 可见 + 训练期间禁止系统休眠，然后 `uv run train` |
| `scripts/percore.py` | 采样器。sysfs 推 P/E/LP-E 分档，读 `/proc/stat`、每线程 `schedstat`、温度、降频增量 |
| `scripts/collect_perf.py` | 从 run 目录抽 Perf 指标 → 固定 schema JSON |
| `scripts/render_summary.py` | 渲染 summary.md，追加台账 |
| `scripts/nan_guard_ab.py` | nan_guard 开关 A/B（固定动作，排除策略行为干扰） |
| `scripts/update_state_breakdown.py` | 把 `update_state` 拆到各 manager |
| `scripts/reward_breakdown.py` | 把 `reward.compute` 拆到各奖励项 |
| `scripts/reset_breakdown.py` | 拆 `reset_done`（含「固定开销 + 每 env 边际」拟合） |
| `scripts/mjwarp_reset_probe.py` | 运行时 monkeypatch 给 mjwarp 后端打点，**不改任何文件** |
| `scripts/mjwarp_setconst_issue_evidence.py` | `set_const` 的 399 次发射拆解 + graph 对照（上游 issue 素材） |
| `scripts/mjwarp_setconst_capture_correctness.py` | 验证捕获成 CUDA graph 后数值是否还正确（含正对照） |
| `scripts/warp_launch_overhead_control.py` | 对照：零成本 kernel 的纯发射开销（8.4 µs/次） |
| `scripts/compare_train_curves.py` | 比两次训练的 reward 曲线，出十分位表（**别用 `run_summary.json` 的 `best_*` 判质量**） |

`percore.py` 只用标准库。其余通过 `uv run --project $UNILAB_ROOT` 跑，所以能 import 到 UniLab 的依赖。

## 为什么有 train_safe.sh

2026-09-18 踩过一次：系统进 s2idle 休眠，杀掉训练进程；唤醒后 NVIDIA 报 Xid 31，
`torch.cuda.is_available()` 返回 False。而 UniLab 在 `training.device` 没写死时靠运行时探测，
探测失败就静默走 CPU —— 整场 3000 轮慢 2.3 倍，零报错。

`train_safe.sh` 启动前确认 CUDA 可见（不可见就退出），并用 `systemd-inhibit` 挡住休眠。

想彻底关掉这个静默降级，在 task 的 owner yaml 里写死 `training.device: cuda`
—— 这样 CUDA 不可用时抛异常，而不是默默跑 CPU。

## 测量时注意

这三条是踩出来的，不遵守会得到假结论。

**别用时序趋势推断硬件效应。** `collection_time` 随训练变化，但主要是策略行为在变，不是硬件。
实测温度和采集耗时的相关系数只有 0.163（温度解释 2.7%），而分段趋势显示涨了 10.3%。
要测硬件，负载行为必须恒定 —— 用固定动作的 benchmark。

**累积计数器要取增量。** `thermal_throttle/*_count` 是自开机累积的，直接读没意义，必须前后取差。

**先量噪声底。** 训练指标噪声约 1.2%，env benchmark 单轮 4-13%。小于噪声底的差异不下结论。

**参照必须是当场的，不能是早先存的快照。** 测 mjwarp 的 graph 捕获时，
我拿程序开头的快照当参照，跑出 17.6% 的「差异」，差点当成 bug 报上去。
真凶是中间那 60 步 `env.step` 的 reset 期 DR 改掉了 `body_ipos` / `dof_armature`，
而 `body_invweight0` / `dof_invweight0` 依赖它们 —— 等于在比两个不同的模型状态。

识别信号很明确：**两个独立测试给出了完全相同的差异数值**（3.801 / 2.215）。
内存失效不可能是这么确定的。改成「每个测试都用当场的调用作参照」后全 0。

## 依赖

- Linux，`/proc` 和 `/sys` 可读
- `nvidia-smi`（非 NVIDIA 卡可跳过）
- `uv`
- 一个 UniLab 检出目录，用 `UNILAB_ROOT` 指定，默认 `$HOME/UniLab`

## data/

`data/2026-09-19/` 是得出上面那些结论的原始数据，6 次观测 + 1 次相关性实验。
里面的 `$UNILAB_ROOT` 是打码后的（原值是本机检出目录）。

当时机器状态：`platform_profile=performance`，22 核 EPP 全 `performance`，
`no_turbo=0`，`max_perf_pct=100`，GPU power limit 55W（上限 115W）。
