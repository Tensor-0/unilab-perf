# 一次训练把时间花在哪

**结论：物理仿真占一个控制步的 71.3%，已经跑到 17.96 核，受限于内存带宽 —— 加核、提频都没用。
剩下 28.7% 是三块串行代码，全消掉也就 1.4 倍。能显著改变局面的只有把物理搬走（mjwarp），预期 1.9 倍。
另外推翻了四个听起来合理但实测没用的优化方向。**

2026-09-19 测的。DM10（双足 10 自由度）/ UniLab + MuJoCo / PPO / 512 envs / 物理跑 CPU。
机器：ROG G16 GU605MV，Core Ultra 9 185H（16 物理核 22 逻辑核），RTX 4060 Laptop 8G。

---

## 时间预算

### 一个控制步（512 envs × 4 物理子步 = 15.10 ms）

用 `benchmark_env_step_phase_cpu.py` 测的，它对每个阶段同时记 wall 和进程 CPU 时间，
两者相除就是那一段的平均并行度：

```
phase              wall_ms   wall%   并行度
backend_step         10.76    71.3%  17.96 核
reset_done            2.10    13.9%   1.27 核
update_state          2.02    13.4%   1.12 核
other(step glue)      0.22     1.4%   0.31 核
TOTAL                15.10   100.0%  13.14 核
```

训练里读 tensorboard 是 `0.3764s / 24 = 15.68 ms`，对得上。

物理段 17.96 核。这台机器 16 个物理核，所以基本到顶了。

### update_state 内部（2.02 ms）

逐个 manager 打点：

```
reward.compute          1.1055 ms   57.7%
event.apply             0.3825      20.0%
command.compute         0.2199      11.5%
observation.compute     0.1788       9.3%
termination.compute     0.1681       8.8%
_map_observations       0.0032       0.2%
metrics / post_compute  ~0.002       0.1%
```

本来以为 obs 构建是大头，实际不是，reward 才是。

### reward 内部（1.106 ms，16 项）

```
feet_phase              0.1552 ms  13.6%
feet_slip               0.1379      12.1%
feet_contact_number     0.1212      10.7%
feet_phase_contrast     0.1123       9.9%
feet_air_time           0.0757       6.7%
feet_clearance          0.0748       6.6%
tracking_lin_vel        0.0719       6.3%
pose                    0.0519       4.6%
（其余 8 项都 < 0.03）
Top5 = 53.0%，分项和 84.0%
```

脚部相关的 6 项占了六成。它们都要读接触传感器（sensordata 968 B/env）。

### 迭代级

```
Perf/collection_time   0.3764 s   84.1%
Perf/learning_time     0.0710 s   15.9%
合计                   0.4474 s   fps 27851   3000 轮约 22 分钟
```

### 运行中

```
采集段（84% 时间）  CPU 约 18 核，93-100°C，GPU 空转（16-21%，22W，Idle 降频）
学习段（16% 时间）  CPU 约 0.8 核，GPU 忙（峰值 56%）
```

22 个物理 worker 各约 19.7s on-CPU / 1.4s 等调度，很均匀。
主线程 20.7s on-CPU / 7.3s 等调度——那 7.3s 是在 `WaitCount` 屏障上空等，占 31%。

---

## 试过但没用的四个方向

这些都是先有假设、再实测，结果全被推翻。写下来免得以后重复投入。

### 1. 把 MuJoCo 池钉到 P 核

假设：混合 CPU 上，22 个 worker 撒在 P/E/LP-E 三档核上，被最慢的拖后腿，钉到 P 核会更快。

实测（`benchmark_mujoco_pool_thread_scaling.py`，`--num-envs 512`）：

```
22t unpinned   19.68 ms
22t pinned     18.94 ms     <- 3.8%，重复 3 轮方向不一致，是噪声
12t unpinned   21.75 ms
12t pinned     31.32 ms     <- 比 unpinned 还慢 44%
 8t unpinned   28.39 ms
 6t pinned     60.51 ms
```

钉到 cpu 0-11 反而更慢，因为那 12 个逻辑核是 6 个 P 核的超线程对，互相抢。
22 线程 unpinned 已经是最优。

### 2. 「16 物理核是甜点」

Xeon 上的公开数据说物理吞吐在物理核数附近饱和。这台也是 16 物理核。

实测 14/16/18/20/22 五档，各重复 3 轮：

```
14t  19.33   16t  19.29   18t  19.30   20t  19.33   22t  19.43   (均值, ms)
```

极差 0.7%，而同一配置跨轮的波动是 4-13%。没有可分辨差异。

### 3. CPU 热降频拖慢了训练

现象是真的：训练时 CPU 封装 93-100°C，P 核顶到 100°C；45 秒窗口里
`package_throttle` 增量 8908，16 个核都在涨；P 核频率从冷机的 4554 MHz 掉到 4100-4200。

用固定随机动作的 benchmark 做冷热对比（这样负载行为恒定）：

```
            第1次    第2次    第3次
冷机 53°C   15.71    14.47    14.44   ms/step
热机 78-90°C 15.48   14.65    14.64   ms/step
```

没有可分辨差异。原因是 MuJoCo 物理受限于内存带宽，频率高低打不到痛点
（加核无效、提频无效，三个方向互相印证）。

测的温度范围是 53-90°C，没测到训练时的 99-100°C，但全程平坦，且训练实测值落在这个区间里。

### 4. nan_guard 的全状态拷贝

`np_env.py:251-261` 每步调一次 `get_physics_state_snapshot()` 再传给 `nan_guard.capture()`。
看起来像是每步一次全量拷贝的开销。

实测：

```
get_physics_state_snapshot() 单次 = 0.003 ms
```

算一下：state 是 512 × 34 float32 = 68 KiB，68 KiB / 3 µs ≈ 23 GB/s，正好是内存带宽量级。
state 太小了，不可能成为瓶颈。

12 轮交替 A/B 的代价是 +0.63%，而噪声底是 17-27%。没有信号。

> 顺带更正一处：我一度以为「配置里把 `nan_guard.enabled` 设成 false 也省不掉这个拷贝」。
> 读代码后是错的。`resolve_nan_guard_cfg`（`training/run.py:76-89`）在 `enabled=False` 时返回 `None`，
> `apply_env_nan_guard` 就直接不挂载，拷贝也就没有了。
> 当时混淆了两个同名的 `enabled`：配置里的控制挂载，`NanGuardCfg` 内部的只控制 isfinite 扫描。

---

## 三个测量上的坑

这一轮踩到的，会影响以后所有性能测量。

### 别用时序趋势推断硬件效应

`Perf/collection_time` 会随训练进行变化，但那主要是**策略行为在变**，不是硬件。
RL 训练里策略一直在更新，机器人姿态、接触状态就一直在变，物理代价跟着变。

实测：温度和采集耗时的相关系数只有 +0.163（温度只解释 2.7%），
而分段趋势显示首段到末段涨了 10.3%。

要测硬件，负载行为必须恒定。用固定随机动作的 benchmark，不能用训练 run 的时序。

### 累积计数器要取增量

`/sys/.../thermal_throttle/*_count` 是自开机累积的。
我第一次差点把 `package_throttle_count = 70162` 当成"这次降频了"的证据，那是没有意义的。
取增量（45 秒 +8908）才是证据。

### 噪声底

- 训练 `Perf/` 指标：同配置重跑约 1.2%
- env benchmark `ms/step`：单轮 4-13%，12 轮后组内极差 17-27%

小于噪声底的差异不下结论。上面第 4 条那个 +1.69% 的"效应"，样本量翻倍后缩到 +0.63%，
就是这么识别出来是噪声的。

---

## mjwarp 的预期收益

物理占一个控制步的 71.3%。如果 mjwarp 让物理快 3 倍：

```
现在            0.713 + 0.287 = 1.000
mjwarp 3x       0.713/3 + 0.287 = 0.525      加速 1.90x
物理无限快       0.287                        上限 3.48x
```

（早期估过 1.2-1.5 倍，那个数字用错了基数——当时拿的是一条 learner 跑在 CPU 上的 run，
采集占比算成了 49%。）

已知的坑没变：UniLab 官方只在 `g1_walk_flat` 上验证过 mjwarp；播放链路要改；
要装 extra 加一行注册；8G 卡上实际能开多少 env 未知。

---

## 没做的

- `reset_done`（13.9%）没拆开看
- `torch.compile` 没做 A/B
- mjwarp 没实测
- 并行跑多条 run 没试（本来以为 2026-09-18 那次崩溃说明并行不稳，后来查明是系统休眠杀的进程，和并行无关）

---

## 工具状态（当天实测）

能用的：

- `/proc/<pid>/task/<tid>/schedstat` — 纳秒级每线程 on-CPU / runqueue-wait。零权限，这轮最有用的东西
- `/proc/stat` — 每核 jiffies，10 ms 分辨率
- `/sys/class/hwmon/*/temp*_input` — CPU 每核+封装温度
- `nvidia-smi --query-gpu=...` — 含温度和降频原因
- `benchmark_env_step_phase_cpu.py` — 支持 dm10：`--config-group ppo --task dm10_joystick_flat/mujoco`
- `torch.profiler` CPU 侧（`kineto_available()=True`）

不能用的：

- `perf` — 零权限完全不可用。`kernel.perf_event_paranoid = 4`，比标准 3 还严，
  连软件事件 `duration_time,user_time` 都拒绝。要它必须 root
- `py-spy` / `nsys` / `ncu` / `bpftrace` — 没装
- `benchmark_env_overhead_parallel.py` — 合成形状（8192 envs / obs 98 / 29 奖励项），
  和 dm10（512 / 39 / 16）差太远，绝对数字不能搬
- `visualize_collection_profile.py` — 只认 mlx 的日志格式，本仓库没有产生这种日志的代码
