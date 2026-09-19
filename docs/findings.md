# 一次训练把时间花在哪

**结论：物理仿真占一个控制步的 71.3%，已经跑到 17.96 核，受限于内存带宽 —— 加核、提频都没用。
剩下 28.7% 是三块串行代码，全消掉也就 1.4 倍。
另外推翻了四个听起来合理但实测没用的优化方向。**

**找到两条真的：**

1. **把 MJCF 里的 solver 从 PGS 换成 Newton**，真实训练快 **1.64 倍**（纯物理计算量少 3.3 倍）。
   改动只有一个 XML 属性。
2. **`mjwarp`（GPU 物理）在 dm10 上不要用** —— 物理确实快 1.77 倍，
   但 reset 期的模型域随机化要调 `mujoco_warp.set_const`，每次固定 5.4 ms，全吃回去。
   根因是 399 次 kernel 发射，且**与 env 数无关**。
   已给 UniLab 补上 `startup` 模式把它降到 0（mjwarp 从 0.86× 变 **1.30×**），
   但 **3000 轮 A/B 里 startup 组的 reward 曲线落后 ~20%，一个 seed、未定论** ⇒
   **代码保留、默认值撤回**。详见 mjwarp 一节。

2026-09-19 / 09-20 测的。DM10（双足 10 自由度）/ UniLab + MuJoCo / PPO / 512 envs / 物理跑 CPU。
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

## mjwarp 实测：物理快 1.8 倍，被 reset 全吃掉

2026-09-20 测的。**当前配置下不要切 mjwarp。**

真实训练，同样的启动脚本（`env-steps/s`）：

| 配置 | CPU+Newton | mjwarp | 比值 | EpLen (CPU / mjwarp) |
|---|---|---|---|---|
| 512 envs, DR 全开 | **41,747** | 36,014 | 0.86× | 359 / 477 |
| 2048 envs, DR 全开 | 67,433 | 71,735 | 1.06× | 128 / 223 |
| 2048 envs, 模型 DR 全关 | 70,054 | **100,388** | **1.43×** | 570 / 430 |
| 2048 envs, 只留 foot_friction | 68,110 | **91,533** | **1.34×** | 429 / 608 |

固定负载 benchmark（随机动作，两边 reset 次数接近），@2048：

```
phase             CPU+Newton   mjwarp
backend_step          13.48      7.62      <- 物理确实快 1.77x
update_state           3.08      2.58
reset_done             2.44      9.59      <- 全吃回去
TOTAL                 19.31     20.02
```

mjwarp 自身扩展性没问题（DR 全开）：512 → 2048 → 4096 =
38,766 → 102,320 → 139,794 steps/s，8G 卡 4096 envs 也不 OOM。

### 根因

```
reset → _apply_reset_randomization → mujoco_warp.set_const(m, d)    ← 单次 5.4 ms
```

- **399 次 host 侧调用**（283 `wp.launch` + 100 `wp.launch_tiled` + 11 `wp.zeros` + …），
  host 派发占墙钟 92%
- **与 env 数无关**：nworld=64 和 nworld=512 都是 5.3 ms（命令条数不随规模变）
- 热点是 mujoco_warp 自己的两个 Python 循环 —— `io.py:3405` 的 dof 循环、
  `io.py:3425-3451` 的 body×row 嵌套循环（**上游自己挂着 `TODO(team)` 注释**）
- 200 步里 195 步有 reset ⇒ 这 5.4 ms 是**按「每步」交的**，不是按 reset 次数

哪个 DR term 触发它：

| DR term | 走哪 | 代价 |
|---|---|---|
| `base_mass` / `base_com` | `set_const` | 5.4 ms |
| `joint_armature` | `set_const_0` | 5.3 ms |
| **`foot_friction`** | 只 upload，不重算 | **免费** |

### 上游早就知道，但归因是错的

mjlab 文档（`randomization.html`）写明：

> All event manager logic, including `recompute_constants`, runs as regular Python
> between these graph replays, **so it will not break graph capture**.
> That said, **set_const is expensive** ... best randomized with **startup or reset** modes.

mjlab #757 讨论过同一现象，维护者的建议是改用 `startup` 模式。

**但文档对成本的归因和实测对不上** —— 它说贵是因为算得「across all worlds」，
实测是**固定成本、与 world 数无关**。这是值得反馈给上游的一个更正。

### 捕获成 CUDA graph：快 2.9x，且数值安全

mjlab 说「不捕获是有意设计」。实测「如果捕获会怎样」：

```
eager 逐 kernel 发射    5.73 ms
同一段 work 走 graph     2.00 ms   -> 2.87x
```

数值正确性，逐位比较：

| 测试 | 结果 |
|---|---|
| 自对照 eager vs eager | 0.00e+00 |
| 立即 replay | 0.00e+00 |
| 经 60 步 `env.step`（内存 churn）后 replay | 0.00e+00 |
| 同一输入连续 replay 两次 | 0.00e+00 |
| churn 后重新捕获再 replay | 0.00e+00 |
| **正对照：换一个扰动** | 2.04e-01（证明这个比较有分辨力）|

### ⚠️ 这条实验的第一版是错的

我拿「程序开头存的快照」当参照，跑出 `dof_invweight0` 差 17.6%、`body_invweight0` 差 8.3%。
差点当成「捕获期分配的临时数组被回收 ⇒ 指针失效」发出去。

**识别信号：测 2 和测 4 的差异数值一模一样**（3.801 / 2.215）。
内存被复用导致的失效不可能这么确定 —— 真凶是中间那 60 步 `env.step` 的
reset 期 DR 改掉了 `body_ipos` / `dof_armature`，而 `body_invweight0` 依赖前者、
`dof_invweight0` 依赖后者。等于在比两个不同的模型状态。

**修法：每个测试都用【当场的】eager 调用作参照，绝不用早先存的快照。**
改完之后全 0。

### 正解是 startup 模式 —— 已实现，但训练质量没验出来

mjlab 文档给的正解是 `startup` 模式：这些字段**初始化抽一次**，不每 episode 重抽。
UniLab 的 `EventMode` 里本来就有 `"startup"`，但模型类 DR term 把 mode **硬锁在 `reset`**：

```
NotImplementedError: EventManager term 'randomize_rigid_body_mass'
                      only supports mode='reset' on the UniLab runtime
```

放开它只需要改校验，但**真让它跑起来撞了两堵墙**：

1. 原地 `apply(mode="startup")` 写不进模型字段 —— 写入要求事务 active
   （`reset_state.py:1519`），报 `requires an active reset event`
2. 包了事务还不够：保持原顺序（startup → materialize）时 mujoco 直接崩
   `'NoneType' object has no attribute 'reset'` —— `materialize()` 才建线程池。
   ⇒ 时序契约必须从 `startup → materialize` 反过来

#### 已验证的（这部分可信）

| 项 | 改前 | 改后 |
|---|---|---|
| 每个 env 独立随机值 | — | ✅ env0=8.225 / env1=5.019 |
| `set_const` 调用 | 195 / 200 步 | **0**（只在 init 一次） |
| `reset_done` @2048 | 9.59 ms | **3.54 ms** |
| benchmark @2048 | 102,320 | **148,608** steps/s |
| 真实训练 @2048（60 轮） | CPU 68,364 | mjwarp **88,770 = 1.30×** |
| mujoco 兼容 | — | reset 档 136 次写/218 reset 不变 |
| 回归 | — | 无（基线 24 失败 vs 改后 24 失败，集合完全相同） |

#### ⚠️ 没验证出来的：3000 轮 A/B

A = startup / B = reset，mjwarp @512，各 3000 轮：

| 指标 | A | B | A/B |
|---|---|---|---|
| wall（A 快 1.31×） | 1100 s | 1441 s | 0.763 |
| `best_mean_reward` | 93.12 | 94.92 | **0.981** |
| `Train/mean_reward` max | 73.56 | 86.57 | **0.850** |
| 末值 | 59.40 | 70.17 | **0.847** |
| 全程均值 | 35.94 | 44.41 | **0.809** |

十分位曲线，**10 个十分位里 9 个落后**（只有开头 0-300 轮 startup 领先）：

```
   0- 300   11.18    9.26   1.207   ← 只有开头领先
 300- 600   36.48   37.46   0.974
 600- 900   37.09   44.82   0.828   ← 从这里开始
 900-1200   37.66   50.16   0.751
1200-1500   38.47   51.36   0.749
1500-1800   39.56   51.19   0.773
1800-2100   37.75   51.99   0.726
2100-2400   39.10   49.61   0.788
2400-2700   41.77   45.73   0.913
2700-3000   40.31   52.51   0.768
```

**但这不足以定罪**：n=1/arm，两臂 RNG 流不同源（DR 抽样时机不同 ⇒ 从早期就发散），
**不是配对比较**；RL 跨 seed 方差通常远大于此。

**一个指向"测试设计"的机制解释**：reset 模式下同一 env 每 episode 换一副身体，
等价于**持续的数据增强**；startup 下 512 envs 全程只有 512 套动力学。
策略观测里没有质量/质心，必须对全部动力学鲁棒 ⇒ 池子小了就是更难。
**mjlab 建议的前提正是 4k+ envs —— 选 512 是想保守，但保守的方向恰好可能制造了人造劣势。**

#### ⚠️ 顺带一个判据自坑

我事前定的判据是「`best_mean_reward` ≥ 90%」，结果 0.981，判"通过"。
然后才发现**同一次训练有四个口径，三个说失败**，而我挑中了唯一说通过的那个
（`best_*` 是最平滑、最容易被"曾经有过一个好窗口"拉平的）。
曲线一看就明白不是噪声形状。

⇒ **判据要用 `Train/mean_reward` 的逐迭代原始序列（十分位/滑窗），
不要用 `run_summary.json` 里任何 `best_*`。**

#### 当前状态

**代码能力保留，yaml 默认值撤回**（`dm10_joystick_flat/mjwarp.yaml` 恢复 `reset`）。
下一步是先在 2048 envs 上跑 A/B 验证「512 太小」这个假设 —— 它决定了是继续还是废弃。

---

## 真正有用的那条：换求解器

上面说完"物理到顶了、加核提频都没用"，最后发现 **dm10 的 MJCF 把求解器锁死在 PGS 上**。

`src/unilab/assets/robots/dm10/dm10.xml:8`：

```xml
<option timestep="0.001" iterations="50" solver="PGS" gravity="0 0 -9.81"/>
```

MuJoCo 3.11 的默认求解器已经是 NEWTON，但这个 XML 显式写回 PGS。改成 `solver="Newton"`
（注意大小写，写成 `NEWTON` 会 XML 解析失败）之后：

```
                PGS        Newton
backend_step     10.76 ms    3.51 ms     3.1x
  cpu_ms        193.33       58.50       3.3x
reset_done        2.10 ms    1.35 ms
update_state      2.02 ms    1.50 ms
TOTAL step       15.10 ms    6.51 ms     2.3x
```

真实训练（250 iter，同配置）：

```
采集     0.4139 s  ->  0.2279 s    1.82x
合计     0.4813 s  ->  0.2943 s    1.64x
fps      25,742    ->  42,125
3000 轮  22.4 min  ->  14.7 min
```

（合计倍数比 env 步进低，因为 0.067 s 的学习时间不变，被摊薄了。）

### 等价性

同模型两份 XML（只差 solver），喂相同动作，逐步比 qpos：

| 场景 | 最大偏差 |
|---|---|
| 小动作 ±0.3 | 0.00° |
| 大动作 ±1.5 | 0.00° |
| 大动作 + 60N 横向推力（摔倒、14 个接触） | 0.00° |
| 200-250 控制步全程 | 0.00° |

对照组证明这个比较有分辨力：PGS 用 `iterations=1`（故意不收敛）对比 Newton，最大偏差 13.45°。

⇒ 短程内 PGS(50 迭代) 和 Newton 收敛到同一个解。**但这是短程结论，闭环策略下不能外推。**

长程验证（播放 1000 步量步态指标）：

| | Newton | PGS |
|---|---|---|
| 抬脚高度 | 6.1 / 6.6 cm | 5.3 / 7.1 cm |
| 腾空占比 | 47.4 / 45.6% | 47.9 / 53.2% |
| 接触期滑移 | 0.685 / 0.659 m/s | 0.607 / 0.654 m/s |
| 步频 | 3.65 / 3.75 Hz | 3.90 / 3.95 Hz |
| 速度 | 0.646 m/s | 0.634 m/s |

两种求解器在噪声范围内一致。换求解器没把策略弄坏。

### 一个踩到的坑

我一度判"Newton 下策略不能走"，依据是拿 9-18 报告里记的 `滑移 0.154 / 0.387`
当基准，对比 Newton 实测的 `0.685 / 0.659`，看起来差 2-4 倍。

错的。用同一个脚本、同样的 1000 步在 PGS 下复测得到 `0.607 / 0.654`，跟 Newton 差不多。
那组 `0.154/0.387` 是另一次播放产物测的，当前脚本复现不出来。

拿历史记录当基准判好坏之前，必须先用同一方法复现那个基准。

顺带：两种求解器下滑移都是 0.6-0.7 m/s，比前进速度 0.63 m/s 还大。这个策略本来就在蹭地，
不是求解器造成的，也一直没被可靠测过。

### 副作用

mjwarp 只支持 CG 和 NEWTON（`mujoco_warp/_src/types.py:496-498` 写着 `unsupported: PGS`），
PGS 模型直接抛 `NotImplementedError: mjSOL_PGS is unsupported`。
换 Newton 顺路把这个障碍清掉了。

## 没做的

- `torch.compile` 没做 A/B
- 并行跑多条 run 没试（本来以为 2026-09-18 那次崩溃说明并行不稳，后来查明是系统休眠杀的进程，和并行无关）
- **2048 envs 上的 startup A/B** —— 验证「512 envs 太小」这个假设，它决定 startup 模式是继续还是废弃
- startup 模式的种子扫描（若 2048 上差距仍在）
- mjwarp 上的长训练（>250 轮）没跑，没看最终步态质量
- `set_const` 那两个嵌套循环的修法（上游 TODO）没验证能省多少

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
