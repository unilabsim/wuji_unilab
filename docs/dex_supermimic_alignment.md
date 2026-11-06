# Hand 1 / Pen Spin 迁移对照

算法依据 `references/Dex-Supermimic/src/dex_supermimic/rl/dex_supermimic`，
UniLab 生命周期参考 `references/Uni-SuperMimic` 和 `references/wuji_unilab`。
新任务使用独立的 `rl/algos/dex_supermimic`，不修改完成的 Hand2 Cube learner、
任务文件或已有 checkpoint。

2026-11-05 的 Hand1 复核发现了求解器、读阶段姿态缓存、腕部 reset 和课程状态
方面的实际差异；Pen 复核进一步发现真实观测噪声缩放、reset 参考帧、play 配置及
笔接触默认值的差异，均已按源行为修复。此前通过维度和短测得出的“完全对齐”判断
过强；核心公式相同不能证明依赖中的处理顺序或环境生命周期等价。具体问题、验证
结果及旧 checkpoint 的适用范围见下文两项 2026-11-05 复核记录。

## 训练契约

| 项目 | Hand 1 Cube | Hand 2 Pen Spin |
|---|---:|---:|
| 状态 | 58 | 58 |
| 仿真参数 context | 46 | 46 |
| Actor 输入 | 207 | 155 |
| Critic 输入 | 211 | 301 |
| 世界模型输入（含已执行关节目标） | 133 | 298 |
| 世界模型接触输出 | 23 | 15 |
| 世界模型总输出（52 维状态增量 + 接触 logits） | 75 | 67 |

已核对的网络结构、归一化、状态编码、世界模型监督、可微想象、lambda-return
和失败头公式按源实现移植；这项代码比对不覆盖所有后端执行或 reset 时序。
`unisupermimic-hand1-train`、`unisupermimic-pen-spin-train` 使用
SuperMimic；名称带 `ppo` 的旧入口是独立基线。Pen 使用 hard joint limits，
Hand1 使用源配置的 0.9 soft limits，二者不共用错误的归一化表。

Hand1 的固定物理随机化在 startup 发布，局部 reset 保留同一组参数；几何尺寸改变时
同步重建碰撞包围盒。关节归一化保留源式的 `half + 1e-6`。源版本的 XML actuator
没有可选的 `actuator_effort_limit` 字段，因此力矩奖励直接使用 Nm 的平方，不用关节
force clamp 另做归一化。

## Pen 环境对照

| 源任务行为 | UniLab 实现 |
|---|---|
| adaptive RSI、分箱、失败/存活统计、EMA、平滑核、uniform mix | `tasks/pen_spin/commands.py` |
| reset 将采样帧 k 写入物理状态，reset 观测使用 k+1 参考；局部 reset 不影响存活行 | `events.py` 的 `prepare_reset` 与 command 的选定行 `refresh_reset` |
| 参考加残差、EMA、encoder bias、短暂冻结关节目标 | `actions.py` |
| 相机偏置、噪声、历史延迟、丢帧与恢复 | `observations.py` 的原生相机 term |
| 姿态/关节/链接跟踪、动作平滑、力矩、持续过流、自碰撞八项奖励 | `rewards.py`；读取真实传感器与当前力矩上限 |
| 跟踪误差、动作/关节平滑度、饱和比例 | `metrics.py`；加速度读取实际 qacc |
| 启动物理 DR、pen 长度/直径及一致惯量、每世界 PD/effort/encoder 等 | `randomization.py`、源 `inertia.py` |
| 物体衰减脉冲、五指独立时钟和随机施力点 | `disturbances.py`；合并一次后端 wrench transaction |
| 轴向滑动、绕笔轴旋转的 reset noise；40k–240k 步课程 | `events.py`、`curriculums.py` |

source 的 `pseudo_inertia` 从默认惯性字段重建，因此会覆盖此前 COM event 的
`body_ipos` 写入。这里保留锁定版本的实际执行语义；没有按注释另造一个不同的
COM 合成规则。不同框架的 RNG 消费顺序和接触求解后端不保证逐位相同。

## 适配边界

`reset_bridge.py` 集中承载 UniLab 当前未公开的模型字段和状态读取：
`geom_pos`、`actuator_forcerange`、`body_iquat`、qacc/失稳检查。
使用实际编译模型绑定列索引，避免 MuJoCo 移除可视几何后引用错误的 geometry ID。
MJWarp 额外字段只在 startup 扩展一次，发布后重建 CUDA graphs；训练期间不替换
模型数组。物体和指尖的 wrench 一次合并提交，以保留两者的真实合力。

静态关节索引始终使用 CPU 整数序列；qpos/qvel/动作继续使用训练设备的 Tensor。
checkpoint 恢复通过 UniLab 的公开 training-state 接口恢复权威 step counter，
避免下一步覆盖已恢复的课程进度。

## Pen 长训遥测

新 learner 将 UniLab 在 reset 发布的 `Episode_Metrics` 原样转发到 CSV 和
TensorBoard 的 `task/episode_*` 标签：位置误差 `object_pos_tracking_error_cm`、
笔轴夹角 `object_ori_tracking_error_deg`、动作一阶与二阶差分 `action_rate_sq` /
`action_acc_sq`、关节速度与加速度 RMS `joint_vel_rms_rad_s` /
`joint_acc_rms_rad_s2`、力矩饱和占比 `actuator_saturation_share`。
这些是每个已完成 episode 的原生时间平均，再按完成 episode 数量等权汇总到本次
collect；没有 episode 结束时留空，不填伪零，也不在 reset 后重新求值。
续训旧 CSV 会保留历史行和原列，并补充新增指标列。

`task/drop_rate` 是每控制步的真实物理终止比例，不是 episode 成功率；
Pen 的共享 `task/rot_error` 把物体旋转与 `memory.goal` 的恒等矩阵比较，并未读取
当前轨迹参考；它不是 Pen 参考姿态或笔轴跟踪误差，不能用于评价转笔精度。
判断转笔效果应结合以度为单位的原生笔轴夹角、位置误差和确定性回放；
`wm/survival` 是模型想象中的存活率，不能替代真实连续转笔时长。

## Pen 本地回放

回放依据参考仓库 `rl/dex_supermimic/cli.py:play` 和
`tasks/pen_spin/config/wuji_hand2/env_cfgs.py` 的最终 `play=True` 配置，
使用单环境、完整 clip 的 uniform 采样、首帧保持 1 秒及干净的真实笔轴端点观测。
固定笔直径 14.9mm、长度 250mm、质量 30g，沿用源几何/惯量计算。
只加载 Actor 与保存的归一化，直接输出动作，不加训练探索噪声；不实例化 Runner，
不更新 checkpoint、训练统计或训练输出。可选 recipe 路径与源 CLI 一样允许播放其他动作。

源 play 的位置终止阈值是 20cm，且移除方向提前终止；因此它适合观察完整动作，
不能把 play 的完成次数当作训练阈值下的成功率。源 reward/curriculum helper 在
`play` 分支提前返回，不挂载训练课程、reset 噪声、动作冻结或训练指标；迁移回放
现已显式移除这些继承项。此前仅看赋值位置而忽略提前返回的解释有误；旧回放前
40000 步的课程恰为零，掩盖了错误继承，并不能证明配置相同。

UniLab 不支持 command 自带的 `debug_vis` 回调，改由独立 viewer 绘制参考手/笔和
参考链接点。源实现使用绿色参考模型和三组坐标轴；当前按用户偏好将参考模型改为
浅粉色，并隐藏坐标轴。视觉模型按名称映射实际物理字段，避免执行器剥离
visual geoms 后索引错位。该模型与物理状态独立，绘制不改变训练或仿真数据。

```bash
MUJOCO_GL=glfw XDG_CACHE_HOME=/tmp/unisupermimic-cache \
uv run unisupermimic-pen-spin-play --checkpoint m_ends/model_8000.pt
```

`R` 重置、空格暂停/继续、`N` 单步，`--manual-reset` 在终止后等待 `R`。
默认物理后端为 MJWarp；`--backend mujoco` 使用 CPU。`--render none --steps 150`
可在不打开窗口的情况下检查加载和策略执行，不会生成视频。

## Hand1 本地回放

`unisupermimic-hand1-play` 使用同一套源 Actor 和保存的归一化，校验 Hand1
任务、format 3、207 维输入、控制频率和关节顺序；不创建 Runner，不更新统计。
环境使用 `factory.make_hand1_env(play=True)`：干净观测、无 startup DR 或 interval
扰动、无课程、60 秒 episode，保留源 reset 初态随机化。CPU 与 MJWarp 均可选择。
独立视觉模型从实际物理 snapshot 同步 qpos/qvel 和 mocap 腕部姿态，避免 reset
腕部随机旋转后画面与物理状态分离。host snapshot 不带 mocap 且公开 mocap getter
不支持时，使用 read plan 中实际掌心的世界姿态（Hand1 掌心即唯一 mocap body）。
六面目标和状态灯复用已有 Hand2 可视化，
只绘制到 viewer 的 user scene，隐藏坐标轴，不修改 Hand2 代码。

```bash
MUJOCO_GL=glfw XDG_CACHE_HOME=/tmp/unisupermimic-cache \
uv run unisupermimic-hand1-play \
  --checkpoint m_ends/hand1_reorient/model_final.pt
```

按 `R` 重置、空格暂停/继续、`N` 单步；`--manual-reset` 在 episode 结束后保留
终止状态等待 `R`。加 `--backend mujoco --render none --steps 150` 可无窗口验证
checkpoint 加载与执行。首次实现入口时，实际 `model_final.pt`（8000 轮）完成
150 步 CPU 回放，没有失败终止并达到 1 次目标保持。这是本轮环境修复之前的短测，
不能作为完整策略评估或训练与回放已等价的证据；修复后的检查见最新复核记录。
新增 11 项 checkpoint/策略/真实 CPU 状态同步/窗口按键测试及 4 项目标绘制测试，
均通过；现有 Hand1 的 12 项测试也通过。窗口按键测试使用真实物理环境和窗口替身，
本次没有打开实际 OpenGL 窗口或运行 GPU 回放。

## 验证边界

以下早期训练和 GPU 验证记录发生在本轮 Hand1/Pen 环境修复之前，证明对应版本的
链路可以执行，不能证明已覆盖本轮发现的差异。修复后的 CPU/GPU 结果单独记录在
最新复核中。

已有测试直接加载参考仓库的源函数，比较 sampler、camera/action、奖励与扰动数学；
另有真实 MuJoCo 多环境测试验证 startup 参数确实进入 executor、非零课程下的
force/torque、局部 reset、参考速度、相机记忆与奖励采集。
Pen 已使用真实 multi-axis catalog、原始 `[2048,2048,2048]` 世界模型在 CPU 和
RTX 5070 Ti 的 MJWarp 上分别完成 20 轮训练，包含第 5 轮后的 Actor/Critic 更新；
GPU checkpoint 已从第 20 轮恢复训练到 22 轮。GPU 下真实奖励与独立可微奖励的
平均绝对差约 `1e-8`；CPU 的体姿态传感器相位差使其约为 `1e-3`。
默认 `torch.compile` 与 `compile_policy` 路径也已完成 6 轮 GPU 验证，覆盖首次
策略和 Critic 更新。另有两项真实 CUDA 集成测试验证完整 startup 参数、CUDA graph、
非零课程 force/torque、设备传感器与局部 reset 隔离。

Hand1 也已使用原始世界模型规模在真实 MJWarp GPU 上完成 20 轮、16 环境短训练，
包含 Actor/Critic 更新，累计 1,056 个仿真步；原生奖励与独立可微奖励的误差约为
`1e-8`。CPU 测试另核对 startup 随机化保持、碰撞包围盒、encoder 校正、输入维度、
实际力矩奖励与正运动学梯度。

Warp CPU 数组测试另覆盖字段类型、维度、扩展、镜像和 graph 重建调用契约。
这些验证确认了训练链路和接口行为；长期收敛仍需要正式训练评估。

正式 Pen 长训保留源网络、1024 个环境、1000 步 replay、WM batch 2048 和
32 步想象；在 RTX 5070 Ti 上仅将 `pi_batch` 调整为 512。完整配置已在临时目录
验证策略更新、更新后的真实仿真采样和再次更新，未发生显存不足。正式训练从头开始，
不延续小环境 smoke 的训练统计，8000 轮结果写入 `m_ends/`，短训归档到
`m_ends/smoke_20/`。启动参数与日志分别保存在 `long_train_command.sh` 和
`long_train.log`，每 100 轮保存一次 checkpoint。

源配置的外力课程在 40000 个公共控制步后才开始；8000 轮、UTD 0.4 的新训练约有
20064 个公共控制步，因此这轮评估基本动作跟踪，不代表已经验证带课程扰动的稳健性。
保持这个源行为，后续需要扰动评估时单独记录评估条件。

```bash
XDG_CACHE_HOME=/tmp/unisupermimic-cache uv run pytest -q \
  tests/tasks/test_pen* tests/rl/algos/test_pen* \
  tests/rl/algos/test_dex_supermimic.py tests/tasks/test_hand1_supermimic.py
```

```bash
XDG_CACHE_HOME=/tmp/unisupermimic-cache uv run pytest -q \
  -m gpu tests/tasks/test_pen_spin_mjwarp.py
```

## 2026-11-04 复核与 Hand1 验证准备

本节保留当时的检查和运行命令。Hand1 当时未覆盖完整环境配置优先级、无训练 reader
的回放读阶段，以及实际后端 wrist reset 提交；本轮查出的差异修正了此前过强的结论。

本次对照只使用当前工作区的三个 `references/` checkout：

- Dex-Supermimic：`2becb7380fed86171056f9bf4338b2a6a0c14b01`，任务和算法依据。
- Uni-SuperMimic：`faee48a875579c45b3a41e6399b05ad014641ebf`，UniLab 生命周期参考。
- wuji_unilab：`f454719c8dd9d9afae0539c069c14652f85dad6f`，手部任务接口参考。

下表保留当时的 Pen 核对范围。该轮未检查锁定 mjlab 的噪声处理顺序、reset 内命令
推进、play helper 的提前返回和 MJCF childclass 继承；因此当时“没有新偏差”的
结论不能覆盖这些环节，2026-11-05 的新证据和修复见文末。

| 核对项 | 结果 |
|---|---|
| Actor / WM / Critic | 输入 155 / 298 / 301，WM 输出 67；层宽、激活、归一化、CUDA AMP 与源一致 |
| WM 监督与策略训练 | 多步监督、失败采样、解析递推、lambda-return、优化器及梯度裁剪与源一致 |
| 观测与动作 | 已核对项目顺序、坐标系、相机偏置/延迟/丢帧、残差 EMA、encoder 校正和冻结公式；未覆盖真实关节速度噪声与 scale 的顺序 |
| 任务与数据 | 已核对真实 `03_continuous_multi_axis` catalog、adaptive RSI 和局部 reset；未覆盖 reset 观测前的参考帧推进 |
| 物理与环境 | 已核对 50 Hz / 4×5 ms、Pen 标定、30 g 笔、startup DR 和奖励公式；未覆盖笔 geom 的继承默认值及 play helper 提前返回 |
| 本地回放外观 | 隐藏坐标轴、浅粉色参考模型只作用于 viewer，不改变训练配置或物理状态 |

已保存的 `m_ends/config.json` 是 `task=hand2-pen-spin`、
`algorithm=world_model`，不是 PPO。实际长训的 `pi_batch=512`，源默认是
1024；这是实际存在的训练批量差异，保留原网络和损失定义，但不应声称超参数完全相同。
本地配置还将 checkpoint 保存间隔由源默认 1000 改为 100，不改变优化过程。

此外，迁移版遇到 NaN/Inf 会立即报错；源版本部分路径会过滤无效行、
`nan_to_num` 或跳过 Actor 更新。此差异遵循用户此前要求，不改变正常有限数据的计算。
该次 Pen 代码比对所记录的其他改动包括 UniLab reader/bridge、CPU autocast、RNG
恢复、日志及输出目录；不能据此排除 Hand1 的环境执行差异。

本次实际执行的 Pen CPU 对照测试为 51 项，全部通过；另执行核心 learner 与
已有 Hand2 的 20 项回归测试，全部通过。当前会话无法访问 CUDA，真实 MJWarp 的
2 项集成测试跳过；前文 GPU 记录属于此前验证，不是本次重新执行的结果。
未进行同初态、同随机参数下的 mjlab/UniLab 双环境轨迹数值对比，也未在本次启动
长训或评估现有策略。上述结论支持进入 Hand1 验证，不代表已验证稳定转笔的策略效果。

Hand1 准备时修复了三项外围差异：`play=True` 将位置和姿态的
`injection_prob` 清零（训练仍为 0.02）；方块高度使用掌心局部 X 法线；
关节加速度 RMS 读取原生 `qacc`，并在首次 reset forward 前绑定 CPU 数据，
不再使用全零占位。不支持原生读取的后端明确报错。Hand1 的 12 项 CPU 测试
全部通过；新增 GPU 读取分支采用现有 Pen 路径，本次尚未实测。修改未涉及 Hand2
任务、Pen 训练代码或已有 checkpoint。

Hand1 的验证目录使用 `m_ends/hand1_smoke`，与现有 Pen 和 Hand2 结果分开。
先运行以下 20 轮短验证；它保留源 `[2048,2048,2048]` 世界模型，缩小采样批量并
从第 5 轮开始更新 Actor/Critic，以便检查完整学习链路。

```bash
XDG_CACHE_HOME=/tmp/unisupermimic-cache \
uv run unisupermimic-hand1-train \
  training.sim_backend=mjwarp \
  training.log_dir=m_ends/hand1_smoke \
  algo.iterations=20 \
  algo.num_envs=16 \
  algo.buffer_steps=128 \
  algo.warmup_steps=16 \
  algo.wm_window=4 \
  algo.wm_batches=2 \
  algo.wm_batch=64 \
  algo.pi_batch=32 \
  algo.horizon=8 \
  algo.max_horizon=8 \
  algo.wm_only_iters=5 \
  algo.critic_epochs=1 \
  algo.critic_minibatches=2 \
  algo.compile=false \
  algo.compile_policy=false \
  algo.log_interval=1 \
  algo.save_interval=10
```

启动日志应为 `algorithm=SuperMimic task=hand1-reorient`，并显示
`actor_input=207 wm_input=133 critic_input=211`。检查第 5 轮后的策略与 Critic
更新正常、损失均有限、`wm/reward_parity` 接近零。完成后生成 `model_10.pt`、
`model_20.pt` 和 `model_final.pt`。这次短验证用于确认链路，训练效果需后续正式训练
和回放验收。没有在后台自动启动此命令。

## 2026-11-05 Hand1 训练与回放复核

本轮以当前工作区 `references/Dex-Supermimic` 的最终 Hand1 50 Hz 环境配置和
`rl/dex_supermimic/cli.py` 的 Actor 回放路径为依据。用户观察到旋转不灵活后，重新
检查了实际 executor、读阶段、reset 和回放，而非只检查网络维度和能否运行。

### 确定存在的差异及修复

| 项目 | 原迁移代码的实际行为 | 源行为与本轮修复 |
|---|---|---|
| 最终求解器配置 | 场景继承资产的 `iterations=5`、`ls_iterations=8` | 源任务配置最终覆盖为 `10`、`20`；`hand1/scene.py` 现使用最终值，同时作用于训练和回放 |
| 物理读阶段的姿态缓存 | read plan 已更新，但命令可能仍返回上一控制帧的掌心/方块缓存；训练 reader 的刷新会掩盖部分回放问题 | `hand1/env.py` 在 post-physics 读阶段入口先使命令缓存失效，再读取状态及计算终止/奖励，回放不再依赖训练 reader 刷新 |
| CPU wrist reset | 旧 MJCF host 后端不支持公开 mocap reset binding 时，静默设为 `None`，跳过了配置中的腕部随机化 | `hand1/reset_bridge.py` 对该后端提供局部兼容绑定；`ResetWuji` 不再静默跳过，真实掌心姿态按源 pitch 范围 `[-0.4, 0.1]` rad 重置 |
| MJWarp wrist reset 提交 | 后端缺少 packed mocap reset 能力时，提前写入的腕姿态可能被随后 qpos/qvel reset 提交恢复为默认值 | 将选定环境的 mocap 写入延迟到 reset transaction 提交之后、read plan 刷新之前，并保留未 reset 的环境 |
| 自适应外力课程 | 原生 manager 返回的课程进度已变化，但没有同步 env 的 `_adaptive_episode_curriculum`，想象扰动长期读取默认 0.05，checkpoint 也未保存该 env 状态 | `hand1/rewards.py` 同步 env 标量并保留 command 镜像；manager 结果、想象和 checkpoint 按源约定分别读取对应状态 |
| Hand1 episode 监控 | 共享 collector 只为 Pen 转发 `Episode_Metrics`，旧 Hand1 CSV 缺少这些原生指标 | `runner.py` 复用现有转发机制，`metrics.py` 增加 13 个 Hand1 列；旧行及自定义列保留，新列在旧行留空 |

wrist reset 修复同时影响训练和 `play=True`：源评估关闭 startup DR、interval
扰动、观测噪声和课程，但保留 reset 时的关节、方块和腕部随机初态，因此不能以
“干净评估”为由固定腕姿态。显示层仍只复制真实 qpos/qvel/mocap 到独立视觉模型，
世界系目标四元数直接用于六面目标及状态灯，没有额外乘掌心或 tag 旋转。

课程的读取边界保持源行为：真实外力从 manager 的课程缓存读取，想象直接读取 env
标量，checkpoint 保存/恢复 env 标量。源恢复流程先设置 env 标量，在下一次 reset
计算课程时同步 manager 缓存；本轮没有另改这项恢复顺序。command 的 `adaptive`
仅保留镜像，不作为额外消费者或新的课程权威。

续训时的课程恢复保留源行为：先恢复 env 标量，manager 的真实扰动缓存到下一次
reset 计算课程时才同步；没有额外改动源 `load()` 的顺序。本轮修正的是 env 标量
未被维护，导致想象长期回退到默认值、checkpoint 缺失实际进度的迁移差异。

### 未发现偏差的核心部分与保留的适配

- Actor 输入 207，三帧按观测项拼接、每项由旧到新；Actor 输出 20，层宽
  `[512, 256, 128]`。回放直接执行已保存的 Actor 和归一化，不添加探索噪声，
  不更新归一化，也没有重新排列历史帧。
- WM 输入 133，输出 75（52 维状态增量和 23 维接触）；Critic 输入 211，
  隐藏层 `[512, 512, 256, 128]`。本轮没有发现多步监督、想象递推、lambda-return
  或梯度更新公式的新差异。
- 已核对的控制配置为 50 Hz、`0.01 s × 2` 物理子步、EMA 0.3、`0.4 s` 初始
  姿态保持、0.9 soft limits 和 encoder 校正；成功阈值 `0.2 rad`、连续 13 步保持、
  成功后 50 步切换目标沿用源设置。真实执行目标与 WM replay 的 `u=c−encoder_bias`
  对应，不能因视觉上动作不灵活就另改动作幅度或奖励。
- 实际 `m_ends/hand1_reorient/config.json` 是 SuperMimic `world_model`，1024 个
  环境、8000 轮、WM `[2048, 2048, 2048]`；`pi_batch=512` 是显存适配，源默认
  为 1024，因此超参数并非完全相同。实际保存间隔为 500，源默认 1000，该项不改变
  优化过程。

### 实际 checkpoint 与训练记录

旧 8000 轮日志的末尾 20 条 `task/rot_error` 均值约为 `0.993 rad`（56.9°），
相比初期约 125° 有改善，但仍有明显残余误差。对应 WM train/holdout loss 均值约
为 4.84 / 13.66，Critic loss 均值约为 48.1；这些 loss 受归一化、累加窗口及回报
标度影响，不能单凭绝对值判定原因或策略效果。

真实奖励与解析奖励的差值多数约 `5e-8`，少数离散峰值约 `0.000781` 和 `0.003125`；
这些值与少数环境处于 cage 边界时的半步/整步罚分标度吻合，不是整条奖励计算大面积
失配的证据。旧 CSV 没有 Hand1 的原生 episode 指标，不能从中还原成功目标总数；
`drop_rate` 接近零只能说明采样步中的掉落终止少，`wm/survival` 是想象存活率。

新增的 `task/episode_*` 保留 UniLab 原生的 episode 内时间平均，再按本次 collect
完成的 episode 数量加权。尤其 `goal_reach_count` 在这里是被时间平均的计数，不能
另解释成最终目标总数或成功率。无 episode 完成时留空，不填伪零；新增日志仅作用于
后续运行，没有改写已有训练记录。

### 两个 seed 的 CPU 回放检查

读取同一个旧 `model_final.pt`，每个 seed 各运行 3000 控制步（60 秒），比较修复前后
的实际执行。两组均为单环境干净评估，没有训练或 checkpoint 更新。

| Seed | 完成目标数：旧→修复后 | 平均姿态误差：旧→修复后 | 阈值内时间占比：旧→修复后 |
|---|---:|---:|---:|
| 42 | 2 → 3 | 48.9° → 57.7° | 5.43% → 4.77% |
| 43 | 5 → 9 | 44.2° → 53.6° | 12.03% → 25.43% |

旧 executor 的 solver 实测为 5/8，修复后为 10/20。四次检查都没有失败终止，
control 张量全部有限；修复后平均绝对动作约为 0.663 / 0.616，方块平均角速度约为
2.775 / 3.224 rad/s。此处完成目标数直接读取命令的成功事件计数，区别于上述原生
episode 时间平均日志。

目标完成数增加，但平均误差未降低，不能宣称旋转问题已经解决。此次同时修正了
solver、姿态读缓存和 wrist/reset；达到目标的次数变化也会改变后续随机目标序列。
这不是单因子对比，不能把变化全部归因于 solver，更不能用两条短轨迹证明训练收敛。

### 本轮验证范围与后续验证

- 聚焦 CPU 回归记录为 **51 项通过**，包括 Hand1、共享 learner 及相关回放/监控检查。
- `tests/tasks/test_hand1_mjwarp.py` 的真实 GPU 检查为 **2 项通过**，不是 skip；
  覆盖实际设备侧环境契约。依赖告警涉及 MJWarp capsule/mesh 多点接触限制及只读
  NumPy 转换，不表示这两项检查失败。
- 同一个旧 8000 轮 checkpoint 的真实 MJWarp 无窗口 150 步推理通过：0 次 episode
  结束、0 次失败、0 个完成目标，平均单步奖励约 0.2204。这只验证加载和执行。
- 真实 MJWarp 的两 seed、各 3000 步（60 秒）检查完成，结果见下表。
  两组均为正常 60 秒超时结束，0 次失败，control 全部有限；没有训练或录制视频。
- 本轮没有运行新的长训练，也没有进行同初态、同物理参数下 mjlab 与 UniLab 的
  完整逐轨迹数值对比。CPU 结果、GPU 接口检查及源码对照分别提供有限证据，不能
  互相替代，也不能作为稳定灵活旋转的最终验收。

| GPU Seed | 完成目标数 | 平均姿态误差 | 阈值内时间占比 |
|---|---:|---:|---:|
| 42 | 9 | 64.35° | 19.40% |
| 43 | 7 | 44.19° | 18.07% |

两组 GPU 回放的平均绝对动作约为 0.661 / 0.596，方块平均角速度约为
3.691 / 3.382 rad/s。动作已经实际执行，旧策略仍有明显目标跟踪误差。
这不是修复前后的 GPU 对照；CPU 的完成次数和误差也不能代替 GPU 后端结果。

旧 checkpoint 是在原求解器和原 reset 行为下训练的；只修回放无法补上训练中缺失
的腕姿态分布或重新拟合其世界模型。接下来应使用修复后的环境从头验证一轮训练，
采用新目录（例如 `m_ends/hand1_aligned`）保留原结果，再结合真实目标完成情况和
回放判断效果。不要将旧 checkpoint 在新环境中能运行等同于迁移已经完整对齐。

## 2026-11-05 Pen 训练与回放复核

本轮针对“笔转不准”重新检查最终任务配置、真实 Actor 输入、reset 与回放。
除当前工作区的源 checkout 外，还核对了其 `uv.lock` 精确锁定的 mjlab v1.6.0
提交 `0fb8a681136be94ffc636a3dd423cabb97d91f10` 中的 ObservationManager 和
reset 流程。仅比较本仓库的解析 MDP 与源解析 MDP，会漏掉这些依赖层行为。
依赖层依据为该提交的
[观测处理顺序](https://github.com/mujocolab/mjlab/blob/0fb8a681136be94ffc636a3dd423cabb97d91f10/src/mjlab/managers/observation_manager.py#L346)
及 [reset 调度](https://github.com/mujocolab/mjlab/blob/0fb8a681136be94ffc636a3dd423cabb97d91f10/src/mjlab/envs/manager_based_rl_env.py#L358)。

### 已确认的差异及修复

| 项目 | 原迁移行为 | 源行为与本轮修复 |
|---|---|---|
| 真实关节速度观测的噪声顺序 | `0.25*qd + Uniform(-0.05,0.05)`，实际 Actor 噪声幅度为源的四倍 | 锁定 mjlab 的处理顺序为 compute → noise → clip → scale；源 term 配置 noise ±0.05、scale 0.25，因此实际输入为 `(qd+noise)*0.25`，有效噪声 ±0.0125；`observations.py` 已按此修复，clean Critic/play 仍为 `0.25*qd` |
| reset 后的参考帧 | 采样帧 k 写入物理状态后，`refresh_reset` 未推进命令，Actor reset 观测仍使用参考 k | 源 reset 在物理 forward 后调用 `command.compute(dt=0, env_ids=selected)`，使 Actor 使用参考 k+1；`commands.py` 现仅推进选定行，保持未 reset 行及其控制/相机状态。首帧 hold 的 -50 也相应推进为 -49 |
| play 的训练项继承 | 回放从训练配置组合后仍保留课程、reset 噪声、动作冻结及训练指标 | 源 `_configure_hand2_reward_and_curriculum_surface` 在 play 分支提前返回；`play_config.py` 已清空这些训练项，保留固定硬件笔、clean 观测、uniform 完整 clip、首帧 hold 和源 play 终止阈值 |
| 笔 geom 的接触默认值 | 新笔 geom 继承共享 Cube body 的 childclass，携带 Cube 接触调参 | `scene.py` 对笔的三个 geom 显式设置源字段：friction=`1 .005 .0001`、solref=`.02 1`、solimp=`.9 .95 .001 .5 2`，避免 Cube 默认值泄漏 |
| CPU 观测的物理阶段 | 本任务 host read plan 未命中聚合 body 快路径，逐个读取 tracked sensor 时未刷新标脏的派生姿态，实际 Actor 端点观测较最终 qpos 差约 0.57–0.68 mm | `registry.py` 仅为 Pen CPU 环境安装 `reset_bridge.py` 的幂等观测钩子；在计算观测前刷新 tracked kinematics 并重新发布 read plan。保留 reward/termination 阶段及 GPU 路径，不增加物理积分或完整 forward |

噪声和 reset 参考帧修复会改变训练数据与 Actor 输入；它们不是仅有显示效果的
修正。旧 checkpoint 可以继续用于观察和对比，但其训练期间经历的噪声与参考相位
不会因替换回放代码而补齐。

笔接触字段的差异确实存在，但普通 hand–pen 接触中，手部 geom 的 `priority`
高于笔，有效接触参数取手部优先项。因此不能把该字段修复直接解释为此前手与笔
接触异常或跟踪误差的主要原因；它仍应按源显式设置，以消除其他接触路径的偏差。
显示层的浅粉色参考模型及隐藏坐标轴保持不变，不参与控制或碰撞计算。

### 已核对的网络、数据流与保留的源边界

- Actor 输入 155、输出 20，隐藏层 `[512,256,128]`；WM 输入 298、输出 67
  （52 维物理增量 + 15 个自碰撞 logits），隐藏层 `[2048,2048,2048]`；Critic
  输入 301，隐藏层 `[512,512,256,128]`。FailureHead 输入 426。核对实际源码及
  checkpoint 权重形状，没有发现需要修改的新网络结构或优化公式差异；源说明文档
  部分旧层宽描述不能替代实际代码。
- 物理状态为 `[q20, qd20, p3, R9, v3, w3]`，WM 条件保留 46 维固定仿真
  context、174 维 body wrench 和 20 维实际执行绝对目标。参考、残差 EMA、冻结、
  相机记忆和过流积分按源解析递推，reward/termination 使用步入口参考，再推进时间。
- 本轮真实 Actor 噪声修复不同时改动 `PenMDP.policy_observation()`：源解析想象
  自身仍在 `0.25*qd` 后加入 ±0.05 噪声。本轮保留这个源内部差别，未另造一套
  想象噪声幅度。源 FailureHead 的 true/predicted context 时相差、想象期间固定
  起点 wrench 和 reference-end 的处理等已有边界也未借此更改。
- 实际 `m_ends/config.json` 与 checkpoint 为 SuperMimic `world_model`，8000 轮、
  1024 环境；`pi_batch=512`，源默认 1024。保存间隔为 100，源默认 1000。
  本地 PyTorch 为 2.14.1，源锁定 2.10.0；Warp 为 1.16.0，源锁定 1.14.0；
  MuJoCo 和 mujoco-warp 均为 3.11.0。这些是仍需记录的环境差异，没有自动降级
  依赖，也不能声称运行环境完全一致。

读阶段已按实际 Actor 输出复核：锁定 mjlab 在 reward/termination 后无条件
forward，再计算观测；UniLab 的 read plan 调用刷新并不保证所有 CPU sensor
读取路径已经更新。新增回归先复制实际返回的 155 维 Actor 输入及 Critic 链接
位置，再从最终 qpos 独立 `mj_forward` 核对；CPU 修复后与原有 MJWarp 路径均
通过。训练相机也应接收这个已刷新的物理输入，再叠加源相机延迟与噪声。

源 native reward 的派生 pose 允许落后一个 5 ms 子步；MJWarp 的 UniLab tracked
pose 在 reward 前已经刷新。这是保留的后端生命周期边界，没有为追求 reward
数值一致而人为制造旧 pose。CPU 修复只插在观测前，不改变已采集的奖励或终止。
WM 真实监督与想象仍按源最终 qpos 路径处理；这些局部检查不等同于双环境轨迹等价。

源训练课程的 warmup 是 40000 个公共控制步、到 240000 步达到完整强度。
现有 8000 轮 checkpoint 的 `common_step=20064`，因此该次外力、冻结及 reset
噪声课程仍为零，startup 物理 DR 正常生效。这是源最终配置在 SuperMimic 的 UTD
时间尺度下的实际行为，不能直接套用 PPO 每轮 40 步的注释解释，也未自行压缩课程。
旧 play 虽误留课程，前 40000 步的零缩放恰好隐藏了这些错误继承。

### 现有训练记录的含义

`m_ends/train_log.csv` 最后 20 条已完成 episode 的原生中心误差均值为 **1.42 cm**、
笔轴夹角均值为 **13.02°**；前 10 条分别为 2.09 cm、13.83°。位置改善较明显，
方向改善有限。关节加速度 RMS 均值由约 48.56 升至 97.45 rad/s²，动作仍存在
较明显的快速变化。末尾 WM train/holdout loss 均值约 1.21 / 3.77，不能把训练
loss 下降或低 drop 值直接解释成精准跟踪。

以上是被本次 collect 采到的已完成 episode 时间平均，受 RSI 起点和提前终止影响，
不是所有完整 clip 的统一评估。`task/drop_rate` 是采样步终止比例，`wm/survival`
是想象存活率。Pen 的 `task/rot_error` 使用恒等目标，不是参考轨迹；此前将它描述
为参考 SO(3) 误差有误。应使用 native 中心/笔轴指标和独立整段回放结果。

### 修复前的真实 MJWarp 策略基线

同一个旧 8000 轮 Actor，每个 seed 回放 3000 控制步（60 秒），不训练、不改统计、
不生成视频。下表是本轮修改前的 GPU 基线；中心及笔轴误差直接与实际参考帧比较，
control 均为有限值。

| Seed | 中心平均误差 | 笔轴平均夹角 | 笔轴夹角 P95 | 完整 SO(3) 平均误差 | Episode 结束数 | 失败数 | 完成 clip 数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8844 cm | 8.4896° | 22.1660° | 40.3370° | 3 | 1 | 2 |
| 43 | 0.9705 cm | 10.5984° | 45.9344° | 68.4315° | 4 | 2 | 2 |

源目标用笔轴两端点跟踪位置及轴方向，不约束绕笔轴的 roll；因此完整 SO(3) 误差
可能较大，不能把表中 40°/68° 解释成同等程度的任务笔轴偏差。与此同时，笔轴
P95 和失败次数表明旧策略并非始终准确；平均轴误差较小也不足以证明稳定完整跟踪。
这里使用源 play 的 20 cm 位置终止阈值，并移除方向提前终止，完成 clip 数不能等同
于训练中 5 cm / 60° 阈值下的成功率。

### 修复后的旧模型回放

重新加载同一个 8000 轮 Actor，以更新后的完整 play 配置运行同样两个 seed，
各 3000 控制步（60 秒）：

| Seed | 中心平均误差 | 笔轴平均夹角 | 笔轴夹角 P95 | Episode 结束数 | 失败数 | 完成 clip 数 |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.8000 cm | 6.8639° | 18.6376° | 2 | 0 | 2 |
| 43 | 0.9117 cm | 9.2718° | 21.3240° | 3 | 1 | 2 |

控制量均有限，实际轴误差仍非零且存在失败。两组短测中误差降低，但重置参考相位、
play 随机数消费和后续采样的 clip 序列也已变化，不能将此解释为相同轨迹上的
单因素改进，更不能证明修复后的训练效果。原模型是在旧噪声和 reset 相位下训练的，
本轮没有更新任何权重。

### 本轮验证与结论边界

本轮最终验证：

- Pen 相关 CPU 回归 **69 项通过**，GPU 项按标记排除，未用 skip 充当 GPU 验证。
- 真实 RTX 5070 Ti 上的 MJWarp 集成检查 **3 项通过**：startup 模型表、局部
  reset、非零外力与 WM 标签，以及实际 Actor 端点/Critic 链接对最终 qpos 的独立 FK。
- 现有 Hand1 环境/回放及 Hand2 迁移基础回归 **30 项通过**。
- 本轮改动的 Pen 源文件和测试全部通过 Ruff。

CPU 观测回归从源 Pen 配置 AST 提取 noise/scale，以受控 RNG 对比实际 155 维
Actor 的速度切片，检查有效 ±0.0125 幅度及 clean Critic/play。其他新增回归覆盖
选定行 reset 的 k→k+1、非零训练时间下 play 无冻结、实际笔 geom 字段和接触
priority。CPU 刷新检查另验证真实训练相机原始输入与最终 qpos 一致、配置的一步
延迟保留、每步只推进一次相机历史、重复安装幂等、局部 reset 保留存活行。

```bash
XDG_CACHE_HOME=/tmp/unisupermimic-cache uv run pytest -q -m 'not gpu' \
  tests/tasks/test_pen*.py tests/rl/algos/test_pen*.py \
  tests/rl/algos/test_dex_supermimic.py

XDG_CACHE_HOME=/tmp/unisupermimic-cache uv run pytest -q -m gpu \
  tests/tasks/test_pen_spin_mjwarp.py tests/tasks/test_pen_observation_phase.py
```

本轮没有重新长训，没有改写 `m_ends/` 的日志或 checkpoint，也没有改动已完成的
Hand2 Cube 任务。修复后旧模型的回放只能检查新执行行为，不能证明修复后的训练
已经收敛；仍需在新目录中进行训练及完整轨迹验证。尚未完成同初态、同物理参数的
mjlab/UniLab 双环境逐步轨迹对照，因此保留上述版本和源算法本身的已知限制。
