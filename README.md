# Wuji UniLab

基于 [UniLab](https://github.com/Motphys/UniLab) 的 Wuji Hand 手内立方体重定向训练任务，
仅使用 `mjwarp` 物理后端。仓库包含任务配置、控制与奖励、Wuji 模型资产、训练状态适配以及
策略导出入口；环境、仿真和训练框架分别由 UniLab、UniSim 和 unilab-rl 提供。

目前提供完整训练、checkpoint 恢复、策略回放和 ONNX 导出。已执行的训练、具体指标和证据边界见
[pc823 完整训练报告](docs/training-result-pc823.md)；迁移过程中的能力探测见
[能力审计与有限预算验证](docs/capability-evidence.md)。训练程序完成运行本身不等于策略质量验收。

## 与 wuji-mjlab 的关系

本项目迁移自 [wuji-technology/wuji-mjlab](https://github.com/wuji-technology/wuji-mjlab)，
源代码审计基线为 `9410a3ac9caf57ac0b8c74c0a5a63f2b466c3003`。
原项目提供 Wuji Hand 的 mjlab 任务和训练工作流；本项目将对应任务接入 UniLab 的环境与训练接口。
原仓库是任务和资产的来源，运行本项目无需另外安装 `wuji-mjlab`。

| 项目 | 本仓库的处理方式 |
| --- | --- |
| 任务目标 | 保留 Wuji Hand 立方体重定向、目标切换、掉落终止、课程和域随机化等任务功能 |
| 环境与配置 | 使用 UniLab manager 环境及 Hydra owner YAML；本仓库通过任务插件注册 |
| 仿真 | 使用 UniSim 的公开后端接口；仅支持 `mjwarp` |
| 训练 | 使用 unilab-rl（Python 包名 `uni_rl`）及 RSL-RL PPO runner，加入 Wuji Softplus 动作分布和显式课程状态保存 |
| 依赖版本 | 优先采用 UniLab 依赖线，不要求与源项目的 Torch、MuJoCo、MuJoCo-Warp、Warp 版本一致 |
| 行为对齐 | 关注任务功能、有效学习、reset 隔离和策略接口；不要求 RNG、配置/逻辑布局、张量顺序和离散事件逐项一致 |
| 模型资产 | XML、mesh、texture 随本仓库保存并校验 SHA-256，不从 Hugging Face 下载 |
| checkpoint | 使用本项目的观测与训练状态 schema；不承诺原 wuji-mjlab checkpoint 可直接加载 |
| 能力边界 | 当前任务的支持范围不代表 UniLab/UniSim/unilab-rl 已覆盖 mjlab 的全部训练能力 |

不支持的后端、接口或训练状态应明确报错并记录给 maintainer 决策，不自动更换后端、关闭功能或伪造结果。
能力缺口及其处理由 [roadmap #1](https://github.com/unilabsim/wuji_unilab/issues/1) 跟踪。

## 环境与安装

- Linux x86_64，Python `>=3.11,<3.14`，安装 [uv](https://docs.astral.sh/uv/)。
- NVIDIA GPU 和可运行 CUDA 12.8 PyTorch wheel 的驱动；训练、回放和导出需要 CUDA。
- 安装时需要访问 GitHub、Python 包索引及 PyTorch CUDA 索引，并具备依赖仓库的读取权限。
- 默认训练并行环境数量较大；实际显存需求和吞吐以机器实测为准。首次运行会编译 Warp kernel。

```bash
git clone https://github.com/unilabsim/wuji_unilab.git
cd wuji_unilab
uv sync --locked --all-groups
uv run --no-sync wuji-assets
uv run --no-sync wuji-list-envs
uv run --no-sync python -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available())'
```

任务列表应包含 `WujiHand_Reorient` 和 `WujiHand_Reorient_Light`，后端均为 `mjwarp`。
CUDA 检查必须为 `True` 才能继续训练。`wuji-assets` 校验本地资产完整性；资产缺失或校验不符会报错。

请使用仓库中的 [pyproject.toml](pyproject.toml) 和 [uv.lock](uv.lock) 安装。
虽然依赖声明包含版本号，`tool.uv.sources` 实际将三个上游固定在以下 Git 提交：

| 依赖 | 固定提交 |
| --- | --- |
| UniLab | `55a5e14952208d3db881421d843a1a13e05690c4` |
| UniSim / `unisim-core` | `c82d8cec796d6ce43862cd0574a14d0a1ba6c3c9` |
| unilab-rl / `uni_rl` | `9bdf5e882c77b67496a2af3bbfd314edde532ceb` |

这组提交包含迁移所需的 reset、域随机化和训练状态接口；仅安装同名已发布版本或最新 `main`
不能替代该锁定环境。当前物理依赖为 MuJoCo/MuJoCo-Warp `3.11.0`、Warp `1.16.0`，
Torch 为 `2.8.0+cu128`，RSL-RL 为 `5.0.1`。更新依赖需要重新验证能力与训练结果。
本工作流通过源码安装，不发布 PyPI 包。

## 训练

### 默认完整训练

```bash
uv run --no-sync wuji-train --task WujiHand_Reorient
```

默认单卡配置：8,192 个环境、5,000 次 PPO iteration、每环境每次采样 40 个控制步，
合计 `1,638,400,000` environment steps。仿真步长为 `0.01 s`，控制周期为 `0.05 s`，
单个 episode 最长 `50 s`。策略输入 207 维、critic 输入 413 维、动作 20 维。
每 50 次 iteration 保存 checkpoint，训练完成后也保存；默认使用 TensorBoard，训练结束不自动回放。

配置源是 [主任务 YAML](src/wuji_unilab/conf/ppo/task/wuji_reorient/mjwarp.yaml)。
通过 `--task` 和 `--sim mjwarp` 选择任务与后端；不能通过 `training.sim_backend` 等 override 绕过入口限制。
其他参数使用 Hydra 的 `key=value` 形式，例如：

```bash
uv run --no-sync wuji-train --task WujiHand_Reorient \
  algo.num_envs=1024 algo.seed=43 training.log_dir=logs/wuji_seed43
```

修改环境数会改变采样预算和训练条件，不能再将其结果称为默认完整训练。
`training.log_dir` 可指定一个独立运行目录；为不同实验使用不同目录以避免混淆产物。

### Light 任务

```bash
uv run --no-sync wuji-train --task WujiHand_Reorient_Light
```

[Light YAML](src/wuji_unilab/conf/ppo/task/wuji_reorient_light/mjwarp.yaml) 继承主任务，
默认改为 4,096 个环境、7,500 次 iteration，并将随机化和扰动的渐进预算设为 300,000 控制步。
默认总采样量为 `1,228,800,000` environment steps；Light 仍是完整训练配置。

### 小规模启动验证

```bash
uv run --no-sync wuji-train --task WujiHand_Reorient \
  algo.num_envs=32 algo.max_iterations=3 algo.num_steps_per_env=8 \
  algo.save_interval=1 training.log_dir=logs/wuji_smoke
```

该命令验证初始化、采样、PPO 更新和 checkpoint 写入，不用于判断是否学会重定向。

### 从 checkpoint 继续训练

```bash
uv run --no-sync wuji-train --task WujiHand_Reorient \
  --checkpoint-file /absolute/path/to/run/model_4999.pt \
  algo.max_iterations=1000 training.log_dir=logs/wuji_resume
```

将路径替换为本项目生成的 checkpoint；使用与原运行一致的任务、模型和相关环境配置。
入口将 `--checkpoint-file` 映射到上游真正使用的 `algo.load_run`。
这里的 `algo.max_iterations=1000` 表示本次再运行 1,000 次 iteration，而非训练总次数上限。
checkpoint 恢复模型、优化器、迭代/日志进度以及版本化的 Wuji 课程/环境训练进度；
它不提供 RNG 或每个物理环境瞬间状态的逐位续接。
缺失训练状态 schema、观测分组不匹配或课程数据非法时拒绝恢复。

### 后台运行与日志

```bash
mkdir -p logs
nohup uv run --no-sync wuji-train --task WujiHand_Reorient \
  > logs/full-training.log 2>&1 < /dev/null &
echo $!
```

默认运行产物位于 `logs/<任务名>/<时间与后端命名的运行目录>/`，包括 `model_*.pt`、
`run_config.json`、TensorBoard events，以及正常完成后的 `run_summary.json`。
`run_config.json` 记录配置和策略兼容性契约，评估或归档时应与 checkpoint 一起保留。
控制台日志重定向文件与该运行产物目录是两个不同位置。

```bash
tail -n 60 logs/full-training.log
uv run --no-sync tensorboard --logdir logs --host 127.0.0.1 --port 6006
```

检查 `run_summary.json` 的完成状态、采样量、耗时，并结合 TensorBoard 中的 reward、episode 长度、
success、orientation error、掉落相关终止和数值稳定性分析结果。日志里的训练期 success 不能直接作为
独立测试集成功率。pc823 的完整训练记录及源项目公开结果可比性说明见
[训练结果报告](docs/training-result-pc823.md)。

## 回放与 eval

使用本项目 checkpoint 进行固定步数的策略推理并录制视频：

```bash
uv run --no-sync wuji-play --task WujiHand_Reorient \
  --checkpoint-file /absolute/path/to/run/model_4999.pt \
  training.play_env_num=4 training.play_steps=1200 \
  training.play_render_mode=record
```

该命令运行 1,200 个控制步，相当于每个环境 `60 s` 的模拟时间，视频写入 checkpoint 所在目录的
`play_video.mp4`；重复回放会使用同一输出文件名。Light checkpoint 应使用
`--task WujiHand_Reorient_Light`。录制需要可用的离屏渲染环境；CUDA 可用并不等于渲染环境已配置。

任务默认启用 `play_profile`：关闭 reset 随机化、外部扰动和策略观测噪声/异常注入，
移除课程更新，并将 episode 上限改为 `60 s`。若要保留训练任务的环境配置，可显式使用：

```bash
uv run --no-sync wuji-play --task WujiHand_Reorient \
  --checkpoint-file /absolute/path/to/run/model_4999.pt \
  play_profile.enabled=false training.play_env_num=4 \
  training.play_steps=1200 training.play_render_mode=record
```

`training.play_render_mode=none` 的含义是跳过回放，不能用它执行无渲染 eval。
当前 `wuji-play` 是 checkpoint 推理/可视化入口，尚未提供独立的标准化质量评测命令或自动生成的
多 seed 成功率报告。正式与 wuji-mjlab 比较时，需要先固定 SO(3) 目标分布、成功/掉落定义、
episode 数量、环境扰动条件和训练交互预算，再测 success、goals/time、drops、survival 等指标。
不同奖励尺度下的 reward 曲线不能直接对比。

## ONNX 导出

```bash
uv run --no-sync wuji-export --task WujiHand_Reorient \
  --checkpoint-file /absolute/path/to/run/model_4999.pt \
  --output exports/wuji_reorient.onnx --max-diff 0.0001
```

导出需要 CUDA，会加载 checkpoint 并创建对应环境，生成 ONNX 及同名 `.json` 策略 schema。
省略 `--output` 时使用 checkpoint 路径替换扩展名。导出同时检查 PyTorch 与 ONNX Runtime
输出差异，默认最大绝对误差阈值为 `1e-4`；验证失败时不能将该产物作为已验证策略使用。

当前 schema 版本为 `wuji_unilab_policy` v1，包含任务、后端、控制周期、输入输出名/形状、
有序关节名、checkpoint 路径及数值误差。默认策略输入名为 `obs`、形状 `[1, 207]`，
输出名为 `actions`、形状 `[1, 20]`。部署端仍需按本项目观测构造、历史、动作缩放与控制语义接入；
schema 和一次数值一致性验证不等于真实硬件验证，也不承诺与原项目部署程序直接兼容。

## 验证与开发

```bash
make check
make test
make test-all
```

`make check` 执行 whitespace、Ruff 和 Pyright 检查；`make test` 默认排除 GPU/slow 标记；
`make test-all` 包含真实 CUDA/mjwarp 测试，需要可用 GPU。GPU 测试覆盖随机化后的稳定步进、
局部 reset 隔离和课程状态恢复；算法/导出 schema 等检查不替代完整训练与独立策略质量评测。

任务代码在 `src/wuji_unilab/tasks/reorient/`，权威配置在 `src/wuji_unilab/conf/ppo/task/`，
训练 wrapper 和动作分布在 `src/wuji_unilab/rl/`。通用环境、仿真能力和训练 runner 的修改应分别进入
UniLab、UniSim、unilab-rl owner；CLI 只负责配置与调用，不复制上游训练框架。

审计仍记录 post-substep sensor history、RND wrapper、camera/terrain、IK 等一般能力缺口；
这些不作为当前向量观测 Wuji 任务已支持的能力。多 GPU、跨后端和真实机器人控制也没有在本仓库完成验收。
新支持范围需在 roadmap 中明确，并附对应验证证据。

## 资产与许可

本项目采用 [Apache-2.0](LICENSE)，保留来源的 [NOTICE](NOTICE) 和 [第三方许可](LICENSES/rsl-rl.txt)。
Wuji 手模型及立方体资产来自上述 wuji-mjlab 基线，具体来源、许可与每文件 SHA-256 见
[assets/manifest.json](src/wuji_unilab/assets/manifest.json)。资产保存在仓库内，不上传 Hugging Face。
NOTICE 保留源项目的归属说明，其中历史路径并不表示本仓库仍以相同路径 vendoring 对应依赖。
