# Wuji UniLab

[中文](README_zh.md)

Wuji Hand 五指灵巧手的立方体掌上重定向任务：策略控制 20 个手指关节，把掌中立方体连续转向随机目标姿态。本项目是 [wuji-mjlab](https://github.com/wuji-technology/wuji-mjlab) 该任务向 [UniLab](https://github.com/Motphys/UniLab) + [UniSim](https://github.com/unilabsim/unisim) MuJoCo-Warp（`mjwarp`）GPU 后端 + [unilab-rl](https://github.com/unilabsim/unilab_rl)（rsl-rl PPO）的迁移实现，机器人模型与纹理随仓库分发。

![Wuji Hand 立方体重定向回放](docs/assets/wuji_reorient_play.gif)

*本实现的仿真回放（`wuji-play` record 模式，mjwarp，4 个并行环境）；线框立方体为目标姿态 ghost。*

## 可用能力

| 能力 | 状态 |
| --- | --- |
| 任务 `WujiHand_Reorient` | 默认规模（8192 环境 × 5000 迭代）训练已跑通，策略可持续完成重定向 |
| 任务 `WujiHand_Reorient_Light` | 轻量配置变体（4096 环境 × 7500 迭代） |
| 物理后端 | 仅 `mjwarp`，需要 NVIDIA GPU + CUDA；不静默切换后端 |
| 统计评估 | `wuji-eval`：与源项目同协议的逐 trial 成功率/掉落率/最小姿态误差 |
| 回放 / 录像 | `wuji-play`：离屏 mp4 或交互式 viewer |
| 策略导出 | `wuji-export`：ONNX + I/O schema，强制数值一致性校验 |
| 真机部署 | 未验证，本仓库不含部署代码 |

上游报告的 sim2real 85% 等数字属于 wuji-mjlab 原实现，本迁移版尚未用同协议复现。

## 首次运行

需要 Linux x86_64、Python 3.11–3.13、NVIDIA GPU 及 CUDA 驱动、[uv](https://docs.astral.sh/uv/)。首次运行会编译 Warp kernel。

```bash
git clone https://github.com/unilabsim/wuji_unilab.git && cd wuji_unilab
uv sync --locked --all-groups
uv run --no-sync wuji-assets       # 校验随仓库分发的资产
uv run --no-sync wuji-list-envs    # 列出任务
```

依赖全部由锁文件从 PyPI 解析（UniLab 1.2.0、UniSim 1.2.0、unilab-rl 1.2.0），无需单独克隆 UniLab；本仓库本身以源码安装，不发布 PyPI 包。仓库不分发训练好的 checkpoint，下面训练一节会得到自己的 checkpoint。

## 训练与评估

```bash
# 默认训练（8192 环境 × 5000 迭代，RTX 4090 约 6 小时）
uv run --no-sync wuji-train --task WujiHand_Reorient

# 小规模启动检查（几分钟，不代表已学会行为）
uv run --no-sync wuji-train --task WujiHand_Reorient algo.num_envs=32 algo.max_iterations=3 algo.num_steps_per_env=8
```

产物在 `logs/<task>/<日期>_<时间>_mjwarp/`（checkpoint、TensorBoard events、运行配置）。可追加 Hydra 参数，如 `algo.num_envs=1024`；任务与后端路由键被 CLI 锁定，不可覆盖。

```bash
# 统计评估（默认 100 trials，协议与源项目 evaluator 对齐，输出 JSON）
uv run --no-sync wuji-eval --task WujiHand_Reorient --checkpoint-file logs/.../model_4999.pt --output eval_results.json

# 回放：离屏录制 mp4（EGL）
MUJOCO_GL=egl uv run --no-sync wuji-play --task WujiHand_Reorient --checkpoint-file logs/.../model_4999.pt \
  training.play_render_mode=record training.play_steps=1200

# 回放：交互式 viewer（需桌面显示；设置过 MUJOCO_GL 请先取消）
env -u MUJOCO_GL uv run --no-sync wuji-play --task WujiHand_Reorient --checkpoint-file logs/.../model_4999.pt \
  training.play_render_mode=interactive

# 导出 ONNX 策略
uv run --no-sync wuji-export --checkpoint-file logs/.../model_4999.pt
```

多 GPU 机器可用 `CUDA_VISIBLE_DEVICES` 选卡。本实现的评估使用 PyTorch checkpoint + mjwarp，与上游 CPU MuJoCo + ONNX 的结果不能直接比较；两个项目的 checkpoint 互不兼容。

## 如何修改

| 想修改什么 | 入口文件 |
| --- | --- |
| 奖励 / 终止 / 课程 / 指标 | `src/wuji_unilab/tasks/reorient/rewards.py` |
| 观测（历史、噪声） | `src/wuji_unilab/tasks/reorient/observations.py` |
| 目标指令（成功阈值、保持步数） | `src/wuji_unilab/tasks/reorient/commands.py` |
| 动作与关节控制 | `src/wuji_unilab/tasks/reorient/actions.py` |
| 重置、域随机化、扰动 | `src/wuji_unilab/tasks/reorient/events.py` |
| 超参、网络、观测/动作等配置 | `src/wuji_unilab/conf/ppo/task/wuji_reorient/mjwarp.yaml` |
| 机器人 / 立方体模型 | `src/wuji_unilab/assets/`（改后更新 `assets/manifest.json` 校验值） |

开发检查：`make check`、`make test`、`make test-all`（后者需 CUDA）。

## 来源与致谢

任务、模型资产与奖励语义来自 [wuji-mjlab](https://github.com/wuji-technology/wuji-mjlab)（Wuji Technology），引用本工作时请同时引用上游。基础设施由 [UniLab](https://github.com/Motphys/UniLab)、[UniSim](https://github.com/unilabsim/unisim)、[unilab-rl](https://github.com/unilabsim/unilab_rl)（基于 [rsl_rl](https://github.com/leggedrobotics/rsl_rl)）和 [mujoco-warp](https://github.com/google-deepmind/mujoco_warp) 提供。

本项目采用 [Apache-2.0](LICENSE) 许可。
