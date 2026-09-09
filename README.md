# Wuji UniLab

Wuji Hand 灵巧手立方体重定向任务，运行在 [UniLab](https://github.com/Motphys/UniLab) 环境、[UniSim](https://github.com/unilabsim/unisim) 的 **MuJoCo-Warp（`mjwarp`）** 后端，以及 [unilab-rl](https://github.com/unilabsim/unilab_rl) 训练框架上。

## 项目定位

本仓库是 [wuji-mjlab](https://github.com/wuji-technology/wuji-mjlab) 的 UniLab 迁移实现。任务目标、模型资产和主要奖励语义保持一致；环境管理、仿真接口和 PPO runner 由上游项目提供。只支持 `mjwarp`，不静默切换后端。两套实现不承诺 RNG、张量排列、离散事件时序或 checkpoint 互相兼容。

模型 XML、网格和纹理随仓库分发，不依赖 Hugging Face。能力覆盖和后续计划见 [roadmap issue #1](https://github.com/unilabsim/wuji_unilab/issues/1)。

## 安装

需要 Linux x86_64、Python 3.11–3.13、NVIDIA 驱动及 CUDA。首次运行会编译 Warp kernel。

```bash
git clone https://github.com/unilabsim/wuji_unilab.git && cd wuji_unilab
uv sync --locked --all-groups
uv run --no-sync wuji-assets
uv run --no-sync wuji-list-envs
```

锁文件优先采用 UniLab 依赖版本；当前环境包含 MuJoCo/MuJoCo-Warp 3.11、Warp 1.16、Torch 2.8 cu128。源码安装不会发布 PyPI 包。

## 训练

```bash
uv run --no-sync wuji-train --task WujiHand_Reorient
uv run --no-sync wuji-train --task WujiHand_Reorient_Light
```

可追加 Hydra 参数，例如 `algo.num_envs=1024 training.log_dir=logs/test`。小规模启动检查：

```bash
uv run --no-sync wuji-train --task WujiHand_Reorient algo.num_envs=32 algo.max_iterations=3 algo.num_steps_per_env=8
```

## 评估与回放

无渲染统计评估（默认 100 trials）：

```bash
uv run --no-sync wuji-eval --task WujiHand_Reorient --checkpoint-file /path/to/model_4999.pt --num-trials 100 --output eval_results.json
```

结果写入 JSON。该口径与源项目 evaluator 对齐，但本入口使用 PyTorch checkpoint 和 mjwarp；源项目公开 evaluator 使用 CPU MuJoCo 与 ONNX，结果需谨慎比较。

离屏录制（EGL）：

```bash
uv run --no-sync wuji-play --task WujiHand_Reorient --checkpoint-file /path/to/model_4999.pt training.play_render_mode=record training.play_steps=1200
```

交互式 MuJoCo viewer（需要桌面显示和 GLFW）：

```bash
uv run --no-sync wuji-play --task WujiHand_Reorient --checkpoint-file /path/to/model_4999.pt training.play_render_mode=interactive
```

交互式模式需要 GLFW/X11；如果 shell 中为离屏录制设置过 `MUJOCO_GL=egl`，请先取消该变量：

```bash
env -u MUJOCO_GL uv run --no-sync wuji-play --task WujiHand_Reorient \
  --checkpoint-file /path/to/model_4999.pt training.play_render_mode=interactive
```

仅使用 `mjwarp` 物理，渲染器复用 MuJoCo；可用 `CUDA_VISIBLE_DEVICES=1` 选择 GPU。`training.play_render_mode=none` 只跳过渲染，不执行统计评估。

## 开发与许可

```bash
make check
make test
make test-all
```

代码位于 `src/wuji_unilab/`，配置位于 `src/wuji_unilab/conf/`。项目采用 [Apache-2.0](LICENSE)；第三方声明见 [`NOTICE`](NOTICE) 与 [`LICENSES/`](LICENSES/)。
