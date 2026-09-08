# pc823 默认完整训练报告（2026-09-08）

## 状态与可复核产物

`WujiHand_Reorient` 完成全部 5000 轮训练；最终 checkpoint 可加载，包含
actor、critic、optimizer、iteration、logger 和显式课程状态。检查时训练进程
已退出，日志最后一轮为 `4999/5000`、ETA 为零。启动时没有额外保存 shell
退出码，因此完成判据是日志、TensorBoard 5000 个标量记录与最终 checkpoint。

- 机器：`pc823`，Linux，NVIDIA RTX 4090，驱动 595.84，报告显存 49140 MiB。
- 工作目录：`/home/pc823/ws/unilabsim/wuji_unilab`。
- 下游提交：`2ea17bba8c3ea283127ef2b16f75aa77002ed7d0`。
- UniLab：`55a5e14952208d3db881421d843a1a13e05690c4`。
- UniSim：`c82d8cec796d6ce43862cd0574a14d0a1ba6c3c9`。
- unilab-rl：`9bdf5e882c77b67496a2af3bbfd314edde532ceb`。
- Torch 2.8.0+cu128、MuJoCo/MuJoCo-Warp 3.11.0、Warp 1.16.0；seed 42。
- 规模：8192 environments × 5000 iterations × 40 rollout steps，共
  **1,638,400,000 environment steps**，未缩小默认规模。
- 开始：2026-09-08 16:20:36 +08:00；最终 checkpoint 修改时间：22:25:34 +08:00。
- logger 训练耗时：**06:02:33**；两端时间之差约 6 小时 5 分钟，包含初始化等开销。
- 启动检查时 worker 显存约 13984 MiB；未采集全程显存峰值。

远端产物均保留在原位置：

```text
/home/pc823/ws/unilabsim/wuji_unilab/full-training.log
/home/pc823/ws/unilabsim/wuji_unilab/logs/WujiHand_Reorient/2026-09-08_16-20-39_mjwarp/
  run_config.json
  model_4999.pt
  events.out.tfevents.*
```

`model_4999.pt` 大小 11,010,549 bytes，SHA-256：

```text
53d213fe1d52a428ea9560755ac00d07a80b7c9478f0877b6426b395a35e67e8
```

恢复状态：`iter=4999`、`step_counter=200000`、`difficulty=0.0`、
`adaptive=1.0`、policy observation 207、critic observation 413、action 20。

## 训练统计

下表从训练日志及 TensorBoard scalar 原始记录读取。最后 100 轮列是轮次标量的
算术均值，不是按 episode 数量加权的统计，也不是独立评估结果。

| 统计 | 最后一轮 | 最后 100 轮均值 |
| --- | ---: | ---: |
| `Train/mean_reward` | 418.2285 | 422.4571 |
| `Episode_Metrics/success` | 0.5202 | 0.5556 |
| `Episode_Metrics/orientation_error` (rad) | 0.5620 | 0.5521 |
| `Metrics/reorient/orientation_error` (rad) | 0.4863 | 0.4811 |
| `Perf/total_fps` (env steps/s) | 74225 | 74564.31 |

最终 mean episode length 为 990.42 控制步，按 `step_dt=0.05s` 折合约
49.52 秒。最终 `Episode_Metrics/survival=24.8974` 是 episode 内已运行秒数
的逐步均值，不能当作完整 episode 寿命。

**`success` 不是成功率。** 当前 owner 将每个 episode 的 `goal_count` 以
`last` 归约后记录在这个名字下；0.5202 表示该训练汇总中的完成目标次数均值，
不能写成 52.02% 成功率。`Episode_Metrics/orientation_error` 是 episode 内
误差均值，也不是每个目标的最小误差。日志中的 termination 数值同样不直接
构成按统一试验分母计算的掉落率。

这是提交 `2ea17bb` 的历史记录。后续指标对齐修改不改写该产物；新训练将这个标量改名为
`Episode_Metrics/goal_reach_count`，并采用与源仓库相同的 mean reduction，避免继续把目标计数
误称为 success。独立 trial 成功率仍需要专用评估协议计算。

训练初始 mean reward 为 -245.4667，episode orientation error 为 2.2151 rad，
最后值改善明显，证明该运行发生了学习。最终 `difficulty=0.0` 而 `adaptive=1.0`，
不能据此声称完成全部课程难度或完整 SO(3) 质量验收。需要独立目标分布评估和
课程状态审计才能进一步解释这组结果。

## 与 wuji-mjlab 公开结果的关系

原仓库的 [v2026.5.29 发布说明](https://github.com/wuji-technology/wuji-mjlab/releases/tag/v2026.5.29)
公开报告了以下 checkpoint 结果：

| 指标 | 上游发布报告 | 本次 UniLab 运行 |
| --- | --- | --- |
| Sim2sim 成功率 | 100%，100 trials | 未运行同协议评估 |
| Sim2real 成功率 | 85%，20 trials | 未做实机试验 |
| 平均最小姿态误差 | 4.1°（约 0.07156 rad） | 训练日志是逐步均值，不能直接比较 |
| 完整默认训练 | 文档默认 8192 env、5000 iterations | 已完成同样规模，约 6 小时 |

以上是上游自行公开的结果，本次没有独立复现其发布 checkpoint，也没有发现
随 release 附带的逐 trial JSON 或完整训练曲线。当前源仓库 eval 默认采用
100 trials、14 秒超时、姿态误差小于 0.2 rad 连续 5 个控制步、目标相对当前
姿态至少旋转 90°；成功后继续当前场景，掉落或超时才 reset。其 eval 使用
CPU MuJoCo + ONNX，与本项目 mjwarp 带 DR 的训练统计不是同一协议。

因此可确认“默认规模训练完成并产生可加载策略”，尚不能确认“达到上游公开的
100% 成功率或 4.1° 最小误差”。同协议 mjwarp eval、多 seed 评估、旧 checkpoint
适配及实机验证仍需分别完成；不以训练 reward 替代这些验收。

## 复查方式

在 `pc823` 项目目录：

```bash
tail -90 full-training.log
uv run --no-sync tensorboard --logdir logs/WujiHand_Reorient --host 127.0.0.1
sha256sum logs/WujiHand_Reorient/2026-09-08_16-20-39_mjwarp/model_4999.pt
```

使用 README 的 `wuji-play`、`wuji-export` 命令可以针对最终 checkpoint 回放和
导出。它们提供回放/策略一致性检查，不自动生成上述上游 trial 协议的成功率报告。

本次核查未发布 PyPI 包、未上传模型、未修改训练 checkpoint。
