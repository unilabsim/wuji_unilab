# Wuji Hand 2 reorientation

This package is the first UniLab migration slice from `Dex-Supermimic`.
It registers `DexHand2Reorient` for the `mjwarp` and `mujoco` backends and
keeps the source task's 20 joint order, 50 Hz control loop, EMA position
controller, random SO(3) cube goals, reset bridge, collision capsule overlay,
and cage-drop termination.

The scene is materialized once from the repository-local
`third_party/wuji-description` Hand 2 MJCF and the cube assets under
`src/unisupermimic/assets/objects/inhand_cube`.  The generated XML is cached
under `XDG_CACHE_HOME` and validated as `(nq, nv, nu, nmocap) = (27, 26, 20, 1)`.

Run a CPU construction smoke check with:

```bash
XDG_CACHE_HOME=/tmp/unisupermimic-cache \
  uv run python -c \
  'from unisupermimic.tasks.dex_reorient.hand2.scene import materialize_scene; print(materialize_scene())'
```

For native policy training, use the package's rsl_rl PPO entrypoint:

```bash
uv run unisupermimic-hand2-ppo \
  algo.num_envs=32 algo.max_iterations=3 algo.num_steps_per_env=8
```

The dedicated Hand 2 SuperMimic world-model learner remains a separate slice;
run it through the task-specific entrypoint rather than the G1 motion-library
command:

```bash
XDG_CACHE_HOME=/tmp/unisupermimic-cache \
  uv run unisupermimic-hand2-train \
  algo.num_envs=1024 training.sim_backend=mujoco
```

The learner owns the Hand 2 replay schema, 58D state/46D context transition,
100D controller memory, and 207D actor history.  A one-step contract check is
available with `uv run python tools/smoke_hand2_env.py --step`.
