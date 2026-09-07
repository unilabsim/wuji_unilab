# Bounded Capability Evidence

Parent roadmap: https://github.com/unilabsim/wuji_unilab/issues/1

This report keeps the approved static matrix in `baseline.json` unchanged. It
records bounded evidence obtained against the listed current dependency line.
It does not certify full mjlab coverage, source equivalence, learning quality,
multi-GPU behavior, or hardware deployment.

## C06 Native Sensors

The native-sensor probe used source Wuji hand/cube assets, current MuJoCo
3.11.0, MuJoCo-Warp 3.11.0, Warp 1.16.0 and unisim-core 1.1.3 on one RTX 4090.
It constructed seven logical contact groups, 20 `actuatorfrc` sensors and five
site `framelinvel` sensors in cold-path MJCF. It constructed a two-world
`mjwarp` backend through `unilab.base.backend_factory.create_backend`, bound
the named values through the public backend sensor view, and stepped 105
physics steps.

Result: 107 native sensors and 197 scalar values were finite and shape-valid.
Near/far cube worlds produced contact-found peaks `[1, 0]`; an induced fingertip
case also produced `[1, 0]`. The maximum actuator force was 0.0545899 and the
maximum site velocity was 0.0979699. This is evidence that current contact,
force and site velocity can be composed without a new generic sensor facade.
It does not cover all self-contact geometries, substep history, air-time, or
the engine's known CAPSULE x MESH single-contact limitation.

Reproduction from the migration environment:

```text
uv run --no-sync --project /home/user/ws/unilabsim/UniLab \
  --with mujoco-warp==3.11.0 --with warp-lang==1.16.0 \
  python /tmp/wuji-c06-probe/probe.py
```

The probe and JSON result are ephemeral `/tmp` artifacts. The downstream task
now has durable GPU tests; their commands and results belong to PR #9.

## C11--C15 Bounded Training

CPU contract probes used torch 2.8.0+cu128, rsl-rl 5.0.1 and uni_rl 1.1.0 with
eight fake environments, eight rollout steps and two iterations. Standard MLP,
GRU actor/critic and dotted Softplus distribution all updated finite parameters,
round-tripped actor/optimizer/iteration state and matched ONNX Runtime output
within 1.2e-7. This establishes public class-resolution and algorithm paths,
not MJWarp training quality.

The default RSL wrapper failed RND initialization with
`AttributeError: RslRlVecEnvWrapper has no attribute unwrapped`, because rsl-rl
needs `unwrapped.step_dt`. A metadata-only probe wrapper succeeded. The upstream
owner response is unilab-rl PR #17; it supplies explicit runner/state extension
without exposing the underlying environment.

The downstream GPU smoke in PR #9 subsequently ran 32 environments, 3 PPO
iterations and 8 rollout steps with the 207-dimensional policy, 413-dimensional
critic and 20 actions. It emitted finite logged action standard deviation and
checkpoints. Resuming `model_2.pt` completed the following iteration through
the explicit training-state provider. The reward is still dominated by early
cage drops; no success-rate or policy-quality claim is made.

## Confirmed Gaps And Decisions

| Capability | Status | Effect |
| --- | --- | --- |
| C08 post-substep sensor history | gap | Not used by the current Wuji reward; remains a general audit gap. |
| C09 DR field mutation | implemented in UniSim PR #42 | Required geometry/contact/DOF writes and primitive bound recomputation; task pins the reviewed commit pending PR review. |
| C10 fixed hand pose | implemented in UniSim #42 / UniLab #1540 | Selected-world mocap binding; no free-joint substitution. |
| C14 curriculum resume | implemented in unilab-rl #17 / UniLab #1540 | Explicit versioned owner state; no silent restart. |
| C15 RND wrapper | confirmed gap | Not a Wuji prerequisite; #17 provides the extension needed for a future owner adapter. |
| C19 camera/terrain and C17 IK | audit-only gap | Not silently enabled or replaced; Wuji uses a flat, vector-observation task. |

All results are tied to non-main reviewed feature SHAs listed in PR #9. They
must be rerun after any dependency update or before claiming a new support
level.
