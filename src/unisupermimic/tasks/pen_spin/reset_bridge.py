"""Pen-only bindings for model fields absent from UniLab's Entity API.

The source requires geom_pos and actuator_forcerange as well as joint qacc.
Startup publication is deliberately cold: extra per-world Warp allocations
are installed once and every CUDA graph is recaptured before training starts.
No existing Hand2 task or installed backend is modified.
"""

from functools import wraps

import numpy as np
import torch

from unisupermimic.tasks.dex_reorient.hand2.reset_bridge import (
    bind_mocap_reset,
    write_mocap_reset,
)


def get_pen_physics_model(env):
    """Cold compiled metadata, after UniSim's visual-geometry removal.

    get_scene_model_file points at the authoring XML, whose geometry columns
    differ from the actual executor. UniSim currently exposes no public host
    compiled-model handle for MJWarp, so bind it here and nowhere in task math.
    """
    backend = env.backend
    if backend.backend_type == "mjwarp":
        return backend._cpu_model
    if backend.backend_type == "mujoco":
        return backend.model
    raise ValueError(f"unsupported Pen backend {backend.backend_type!r}")


def publish_startup_model(env, fields: dict[str, torch.Tensor]) -> None:
    """Publish complete validated source model tables, once at startup."""
    backend = env.backend
    values = {name: value.detach().cpu().numpy() for name, value in fields.items()}
    for name, value in values.items():
        if value.shape[0] != env.num_envs or not np.isfinite(value).all():
            raise ValueError(f"Pen startup model field {name!r} is not finite/per-world")
    if backend.backend_type == "mujoco":
        pool = backend._pool
        if pool is None:
            raise RuntimeError("Pen startup requires a materialized MuJoCo model pool")
        # mjbatch exposes writable per-world model views. These fields include
        # marker positions and current limits, unavailable on Entity.
        for name, value in values.items():
            target = pool.expand(name)
            target[:] = value.reshape(target.shape)
        pool.set_const()
        pool.forward()
        return
    if backend.backend_type != "mjwarp":
        raise ValueError(f"unsupported Pen backend {backend.backend_type!r}")

    model, warp = backend._device_model, backend._warp
    was_captured = backend._cuda_graph_enabled
    replaced = False
    if was_captured:
        backend._disable_cuda_graphs("Pen startup model allocation")
    for name, value in values.items():
        target = getattr(model, name)
        # Warp shapes omit the trailing vector dimensions; construct using
        # the field's actual dtype and element shape, preserving vec2/vec3.
        if target.shape[0] == 1 and env.num_envs != 1:
            target = warp.array(
                shape=(env.num_envs, *target.shape[1:]),
                dtype=target.dtype, device=target.device,
            )
            setattr(model, name, target)
            replaced = True
        if target.shape[0] != env.num_envs:
            raise RuntimeError(f"Pen model field {name} lacks per-world support")
        target.assign(np.ascontiguousarray(value, dtype=np.float32))
        mirror = getattr(backend, f"_dr_{name}", None)
        if mirror is not None:
            mirror[:] = value.reshape(mirror.shape)
        backend._model_torch_views.pop(name, None)
        backend._dr_mirror_stale.discard(name)
    backend._mujoco_warp.set_const(model, backend._device_data)
    backend._mujoco_warp.forward(model, backend._device_data)
    backend._synchronize()
    if was_captured or replaced:
        backend._initialize_cuda_graphs(backend._device_data.qpos.device)


def _physics_tensor(env, name):
    backend = env.backend
    if backend.backend_type == "mjwarp":
        value = backend._warp.to_torch(getattr(backend._device_data, name))
    elif backend.backend_type == "mujoco":
        existing = {
            "qpos": "_qpos_view", "qvel": "_qvel_view",
            "qacc_warmstart": "_warm_view", "sensordata": "_sensor_data",
        }
        if name in existing:
            value = torch.from_numpy(getattr(backend, existing[name]))
        else:
            value = torch.from_numpy(backend._pool.bind(name, backend._np_dtype))
    else:
        raise ValueError(f"unsupported Pen backend {backend.backend_type!r}")
    return value.to(device=env.device, dtype=torch.float32)


def read_joint_acceleration(env, dof_indices) -> torch.Tensor:
    """Read measured final-physics qacc without a finite-difference proxy."""
    return _physics_tensor(env, "qacc")[:, tuple(dof_indices)]


def bind_physics_state(env) -> None:
    """Bind derived CPU qacc before reset's forward fills all tracked views."""
    if env.backend.backend_type == "mujoco":
        env.backend._pool.bind("qacc", env.backend._np_dtype)


def bind_cpu_observation_phase(env) -> None:
    """Publish final CPU body poses at the source's observation boundary.

    MuJoCo's packed bridge refreshes tracked poses only when it can aggregate
    every tracked body sensor. Pen's subset uses individual sensor reads,
    which otherwise keep pre-integration poses. Refresh only the derived
    body channels here, after rewards/terminations and before observations;
    a full forward would also change the source physics-label phase.
    """
    if env.backend.backend_type != "mujoco" or getattr(
        env, "_pen_cpu_observation_phase_bound", False
    ):
        return
    compute_observations = env._compute_observations

    @wraps(compute_observations)
    def compute_final_observations(*args, **kwargs):
        env.backend._sync_tracked_body_state()
        env.scene.read_plan.refresh()
        return compute_observations(*args, **kwargs)

    env._compute_observations = compute_final_observations
    env._pen_cpu_observation_phase_bound = True


def physics_nan_mask(env) -> torch.Tensor:
    """The pinned mjlab NanGuard channels, without altering invalid values."""
    invalid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for name in ("qpos", "qvel", "qacc", "qacc_warmstart", "sensordata"):
        invalid |= ~torch.isfinite(_physics_tensor(env, name)).all(-1)
    return invalid


__all__ = [
    "bind_mocap_reset", "write_mocap_reset", "publish_startup_model",
    "read_joint_acceleration",
    "get_pen_physics_model",
    "physics_nan_mask",
    "bind_physics_state",
    "bind_cpu_observation_phase",
]
