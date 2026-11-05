"""Publish fixed startup randomization and bridge selected mocap resets.

UniLab owns the public ``env_ids`` reset contract and observation history.
The cold-start publisher installs complete per-world model tables and derived
geometry bounds once; the mocap bridge writes selected poses during resets.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Protocol

import numpy as np
import torch
from unisim.backend.base import MocapPoseBinding
from unisim.backend.mjwarp.randomization import PrimitiveGeomBounds


def publish_fixed_model(env, bindings, samples, gain_ids) -> None:
    """Publish source startup DR without a mocap-incompatible row reset.

    UniLab may discard visual geoms while materializing physics. Resolve all
    columns against the executor model and Entity bindings, then refresh the
    source primitive bounds before publishing complete per-world tables.
    """
    import mujoco

    from unisupermimic.tasks.pen_spin.reset_bridge import publish_startup_model

    backend = env.backend
    model = backend._cpu_model if backend.backend_type == "mjwarp" else backend._model
    field_map = {
        "friction": "geom_friction", "hand_size": "geom_size", "cube_size": "geom_size",
        "solref_palm": "geom_solref", "solref_fingers": "geom_solref",
        "solimp_palm": "geom_solimp", "solimp_fingers": "geom_solimp",
        "cube_mass": "body_mass", "hand_mass": "body_mass",
        "cube_inertia": "body_inertia", "hand_inertia": "body_inertia",
        "cube_com": "body_ipos", "armature": "dof_armature",
    }
    names = set(field_map.values()) | {"actuator_gainprm", "actuator_biasprm", "geom_rbound", "geom_aabb"}
    defaults = {name: np.array(getattr(model, name), dtype=np.float32) for name in names}
    defaults["geom_aabb"] = defaults["geom_aabb"].reshape(model.ngeom, 2, 3)
    fields = {
        name: torch.tensor(value, device=env.device).expand(env.num_envs, *value.shape).clone()
        for name, value in defaults.items()
    }
    for name, (owner, local_ids, _, _, selector) in bindings.items():
        if selector == "geom_ids":
            columns = np.asarray(owner._geom_ids)[local_ids]
        elif selector == "body_ids":
            columns = np.asarray(owner._body_ids)[local_ids]
        else:
            columns = np.asarray(backend.get_joint_state_qvel_indices(owner.joint_names))[local_ids]
        fields[field_map[name]][:, columns] = samples[name]
    act_ids = np.asarray(env.scene["robot"]._actuator_ids)[gain_ids]
    fields["actuator_gainprm"][:, act_ids, 0] = samples["kp"]
    fields["actuator_biasprm"][:, act_ids, 1] = -samples["kp"]
    fields["actuator_biasprm"][:, act_ids, 2] = -samples["kd"]
    sizes = fields["geom_size"].detach().cpu().numpy()
    rbound, aabb = PrimitiveGeomBounds(model.geom_type, mujoco.mjtGeom).compute(
        sizes,
        np.broadcast_to(defaults["geom_size"], sizes.shape),
        fields["geom_rbound"].detach().cpu().numpy(),
        fields["geom_aabb"].detach().cpu().numpy(),
    )
    fields["geom_rbound"] = torch.tensor(rbound, device=env.device)
    fields["geom_aabb"] = torch.tensor(aabb, device=env.device)
    publish_startup_model(env, fields)


class ResetEntity(Protocol):
    """The subset of Entity used by these bridges."""

    @property
    def _reset_state(self) -> Any: ...  # upstream-owned transaction storage


def _legacy_mujoco_binding(backend, body_name: str) -> MocapPoseBinding:
    """Bind the legacy MJCF pool missing from UniSim's entity-only API."""
    if backend.backend_type != "mujoco" or backend._entity_layout is not None:
        raise NotImplementedError(f"No Hand1 mocap compatibility binding for {backend.backend_type}")
    model = backend.model
    body_id = model.body(body_name).id
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise ValueError(f"Hand1 wrist {body_name!r} must be a mocap body")
    buffers = {}

    def views():
        pool = backend._pool
        if pool is None:
            raise RuntimeError("Hand1 mocap writes require a materialized MuJoCo pool")
        if buffers.get("pool") is not pool:
            buffers.update(pool=pool, pos=pool.bind("mocap_pos", backend._np_dtype),
                           quat=pool.bind("mocap_quat", backend._np_dtype))
        return pool, buffers["pos"], buffers["quat"]

    def reader():
        _, position, quaternion = views()
        return np.concatenate((position[:, mocap_id], quaternion[:, mocap_id]), axis=1)

    def writer(rows, poses):
        pool, position, quaternion = views()
        position[rows, mocap_id] = poses[:, :3]
        quaternion[rows, mocap_id] = poses[:, 3:]
        pool.forward(np.sort(rows))
        backend._tracked_body_state_dirty[rows] = False

    binding = MocapPoseBinding(
        backend_type="mujoco", body_name=body_name, num_envs=backend.num_envs,
        default_pose=np.concatenate((model.body_pos[body_id], model.body_quat[body_id])),
        _reader=reader, _writer=writer,
    )
    # PackedResetTransaction calls the same backend cache after committing
    # qpos/qvel. Keep this legacy API adaptation local to Hand1.
    backend._mocap_pose_bindings[body_name] = binding
    return binding


def _deferred_mocap_writes(reset_state):
    """Supply the missing MJWarp packed-mocap commit hook on this instance."""
    pending = getattr(reset_state, "_hand1_pending_mocap", None)
    if pending is not None:
        return pending
    pending = {}
    reset_state._hand1_pending_mocap = pending

    def after_commit(original):
        @wraps(original)
        def commit(*args, **kwargs):
            try:
                result = original(*args, **kwargs)
                # Ordinary state reset restores default mocap. Publish wrist
                # poses afterward, before Manager refreshes its read plan.
                for raw, rows, poses in tuple(pending.values()):
                    raw.write(rows.cpu().numpy(), poses.cpu().numpy())
                return result
            finally:
                pending.clear()
        return commit

    reset_state.commit = after_commit(reset_state.commit)
    reset_state.commit_tensor = after_commit(reset_state.commit_tensor)
    original_abort = reset_state.abort

    @wraps(original_abort)
    def abort():
        pending.clear()
        return original_abort()

    reset_state.abort = abort
    return pending


@dataclass
class _MocapResetBinding:
    raw: MocapPoseBinding
    reset_state: Any
    term_name: str

    @property
    def default_pose(self):
        return self.raw.default_pose


def bind_mocap_reset(entity: ResetEntity, body_name: str, *, term_name: str):
    """Bind a wrist reset, retaining the normal selected-row transaction."""
    reset_state = entity._reset_state
    if reset_state is None:
        raise RuntimeError("Wuji reset requires an active reset transaction")
    try:
        raw = reset_state.bind_mocap_pose(body_name)
    except NotImplementedError:
        raw = _legacy_mujoco_binding(reset_state._backend, body_name)
        reset_state._mocap_bindings[body_name] = raw
    if not reset_state._backend.get_tensor_capabilities().packed_reset_mocap:
        _deferred_mocap_writes(reset_state)
    return _MocapResetBinding(raw, reset_state, term_name)


def write_mocap_reset(binding, env_ids: torch.Tensor, poses: torch.Tensor) -> None:
    """Stage selected wrist poses after the ordinary generalized-state reset."""
    transaction, raw = binding.reset_state, binding.raw
    if raw.backend_type == "mujoco":
        # mjbatch must bind its input buffers before the reset forward.
        raw.read()
    selected = poses.index_select(0, env_ids.to(poses.device)).detach().clone()
    if transaction._backend.get_tensor_capabilities().packed_reset_mocap:
        transaction.write_mocap_pose(raw.body_name, env_ids, selected, term_name=binding.term_name)
        return
    transaction._require_active()
    rows = transaction._validate_ids(env_ids, capability="mocap pose")
    if transaction._tensor_active:
        outside = not bool(transaction._tensor_active_mask[rows].all())
    else:
        outside = not bool(torch.isin(rows, transaction._begin_ids).all())
    if outside:
        raise ValueError(f"{binding.term_name}: mocap mutation outside the active reset")
    transaction._validate_values(selected, shape=(rows.numel(), 7), capability="mocap pose",
                                 term_name=binding.term_name)
    transaction._validate_quaternions(selected[:, 3:], term_name=binding.term_name)
    _deferred_mocap_writes(transaction)[raw.body_name] = (raw, rows.detach().clone(), selected)
