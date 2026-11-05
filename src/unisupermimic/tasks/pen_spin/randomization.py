# SPDX-License-Identifier: Apache-2.0
# Adapted from Dex-Supermimic pen_spin/config/wuji_hand2/env_cfgs.py and
# mjlab v1.6.0 envs/mdp/dr (0fb8a681136be94ffc636a3dd423cabb97d91f10).
"""The source production Pen Spin startup randomization.

All samples are immutable per-world model parameters. Model publication lives
in reset_bridge, since UniLab does not yet expose marker positions or effort
limits on Entity. The completed cube tasks do not use this adapter.
"""

from __future__ import annotations

import math
import re

import numpy as np
import torch
from unilab.managers import ManagerTermBase

from .inertia import (
    _build_perturbation_U,
    _cholesky_4x4,
    _decompose_pseudo_inertia_J,
    _reconstruct_pseudo_inertia_J,
)
from .reset_bridge import get_pen_physics_model, publish_startup_model

MODEL_FIELDS = (
    "body_mass", "body_ipos", "body_inertia", "body_iquat",
    "geom_friction", "geom_solref", "geom_solimp", "geom_size",
    "geom_pos", "geom_rbound", "geom_aabb",
    "actuator_gainprm", "actuator_biasprm", "actuator_forcerange",
    "dof_armature", "dof_damping", "dof_frictionloss",
)


def default_model_fields(model, num_envs, device):
    result = {}
    for name in MODEL_FIELDS:
        value = np.array(getattr(model, name), dtype=np.float32)
        if name == "geom_aabb":
            value = value.reshape(model.ngeom, 2, 3)
        result[name] = torch.tensor(value, device=device).expand(num_envs, *value.shape).clone()
    return result


def sample_pseudo_inertia(defaults, body_ids, alpha_range, d_range=(0.0, 0.0)):
    """The pinned mjlab pseudo_inertia math with explicit tensor bindings."""
    mass, ipos, inertia, iquat = (
        defaults[name][:, body_ids]
        for name in ("body_mass", "body_ipos", "body_inertia", "body_iquat")
    )
    shape, device = mass.shape, mass.device

    def sample(bounds):
        return torch.rand(shape, device=device) * (bounds[1] - bounds[0]) + bounds[0]

    factor = _cholesky_4x4(_reconstruct_pseudo_inertia_J(mass, ipos, inertia, iquat))
    alpha = sample(alpha_range)
    stretch = [sample(d_range) for _ in range(3)]
    shear_and_translation = [sample((0.0, 0.0)) for _ in range(6)]
    perturbation = _build_perturbation_U(alpha, *stretch, *shear_and_translation)
    transformed = perturbation @ factor
    return _decompose_pseudo_inertia_J(transformed @ transformed.mT)


def randomize_pen_dimensions(fields, model, diameter_range=(.014, .016), length_range=(.205, .265)):
    """Exact source uniform geometry draw and uniform-volume inertia rebuild."""
    n, device = fields["body_mass"].shape[0], fields["body_mass"].device
    shaft, bottom, top = [model.geom(name).id for name in (
        "object/cube", "object/A_bottom", "object/A_top"
    )]
    body = model.body("object/cube").id
    radius = torch.empty(n, device=device).uniform_(*diameter_range) / 2
    length = torch.empty_like(radius).uniform_(*length_range)
    shaft_length = length - .072
    end_center = length / 2 - .018
    fields["geom_size"][:, shaft, 0] = radius
    fields["geom_size"][:, shaft, 1] = shaft_length / 2
    fields["geom_pos"][:, bottom, 0] = -end_center
    fields["geom_pos"][:, top, 0] = end_center
    fields["geom_rbound"][:, shaft] = torch.sqrt(radius.square() + (shaft_length / 2).square())
    fields["geom_aabb"][:, shaft, 0] = 0
    fields["geom_aabb"][:, shaft, 1] = torch.stack((radius, radius, shaft_length / 2), -1)
    mass = fields["body_mass"][:, body]
    shaft_volume = torch.pi * radius.square() * shaft_length
    box_volume = .018 * .018 * .036
    density = mass / (shaft_volume + 2 * box_volume)
    shaft_mass, box_mass = density * shaft_volume, density * box_volume
    axial = shaft_mass * radius.square() / 2 + 2 * box_mass * (.018**2 + .018**2) / 12
    transverse = shaft_mass * (3 * radius.square() + shaft_length.square()) / 12
    transverse += 2 * box_mass * ((.018**2 + .036**2) / 12 + end_center.square())
    fields["body_inertia"][:, body] = torch.stack((axial, transverse, transverse), -1)
    fields["body_iquat"][:, body] = torch.tensor((1., 0., 0., 0.), device=device)


class SourcePenRandomization(ManagerTermBase):
    """Publish the source startup events once; never resample on reset."""

    def __init__(self, cfg, env):
        del cfg
        super().__init__(env)
        self._applied = False

    def __call__(self, env, env_ids=None):
        del env_ids
        if self._applied:
            raise RuntimeError("Pen physics randomization is a startup-only event")
        model = get_pen_physics_model(env)
        n, device = env.num_envs, env.device
        defaults = default_model_fields(model, n, device)
        fields = {name: value.clone() for name, value in defaults.items()}
        obj = [model.body("object/cube").id]
        hand = [model.body(f"robot/{name}").id for name in (
            "right_palm_link",
            *(f"right_finger{f}_link{j}" for f in range(1, 6) for j in range(1, 5)),
        )]
        contact = [g for g in range(model.ngeom) if re.fullmatch(
            r"robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)",
            model.geom(g).name,
        )]
        joints = [model.joint(name).id for name in env.scene["robot"].joint_names]
        dofs = model.jnt_dofadr[joints].tolist()
        actuators = [model.actuator(f"{name}_actuator").id for name in env.scene["robot"].joint_names]

        def uniform(bounds, shape):
            return torch.rand(shape, device=device) * (bounds[1] - bounds[0]) + bounds[0]

        def log_uniform(bounds, shape):
            return uniform(tuple(math.log(x) for x in bounds), shape).exp()

        # Preserve the source order, including the default-relative semantics
        # of pinned mjlab.pseudo_inertia: its subsequent four-field write uses
        # default ipos, rather than silently introducing a different COM rule.
        fields["body_ipos"][:, obj, 0] += uniform((-.04, .04), (n, 1))
        fields["body_ipos"][:, obj, 1:] += uniform((-.0005, .0005), (n, 1, 2))
        fields["geom_friction"][:, contact, 0] = log_uniform((.35, 1.4), (n, len(contact)))
        fields["geom_friction"][:, contact, 1] *= log_uniform((.5, 2.), (n, len(contact)))
        for name, component, bounds in (
            ("geom_solref", 0, (1., 2.)), ("geom_solref", 1, (.8, 1.2)),
            ("geom_solimp", 2, (1., 2.)), ("geom_solimp", 0, (.5, 1.)),
        ):
            fields[name][:, contact, component] *= uniform(bounds, (n, 1))
        values = sample_pseudo_inertia(defaults, obj, (.5 * math.log(.4), .5 * math.log(1.6)))
        for name, value in zip(("body_mass", "body_ipos", "body_inertia", "body_iquat"), values, strict=True):
            fields[name][:, obj] = value
        kp_scale = log_uniform((.9, 1.1), (n, len(actuators)))
        kd_scale = log_uniform((.8, 1.1), (n, len(actuators)))
        fields["actuator_gainprm"][:, actuators, 0] *= kp_scale
        fields["actuator_biasprm"][:, actuators, 1] *= kp_scale
        fields["actuator_biasprm"][:, actuators, 2] *= kd_scale
        bias = uniform((-.01, .01), (n, len(joints)))
        values = sample_pseudo_inertia(defaults, hand, (-.458, .203), (-.2, .2))
        for name, value in zip(("body_mass", "body_ipos", "body_inertia", "body_iquat"), values, strict=True):
            fields[name][:, hand] = value
        fields["dof_armature"][:, dofs] *= log_uniform((.5, 2.), (n, len(dofs)))
        fields["dof_frictionloss"][:, dofs] *= uniform((.5, 1.5), (n, len(dofs)))
        fields["actuator_forcerange"][:, actuators] *= uniform((.7, 1.05), (n, len(actuators), 1))
        randomize_pen_dimensions(fields, model)
        publish_startup_model(env, fields)
        env.scene["robot"].data.encoder_bias_tensor.copy_(bias)
        env._pen_default_model_fields = defaults
        env._pen_model_fields = fields
        env._pen_physics_model = model
        self._applied = True


__all__ = ["SourcePenRandomization", "default_model_fields", "randomize_pen_dimensions"]
