"""Physical state and exact controller state; rotations are palm-frame matrices."""

import torch

from .rotations import mm, so3_exp, so3_log

NUM_J = 20
STATE_DIM = 2 * NUM_J + 18
DELTA_DIM = 2 * NUM_J + 12


def unpack(x):
    q, qd, p, r, v, w = x.split((NUM_J, NUM_J, 3, 9, 3, 3), -1)
    return q, qd, p, r.reshape(*r.shape[:-1], 3, 3), v, w


def pack(q, qd, p, r, v, w):
    return torch.cat((q, qd, p, r.flatten(-2), v, w), -1)


def integrate(x, delta):
    q, qd, p, r, v, w = unpack(x)
    dq, dqd, dp, dr, dv, dw = delta.split((NUM_J, NUM_J, 3, 3, 3, 3), -1)
    return pack(q + dq, qd + dqd, p + dp, mm(r, so3_exp(dr)), v + dv, w + dw)


def difference(x, target):
    q, qd, p, r, v, w = unpack(x)
    qt, qdt, pt, rt, vt, wt = unpack(target)
    return torch.cat(
        (
            qt - q,
            qdt - qd,
            pt - p,
            so3_log(mm(r.transpose(-1, -2), rt)),
            vt - v,
            wt - w,
        ),
        -1,
    )


def ema_target(action, previous, default, lower, upper, bias, alpha, age, dt):
    raw = (default + 0.5 * action.clamp(-1, 1)).clamp(lower, upper)
    c = alpha * raw + (1 - alpha) * previous
    c = torch.where((age < round(0.4 / dt)).unsqueeze(-1), default, c)
    return c, c - bias


# Controller / command memory. All fields have a final feature dimension.
# c1/c2 and a1/a2 refer to the previous two control steps.
MEMORY_DIMS = {
    "c1": 20,
    "c2": 20,
    "a1": 20,
    "a2": 20,
    "goal": 9,
    "hold": 1,
    "goal_timer": 1,
    "window": 1,
    "window_timer": 1,
    "reward_window": 1,
    "reward_hold": 1,
    "reward_snapshot": 1,
    "within": 1,
    "cage": 1,
    "age": 1,
    "count": 1,
}


def flatten_memory(m):
    return torch.cat([m[k] for k in MEMORY_DIMS], -1)


def unflatten_memory(x):
    return dict(zip(MEMORY_DIMS, x.split(tuple(MEMORY_DIMS.values()), -1), strict=False))
