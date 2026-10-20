from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wuji_unilab.tasks.reorient.reset_bridge import (
    bind_mocap_reset,
    reset_backend,
    write_geom_sizes,
    write_mocap_reset,
)


class Binding:
    def __init__(self):
        self.writes = []

    def write(self, rows, poses):
        self.writes.append((np.asarray(rows), np.asarray(poses)))


def test_bind_mocap_reset_requires_active_transaction():
    entity = SimpleNamespace(_reset_state=None)
    with pytest.raises(RuntimeError, match="active reset transaction"):
        bind_mocap_reset(entity, "palm", term_name="test")


def test_reset_backend_requires_active_transaction():
    entity = SimpleNamespace(_reset_state=None)
    with pytest.raises(RuntimeError, match="active reset transaction"):
        reset_backend(entity)


def test_write_mocap_reset_publishes_only_selected_rows():
    binding = Binding()
    poses = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    write_mocap_reset(binding, torch.tensor([False, True]), poses)
    assert len(binding.writes) == 1
    rows, selected = binding.writes[0]
    np.testing.assert_allclose(rows, [1])
    torch.testing.assert_close(torch.as_tensor(selected[0]), poses[1])


def test_write_geom_sizes_replaces_selected_columns():
    applied = []
    backend = SimpleNamespace(
        _dr_geom_size=np.ones((2, 3, 3), dtype=np.float32),
        _prepare_reset_randomization=lambda rows, payload: ("prepared", payload),
        _apply_model_updates=applied.append,
    )
    values = torch.full((2, 1, 3), 2.0)
    write_geom_sizes(
        backend,
        model_ids=np.asarray([1]),
        backend_geom_ids=np.asarray([2]),
        reset_mask=torch.tensor([False, True]),
        values=values,
    )
    assert len(applied) == 1
    payload = applied[0][1].geom_size
    np.testing.assert_allclose(payload[0, 2], 1.0)
    np.testing.assert_allclose(payload[1, 2], 2.0)
