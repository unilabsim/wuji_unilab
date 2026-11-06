"""Exercise MJWarp publication interfaces with real Warp CPU allocations.

These tests verify shapes, dtypes, aliases, and graph lifecycle dispatch. The
executor owner and recomputation calls are spies; no CUDA execution is implied.
"""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from unisupermimic.tasks.pen_spin.randomization import MODEL_FIELDS, default_model_fields
from unisupermimic.tasks.pen_spin.reset_bridge import publish_startup_model
from unisupermimic.tasks.pen_spin.scene import materialize_scene


@pytest.mark.parametrize("was_captured", [True, False])
def test_warp_startup_publication_preserves_real_array_types_and_recaptures(was_captured):
    wp = pytest.importorskip("warp")
    mjw = pytest.importorskip("mujoco_warp")
    cpu_model = mujoco.MjModel.from_xml_path(str(materialize_scene()))
    count = 3
    missing_world_axis = {"geom_pos", "actuator_forcerange"}
    # Let installed mujoco-warp provide every field's actual Warp vector or
    # scalar dtype and element shape. These are CPU allocations only.
    with wp.ScopedDevice("cpu"):
        model = mjw.put_model(cpu_model, batch_sizes={
            name: count for name in MODEL_FIELDS if name not in missing_world_axis
        })
        qpos = wp.array(
            np.arange(count * cpu_model.nq, dtype=np.float32).reshape(count, cpu_model.nq),
            dtype=wp.float32, device="cpu",
        )
    data = SimpleNamespace(qpos=qpos)
    original = {name: getattr(model, name) for name in MODEL_FIELDS}
    dtype_shapes = {name: (value.dtype, value.shape[1:]) for name, value in original.items()}
    for name, value in original.items():
        assert value.shape[0] == (1 if name in missing_world_axis else count)

    fields = default_model_fields(cpu_model, count, "cpu")
    world_scale = torch.tensor((1.0, 1.01, 1.02))
    for name, value in fields.items():
        if name != "body_iquat":
            value *= world_scale.reshape(count, *(1 for _ in value.shape[1:]))
    expected = {name: value.numpy().copy() for name, value in fields.items()}
    events = []
    allocations = []
    qpos_before = qpos.numpy().copy()
    qpos_pointer = qpos.ptr

    def allocate(**kwargs):
        events.append("allocate")
        allocations.append(kwargs)
        return wp.array(**kwargs)

    unrelated_view = object()
    backend = SimpleNamespace(
        backend_type="mjwarp", _device_model=model, _device_data=data,
        _warp=SimpleNamespace(array=allocate), _cuda_graph_enabled=was_captured,
        _model_torch_views={
            **{name: wp.to_torch(value) for name, value in original.items()},
            "unrelated": unrelated_view,
        },
        _dr_mirror_stale={*MODEL_FIELDS, "unrelated"},
    )
    # Match the native host mirrors. The two newly supported fields have no
    # native mirror; their pointer-backed cached views still need invalidation.
    mirrored_fields = set(MODEL_FIELDS) - missing_world_axis
    for name in mirrored_fields:
        setattr(backend, f"_dr_{name}", np.full(expected[name].shape, np.nan, dtype=np.float64))

    def disable(reason):
        assert reason == "Pen startup model allocation"
        events.append("disable")
        backend._cuda_graph_enabled = False

    def check_published(actual_model, actual_data):
        assert actual_model is model and actual_data is data
        for name, value in expected.items():
            np.testing.assert_array_equal(getattr(actual_model, name).numpy(), value)

    def set_const(actual_model, actual_data):
        events.append("set_const")
        check_published(actual_model, actual_data)
        assert not backend._cuda_graph_enabled

    def forward(actual_model, actual_data):
        events.append("forward")
        check_published(actual_model, actual_data)

    def synchronize():
        events.append("synchronize")

    def capture(device):
        events.append("capture")
        assert device == qpos.device
        backend._cuda_graph_enabled = True

    backend._disable_cuda_graphs = disable
    backend._mujoco_warp = SimpleNamespace(set_const=set_const, forward=forward)
    backend._synchronize = synchronize
    backend._initialize_cuda_graphs = capture
    publish_startup_model(SimpleNamespace(num_envs=count, backend=backend), fields)

    assert events == (["disable"] if was_captured else []) + [
        "allocate", "allocate", "set_const", "forward", "synchronize", "capture",
    ]
    assert len(allocations) == 2
    for name, value in original.items():
        published = getattr(model, name)
        dtype, element_shape = dtype_shapes[name]
        assert published.dtype is dtype
        assert published.shape == (count, *element_shape)
        assert published.device == value.device
        if name in missing_world_axis:
            assert published is not value
        else:
            assert published is value
        np.testing.assert_array_equal(published.numpy(), expected[name])
    for name, allocation in zip(("geom_pos", "actuator_forcerange"), allocations, strict=True):
        assert allocation["shape"] == (count, *dtype_shapes[name][1])
        assert allocation["dtype"] is dtype_shapes[name][0]
        assert allocation["device"] == original[name].device
    for name in mirrored_fields:
        np.testing.assert_array_equal(getattr(backend, f"_dr_{name}"), expected[name])
    assert backend._model_torch_views == {"unrelated": unrelated_view}
    assert backend._dr_mirror_stale == {"unrelated"}
    assert backend._device_data is data and data.qpos is qpos
    assert data.qpos.ptr == qpos_pointer
    np.testing.assert_array_equal(data.qpos.numpy(), qpos_before)
    assert backend._cuda_graph_enabled
