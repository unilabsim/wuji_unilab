import json

import pytest

from wuji_unilab.export import _parser, build_policy_schema


def test_export_requires_checkpoint_and_fixed_output_contract(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr("sys.argv", ["wuji-export", "--checkpoint-file", str(checkpoint)])
    args = _parser().parse_args()
    assert args.task == "WujiHand_Reorient"
    assert args.max_diff == pytest.approx(1e-4)
    monkeypatch.setattr("sys.argv", ["wuji-export"])
    with pytest.raises(SystemExit):
        _parser().parse_args()


def test_schema_preserves_deployment_contract(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload = build_policy_schema(
        task="WujiHand_Reorient",
        control_dt_s=0.05,
        observation_dim=207,
        action_dim=20,
        joint_names=["robot/joint1", "robot/joint2"],
        checkpoint_file=checkpoint,
        max_abs_diff=1e-6,
        mean_abs_diff=1e-7,
    )
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload))
    loaded = json.loads(path.read_text())
    assert loaded == {
        "schema": "wuji_unilab_policy",
        "version": 1,
        "task": "WujiHand_Reorient",
        "physics_backend": "mjwarp",
        "control_dt_s": 0.05,
        "observation": {"name": "obs", "shape": [1, 207]},
        "action": {"name": "actions", "shape": [1, 20]},
        "joint_names": ["robot/joint1", "robot/joint2"],
        "checkpoint": str(checkpoint.resolve()),
        "onnx_max_abs_diff": 1e-6,
        "onnx_mean_abs_diff": 1e-7,
    }


@pytest.mark.parametrize(
    ("observation_dim", "action_dim", "joint_names"),
    [(0, 20, ["j"]), (207, 0, ["j"]), (207, 20, [])],
)
def test_schema_rejects_incomplete_policy_contract(
    tmp_path, observation_dim, action_dim, joint_names
):
    with pytest.raises(ValueError):
        build_policy_schema(
            task="WujiHand_Reorient",
            control_dt_s=0.05,
            observation_dim=observation_dim,
            action_dim=action_dim,
            joint_names=joint_names,
            checkpoint_file=tmp_path / "model.pt",
            max_abs_diff=0.0,
            mean_abs_diff=0.0,
        )
