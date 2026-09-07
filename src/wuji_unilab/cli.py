"""Strict Wuji command routing; training remains in the upstream owner."""

import argparse
import runpy
import sys

from wuji_unilab.config import CONF_ROOT, TASK_OWNERS
from wuji_unilab.tasks.reorient import register_resolvers


def _run(mode):
    parser = argparse.ArgumentParser(prog=f"wuji-{mode}")
    parser.add_argument("--task", choices=TASK_OWNERS, default="WujiHand_Reorient")
    parser.add_argument("--sim", choices=["mjwarp"], default="mjwarp")
    parser.add_argument("--checkpoint-file")
    args, overrides = parser.parse_known_args()
    for value in overrides:
        key = value.lstrip("+~").split("=", 1)[0]
        if key in ("task", "training.sim_backend", "training.task_name", "algo.algo"):
            parser.error(f"Cannot override task/backend routing via {key}")
    if mode == "play" and not args.checkpoint_file:
        parser.error("--checkpoint-file is required for play")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Wuji mjwarp requires CUDA; CPU fallback is not supported")
    register_resolvers()
    generated = [f"task={TASK_OWNERS[args.task]}/mjwarp"]
    if args.checkpoint_file:
        generated += ["algo.resume=true", f"algo.resume_path={args.checkpoint_file}"]
    if mode == "play":
        generated += ["training.play_only=true"]
    sys.argv = ["wuji-" + mode, "--config-dir", str(CONF_ROOT), *generated, *overrides]
    runpy.run_module("unilab.scripts.train_rsl_rl", run_name="__main__")


def train_main():
    _run("train")


def play_main():
    _run("play")


def list_main():
    for task in TASK_OWNERS:
        print(f"{task}\tmjwarp")
