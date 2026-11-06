# Hand1 and PenSpin SuperMimic tasks

These tasks use the independent `unisupermimic.rl.algos.dex_supermimic`
learner adapted from [Dex-Supermimic](https://github.com/CumulusAlpha/Dex-Supermimic).
The existing Hand2 cube learner and Wuji task entry points are preserved.

Run the commands below from the repository root. The tasks use UniLab 1.5.1 and UniSim 2.1.1, matching the source environment.
UniLab 1.5.1 fixes world-to-local angular velocity conversion during packed
root resets; 1.5.0 does not satisfy the PenSpin reset velocity contract.

```bash
uv sync
uv run unisupermimic-hand1-train --cfg job
uv run unisupermimic-pen-spin-train --cfg job
```

## Assets and reference trajectories

Hand1 reuses the verified Hand1 assets already in `wuji_unilab/assets` and
the existing cube assets. PenSpin adds `unisupermimic/assets/objects/pen250`
and reuses the Hand2 hand scene with task-specific calibration.

PenSpin requires the upstream trajectory catalog for real reference tracking:

```bash
uv run unisupermimic-pen-spin-download
```

The downloader extracts the catalog from the pinned wuji-mjlab v2026.10.22
release and validates its per-clip SHA-256 values. Downloaded files live in
`src/unisupermimic/tasks/pen_spin/data/rawdata/clips/` and are ignored by Git.
The five recipe JSON files are versioned. A local transfer may already
contain the catalog; rerunning the command validates it without downloading.

## Training and playback

Both SuperMimic recipes default to MJWarp, 1,024 environments and 8,000
iterations, with `torch.compile` enabled. Training needs a CUDA device and
sufficient GPU memory. Choose separate run directories:

```bash
uv run unisupermimic-hand1-train training.log_dir=m_ends/hand1_reorient
uv run unisupermimic-pen-spin-train training.log_dir=m_ends/pen_spin
```

`algo.checkpoint=/absolute/path/to/model_N.pt` resumes a run;
`algo.iterations` is the final iteration number. The local run directories
above are ignored by Git. Existing checkpoints and training results are
not part of the transfer.

```bash
uv run unisupermimic-hand1-play --checkpoint /absolute/path/to/model_N.pt
uv run unisupermimic-pen-spin-play --checkpoint /absolute/path/to/model_N.pt
```

Both viewers support `--backend mujoco`, `--manual-reset` and
`--start-paused`. Hand1 displays the goal; PenSpin displays the reference.
Use `--help` for camera, speed and reference-data options. PenSpin also
retains a separate native PPO baseline (`unisupermimic-pen-spin-ppo`),
which is not used by the SuperMimic training command.

## Validation

```bash
uv run ruff check .
XDG_CACHE_HOME=/tmp/unisupermimic-cache uv run pytest -q
```

Default tests exclude `gpu` and `slow` markers. Tests that compare with
`references/Dex-Supermimic` skip when that optional checkout is absent.
Catalog-dependent tests skip when downloaded trajectories are absent;
synthetic test fixtures do not require them. These checks validate packaging
and basic contracts, not long-run learning quality.

`dex_supermimic_alignment.md` preserves the source workspace's technical
alignment notes and historical checks. Its recorded training results are
not new validation of this checkout. Task LICENSE/NOTICE files and the root
NOTICE retain upstream attribution.
