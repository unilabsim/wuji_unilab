# Approved delivery and validation

Roadmap: https://github.com/unilabsim/wuji_unilab/issues/1

The parent integration base is `dev/bootstrap` at
`0e391e0b33fca2982ab54d41b80e1e508b0b51fd`. Children target
`dev/issue-1-wuji-training-coverage`. No upstream main branch is changed and
no package is published. P0 through P5 are tracked in issues #2 through #7.

The source and upstream commits in baseline.json are immutable audit facts.
New runtime evidence must name the tested dependencies and distinguish a
baseline result from a downstream composition or an upstream patch.

Wuji assets are committed locally. Prefer current UniLab dependencies and
NumPy manager semantics. Source RNG, configuration, logic, tensor ordering and
discrete events need not be exactly reproduced. Task functionality, effective
training, reset isolation, checkpoint schema and export consistency remain
required. Missing functionality must be reported, never replaced by fallback.

## Initial validation protocol

Start with bounded CPU algorithm probes and 2–8-world GPU capability probes.
Verify native sensor values, physical randomization effects, masked reset and
parameter updates before spending a large training budget. The available local
GPU reports NVIDIA GeForce RTX 4090 and 49140 MiB memory; this is inventory,
not a throughput or memory-capacity promise.

After the task works, declare a learning budget and common SO(3) evaluation
protocol before observing the training results. Report success, goals/time,
drops, survival, total interactions, elapsed time and resource consumption.
Use independent seeds; source and target random streams need not match.
Reward curves alone are not comparable when reward scales differ. Hardware
control is separate from offline deployment validation.

## Decision ownership

Current public APIs are preferred over new protocols. Confirmed required gaps
in mocap/DR/curriculum restore are recorded in owner issues before extending
public contracts. Generic mjlab gaps (such as camera training or substep sensor
history) are audited independently and are not automatically prerequisites for
the vector-observation Wuji task. An unresolved decision blocks only dependent
implementation or acceptance, not the whole roadmap.

The bootstrap Makefile validates documentation JSON and whitespace only. P1
adds package, lint/type and executable tests; no simulation or learning pass is
claimed by the bootstrap checks.
