# Wuji UniLab development

- Roadmap: https://github.com/unilabsim/wuji_unilab/issues/1.
- Use current UniLab dependencies and only mjwarp physics. Missing capabilities
  must fail explicitly; never switch backends, omit features, or fabricate data.
- Assets are committed here per maintainer instruction; do not upload them to
  Hugging Face. Preserve asset provenance, LICENSE and NOTICE.
- Configuration and tensor layout may differ from mjlab. Validate task function,
  learning quality, checkpoint/observation consistency, and training capabilities.
- UniLab owns environments/adapters; UniSim owns physics capabilities; uni_rl
  owns runners/learners/logging and must not import UniLab. Consume public APIs.
- Keep owner YAML authoritative and scripts thin. Parse assets only on cold paths.
- Use rg, apply_patch, and uv run. Preserve unrelated work. No PyPI publication,
  release tags, or changes/merges to any upstream main branch.
- Base child branches and PRs on the roadmap integration branch. Run focused
  tests, make check, make test, and make test-all before creating/updating PRs.
- Record findings and decisions in issues; pending decisions stop only dependent
  work. Use independent read-heavy agents for exploration, probes, and review;
  maintain one writer per overlapping file set.
