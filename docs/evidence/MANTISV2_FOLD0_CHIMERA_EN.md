# MantisV2 fold-0 Chimera integration

## Scope

This branch starts from `origin/main` at the requested clean base. The rejected
`scripts/run_mantisv2_fold0_scored_pilot.py` is absent. MantisV2 is represented
as a frozen `mantisv2_egemaps` feature family; classifier training remains the
existing Chimera-ML pipeline.

## Frozen representation

The generator is `src/audio/features/mantisv2_egemaps.py`. It reuses the accepted
Policy C temporalizer, pins `mantis-tsfm==1.1.0`, checkpoint revision
`99fe0f548960e272fbfa4b82fd9b5b5956779dfd`, transformer layer 2, combined token,
and `MantisTrainer.transform(three_dim=True)`. It uses 25 canonical eGeMAPS LLD
channels, 1.0-second windows, 0.5-second hop, at least 95 consecutive valid
frames, no-extrapolation interpolation to 512 points, at most two windows per
recording, and mean pooling to a flattened 12,800-dimensional artifact.

Each cache manifest records the checkpoint and source contract, Policy C
exclusions, split, corpus, sample identity, and artifact path. Labels remain in
the existing Dataset/DataModule join and are not passed to Mantis.

## Chimera integration

`configs/audio_representation_pilot/mantisv2/fold0.yaml` uses the registered
`carewsm_joint`, `carewsm_pooled_mlp`, masked BCE, six-metric component,
validation monitor, standard checkpoint/early-stopping, summary, console/file,
and MLflow components. Selection is `val_mean_corpora_tasks/macro_recall` and
CARE/WSM validation and TEST loaders remain separate.

## Verification

- New YAML: `./.venv/bin/chimera-ml validate-config -c configs/audio_representation_pilot/mantisv2/fold0.yaml` — PASS.
- Touched Python compilation — PASS.
- `git diff --check` — PASS.
- The pinned checkpoint SHA-256 matched the accepted fingerprint.
- The full real cache build was started with the verified Mantis package and
  CUDA environment but stopped before completion because the current accepted
  CSV temporalizer is CPU-bound over 8,436 fixed fold-0 TRAIN/VAL/TEST rows.
  The incomplete external cache was removed; no Chimera training result,
  coverage, MLflow run, checkpoint, or TEST metric is claimed.
- Ruff/Black were unavailable in the local environment.
