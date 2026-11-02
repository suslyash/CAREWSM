# TASK-005 — First native Chimera-ML training pipeline

## Outcome
A real, config-driven Chimera-ML 0.3.0 fold-0 audio-only smoke completed with joint CARE+WSM native `egemaps_lld`, MPTSNet, unweighted masked multitask BCE, and one epoch/one bounded batch. This is pipeline evidence only, not a comparative result.

## Exact environment and command
- Branch source: `origin/main` at `5ac2a3f`.
- Python: `.venv/bin/python`; Chimera-ML import: `.venv/lib/python3.12/site-packages/chimera_ml`.
- Command: `PYTHONPATH=.:src .venv/bin/chimera-ml train -c configs/audio/task005_native_fold0_smoke.yaml`.
- Config validation: `PYTHONPATH=.:src .venv/bin/chimera-ml validate-config -c configs/audio/task005_native_fold0_smoke.yaml`.

## Data and split
The registered `carewsm_joint` DataModule constructed fold 0 from CARE fold-0 TRAIN + WSM fold-0 TRAIN and matching VAL subsets; DEV was not added and TEST was not used. Full view counts before smoke limits were TRAIN 6,998, VAL 1,780, CARE VAL 352, WSM VAL 1,428, CARE TEST 318, WSM TEST 1,364. The selected family was valid `egemaps_lld`, with 25 channels.

The smoke config limits datasets to 8 TRAIN and 8 rows per VAL view. These are deterministic bounded execution limits, not a dataset filter for research.

## Sequence policy
Observed trajectories reached 15,301 frames. The smoke-only bounded policy pools contiguous bins over the complete trajectory to at most 256 steps using per-bin means. It never selects only the first 32 frames and never discards the remainder of a video. Normal training leaves `max_train_samples`/`max_val_samples` unset and retains native trajectories unchanged.

## Training and validation observations
The completed run was `mptsnet-egemaps-lld_2026-11-02_13-25_carewsm_mptsnet_e9fa57ac`. It performed one train batch (8 samples), then joint, CARE and WSM VAL loaders (8 rows each).

- TRAIN loss: 0.6332; observed count: 8.
- Joint VAL loss: 0.6257; observed count: 8; Macro F1 0.5000; Weighted F1 1.0000; Macro Recall 0.5000; Weighted Recall 1.0000; Patient Recall 0.0000; Healthy Recall 1.0000.
- CARE VAL loss: 0.6257; observed count: 8; same six metric values as joint in this bounded slice.
- WSM VAL loss: 0.5943; observed count: 0 for the selected task metric instances in this first deterministic slice; metric values are therefore 0.0 and are not interpretable as performance.

The trainer instantiated all 12 metric objects: six metrics with `task: 0` and six with `task: 1`. The standard logs expose the metric names/counts; this smoke is intentionally not a scientific ranking.

## Logger and callback evidence
The standard Chimera `console_file_logger`, `checkpoint_callback`, `snapshot_callback`, and `early_stopping_callback` ran. VAL checkpoint monitoring used `joint/loss`; checkpoint and `last.pt` were written under the run directory, and snapshot `code.zip` plus resolved YAML were written.

The standard MLflow logger was initially attempted but could not initialize in the environment because installed MLflow/SQLAlchemy versions are incompatible (`FallbackAsyncAdaptedQueuePool` import error). It is omitted from the final smoke config and recorded as a dependency blocker; no duplicate logger was implemented.

## Limitations and blockers
- This is one bounded epoch and one batch, not valid model selection or feature screening.
- The first WSM VAL slice has no observed label for the selected task instances; larger validation evaluation is required before interpretation.
- TSCMamba remains explicitly blocked without `mamba_ssm`; no substitute was used.
- TEST loaders were constructed by the DataModule but not evaluated in this smoke, preserving TEST discipline.

## Artifacts
Run artifacts are generated under `logs/task005/.../mptsnet-egemaps-lld_2026-11-02_13-25_carewsm_mptsnet_e9fa57ac/` (ignored runtime output): `train.log`, `checkpoints/epoch=1_joint_loss=0.6257.pt`, `checkpoints/last.pt`, snapshot `code.zip`, and resolved YAML.
