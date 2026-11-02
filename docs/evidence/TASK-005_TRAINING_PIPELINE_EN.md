# TASK-005 — Final native Chimera-ML training pipeline

## Outcome
A real bounded fold-0 audio-only Chimera-ML 0.3.0 run completed without `PYTHONPATH`: joint CARE+WSM native eGeMAPS LLD, MPTSNet, one epoch and one train/VAL batch. This remains pipeline smoke evidence, not comparative model evidence.

## Commands and environment
- Editable install: `.venv/bin/pip install -e .`
- Doctor: `.venv/bin/chimera-ml doctor`
- Validation: `.venv/bin/chimera-ml validate-config -c configs/audio/audio_mptsnet_native_egemaps_lld_joint_care_wsm_fold0_smoke.yaml`
- Run: `.venv/bin/chimera-ml train -c configs/audio/audio_mptsnet_native_egemaps_lld_joint_care_wsm_fold0_smoke.yaml`
- Chimera-ML 0.3.0, MLflow 2.22.5, SQLAlchemy 2.0.55.

`chimera-ml doctor` reports one discovered CAREWSM plugin and registered datamodule/model/loss/metrics/callbacks without warnings.

## Data and sequence policy
Fold 0 uses CARE fold-0 TRAIN + WSM fold-0 TRAIN and matching VAL subsets; DEV and TEST were not used. Full views are TRAIN 6,998, VAL 1,780, CARE VAL 352, WSM VAL 1,428, CARE TEST 318, WSM TEST 1,364. The smoke limits TRAIN and each VAL view to 8 deterministic rows.

Native `egemaps_lld` has 25 channels. Long trajectories are not truncated to 32 frames: bounded smoke uses contiguous mean pooling over the complete trajectory to at most 256 time steps. Normal training leaves these limits unset and retains native trajectories.

## Run evidence
Run: `smoke-seed5_2026-11-02_13-40_carewsm_mptsnet_03eec96c`.

- TRAIN: 8 samples, loss 0.6332; depression and Parkinson metric aliases are present in `train.log`.
- Joint VAL: loss 0.6257; both disease blocks expose all six metrics and observed counts.
- CARE VAL: both disease blocks expose all six metrics; observed count 8 for each in this slice.
- WSM VAL: Depression observed count 8; Parkinson observed count 0 and all Parkinson metrics are `nan`/UNDEFINED, never zero quality.
- Checkpoint/early stopping monitor only `joint/loss`.

The single registered `carewsm_multitask_prf` computes Depression and Parkinson separately, all six required metrics, corpus-specific blocks, arithmetic Average where both corpora are available, and observed counts.

## Logging and artifacts
Standard `console_file_logger`, `mlflow_logger`, `checkpoint_callback`, `snapshot_callback`, `early_stopping_callback`, and registered `carewsm_summary_callback` ran. The run directory contains `train.log`, `summary.txt`, resolved YAML, `code.zip`, and `checkpoints/epoch=1_joint_loss=0.6257.pt` plus `last.pt`.

MLflow local SQLite backend `sqlite:///logs/mlflow.db` contains a FINISHED run in experiment `audio-mptsnet-native-egemaps_lld-joint-care-wsm-fold0-smoke`; metrics and parameters are queryable through `mlflow.search_runs`.

`scripts/collect_summary.py --root logs/audio --output all_summary.txt` found and collected one complete run without modifying source logs.

## Limitations
This is a bounded one-epoch smoke and must not drive architecture/feature selection. TSCMamba remains blocked without `mamba_ssm`; no fallback was introduced.
