# VERIFY-001 Post-Cleanup Environment and Real CUDA Smoke

## Outcome

The post-CLEANUP-001 audio-only pipeline completed exactly one real CUDA training epoch using the canonical `emotionmsp_embeds/mlp_fold0.yaml` configuration copied to a temporary `/tmp` YAML. The canonical YAML, implementation code, dependencies, and TASK-011 status were not modified.

This was an integration smoke only, not scientific model evidence or authorization to resume TASK-011.

## Configuration and source

- Baseline/source commit: `13396c7f5a3a24092a1521a56326f2135bf89707`
- Canonical configuration: `configs/audio_feature_selection/emotionmsp_embeds/mlp_fold0.yaml`
- Temporary configuration: `/tmp/carewsm-verify001.DULkmR/config.yaml`
- Temporary changes only: `train.params.epochs: 1`; `experiment_info.params.run_name: verify001_post_cleanup_20261105`
- Resolved run: `verify001_post_cleanup_20261105_2026-11-05_13-07_carewsm_pooled_mlp_fff7b11a`
- CLI: `.venv/bin/chimera-ml train -c /tmp/carewsm-verify001.DULkmR/config.yaml`

## Environment

- Python 3.12.3
- PyTorch 2.11.0+cu128; PyTorch CUDA 12.8
- NumPy 2.5.4
- `mamba-ssm` 2.3.2.post1
- CUDA available: true
- GPU: NVIDIA GeForce RTX 4080; driver 580.178.04
- Active imports for NumPy, SciPy, pandas, scikit-learn, PyTorch, MLflow, Chimera-ML, and Mamba: passed
- `pip check`: one pre-existing environment inconsistency reported: `virtualenv 21.14.10` requires `packaging>=26.3`, while `packaging 24.2` is installed. This did not affect active scientific imports or the smoke.
- No automatic package reinstall or version change was performed.

## Regression verification

- Focused suite: **18 passed, 5 warnings**.
- `bash scripts/run_audio_experiments.sh --check --all`: all 110 canonical configurations passed.
- Native Chimera validation of `configs/audio_feature_selection/emotionmsp_embeds/mlp_fold0.yaml`: valid.
- The NumPy 2.5.4 environment imported the active scientific stack successfully.

## Real CUDA smoke

Exactly one epoch completed without NaN, Inf, mask, dependency, or CUDA runtime errors.

| Population | Samples |
|---|---:|
| train | 6998 |
| val_care | 352 |
| val_wsm | 1429 |
| test_care | 318 |
| test_wsm | 1364 |

All six classification metrics were calculated for both Depression and Parkinson on validation and TEST: Macro F1, Weighted F1, Macro Recall, Weighted Recall, Patient Recall, and Healthy Recall.

Validation monitor components:

- CARE Depression Macro Recall: `0.5921`
- CARE Parkinson Macro Recall: `0.5000`
- WSM Depression Macro Recall: `0.7312`
- WSM Parkinson Macro Recall: `0.5087`
- `dev_mean_macro_recall = (0.5921 + 0.5000 + 0.7312 + 0.5087) / 4 = 0.5830`

The run emitted `[carewsm_combined epoch 1]`. Early stopping and checkpoint selection used `dev_mean_macro_recall`; TEST values were logged descriptively and did not contribute to selection.

## Artifacts and risks

The run produced `train.log`, `summary.txt`, `checkpoints/last.pt`, and `checkpoints/epoch=1_dev_mean_macro_recall=0.5830.pt`. `summary.txt` contains exactly one completed epoch data row. Runtime logs, checkpoints, temporary YAML, code snapshot, caches, and data remain uncommitted.

The only environment risk found is the `pip check` packaging mismatch noted above. No active pipeline compatibility blocker was observed. Results are integration evidence only and must not be treated as final scientific performance.
