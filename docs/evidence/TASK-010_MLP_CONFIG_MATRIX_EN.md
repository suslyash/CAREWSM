# TASK-010 MLP configuration matrix

Status: PREPARED; static-only verification completed. No training, TEST evaluation, MLflow run, preprocessing, or log cleanup was performed.

## Exact matrix

- Temporal `T>1`: `egemaps` / `egemaps_lld` (25), `disvoice_prosody_dynamic` (13), and `disvoice_phonation_dynamic` (7). Each has five MPTSNet and five MD-Former YAMLs: 30 files total.
- Static native `T=1`: `egemaps_func` (88), `disvoice_prosody_static` (103), `disvoice_phonation_static` (28), `disvoice_articulation_static` (488), `wav2vec_xlsr300m` (1024), `trillsson4_1` (1024), `emotionmsp_trajectories` (3), and `emotionmsp_embeds` (1024). Each has five MLP and five MD-Former YAMLs: 80 files total.

The final matrix is 110 literal YAMLs. Static MD-Former is a comparative classifier only and must not be presented as evidence of temporal modeling. MLP is a pooled baseline and does not model time. Static MPTSNet YAMLs are absent from the launchable matrix.

Every MLP YAML uses `carewsm_pooled_mlp`, the verified input dimension, numeric fold 0--4, seed 5, full canonical CARE+WSM TRAIN/VAL settings, CUDA, standard console/file and MLflow logger declarations, snapshot and summary callbacks, and VAL-only `joint/average_macro_f1` checkpoint/early stopping with `mode: max`.

## Launcher guard

`scripts/run_audio_experiments.sh` remains the only launcher. It rejects unsupported pairs such as temporal+MLP and static+MPTSNet. `--check --all` validates all 110 files; static files are marked `STATIC_INPUT / TEMPORAL_INTERPRETATION_LIMITED`, while temporal files are marked `SEQUENCE_INPUT`.
