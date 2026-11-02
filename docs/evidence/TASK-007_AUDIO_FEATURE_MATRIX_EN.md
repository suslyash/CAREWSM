# TASK-007 audio feature matrix

Status: PREPARED; no training, smoke, MLflow run or log cleanup was executed.

## Verified native matrix

The dimensions below were read from the canonical registry and verified with the existing `JointDataModule.load_native` loader on a real VALID artifact. All values were finite `float32` arrays.

| config folder | registry `feature_family` | verified shape | native temporal semantics | YAMLs |
|---|---|---:|---|---:|
| `egemaps` | `egemaps_lld` | `(10364, 25)` | frame-in-turn sequence with timestamps | 10 |
| `egemaps_func` | `egemaps_func` | `(1, 88)` | turn-level pooled vector | 10 |
| `disvoice_prosody_static` | `disvoice_prosody_static` | `(1, 103)` | turn-level pooled vector | 10 |
| `disvoice_prosody_dynamic` | `disvoice_prosody_dynamic` | `(279, 13)` | timestamped segment sequence | 10 |
| `disvoice_phonation_static` | `disvoice_phonation_static` | `(1, 28)` | turn-level pooled vector | 10 |
| `disvoice_phonation_dynamic` | `disvoice_phonation_dynamic` | `(2438, 7)` | timestamped frame sequence | 10 |
| `disvoice_articulation_static` | `disvoice_articulation_static` | `(1, 488)` | turn-level pooled vector | 10 |
| `wav2vec_xlsr300m` | `wav2vec_xlsr300m` | `(1, 1024)` | turn embedding | 10 |
| `trillsson4_1` | `trillsson4_1` | `(1, 1024)` | turn embedding | 10 |
| `emotionmsp_trajectories` | `emotionmsp_trajectories` | `(1, 3)` | turn-level trajectory artifact | 10 |
| `emotionmsp_embeds` | `emotionmsp_embeds` | `(1, 1024)` | turn embedding | 10 |

Static and turn-level families remain native length-1 inputs; no artificial time axis or Mantis representation is introduced. MPTSNet and MD-Former accept the common native `[time, input_dim]` adapter for all verified dimensions.

## Configuration layout

Each supported folder contains literal files `mptsnet_fold0.yaml` through `mptsnet_fold4.yaml` and `mdformer_fold0.yaml` through `mdformer_fold4.yaml`. Every YAML explicitly records fold, registry family, model name, input dimension, `joint/average_macro_f1`, `mode: max`, standard loggers, MLflow, checkpoint, snapshot, early stopping and summary callback.

Logs use the standard Chimera layout rooted at `logs/audio/<feature>/`; no monkey-patching, generation, movement or renaming is used.

## Static verification

- all 110 YAML files parse and validate through Chimera-ML config validation;
- every folder has both models and folds 0–4;
- `bash -n scripts/run_audio_experiments.sh` passes;
- Python syntax and `git diff --check` pass;
- no experiments, MLflow runs, preprocessing or cleanup were executed.


## TASK-008 preflight guard

The single launcher now supports a read-only `--check` mode. It validates every selected literal YAML for fold 0--4, model registry name, feature registry name, input dimension, fixed CARE/WSM roots, CUDA device, standard loggers, and VAL-only checkpoint/early-stopping callbacks monitoring `joint/average_macro_f1` in `max` mode. It does not train, create MLflow runs, evaluate TEST, generate YAML, or touch logs.

The following native families are explicitly marked `STATIC_INPUT / TEMPORAL_INTERPRETATION_LIMITED` by preflight: `egemaps_func`, `disvoice_prosody_static`, `disvoice_phonation_static`, `disvoice_articulation_static`, `wav2vec_xlsr300m`, `trillsson4_1`, `emotionmsp_trajectories`, and `emotionmsp_embeds`. Their native shape is `T=1`; classifier results are exploratory and must not be interpreted as validation of periodic folding or temporal attention mechanisms. `egemaps_lld`, `disvoice_prosody_dynamic`, and `disvoice_phonation_dynamic` retain sequence semantics and are reported as `SEQUENCE_INPUT`.

Static checks completed for `--check --feature egemaps`, one static feature, and `--check --all` (110 configs); no numerical experiment evidence was produced.
