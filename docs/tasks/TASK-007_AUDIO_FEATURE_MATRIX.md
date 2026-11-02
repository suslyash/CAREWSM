# TASK-007 — Extend explicit audio feature-family experiment configurations

## Objective
Extend the TASK-006 **prepared, owner-run** experiment matrix beyond native eGeMAPS LLD. Keep the user's requested folder pattern `configs/audio/<feature>/`, **five literal YAML files per supported architecture**, and exactly **one** Bash launcher `scripts/run_audio_experiments.sh`. Do not run any experiments.

## Discovery first
Inspect the canonical feature registry, TASK-001 availability audit, native feature representation code and feature-family schemas. Determine actual fixed input dimensions and temporal/static nature of each of the 11 available audio feature families from verified project evidence. Do not infer dimensions from names or invent them. Identify which families can be consumed by the MPTSNet and MD-Former temporal models without changing scientific meaning. Keep a short feature-to-dimension mapping in evidence. If dimensionality or required temporal alignment is unavailable, mark that family BLOCKED and skip generating unsupported YAML.

## Configuration contract
- For each confirmed, model-compatible family create `configs/audio/<short-feature-name>/mptsnet_fold0.yaml` … `mptsnet_fold4.yaml` and analogous `mdformer_fold0.yaml` … `mdformer_fold4.yaml`. Literal `data.params.fold`, `data.params.feature_family`, model `input_dim`, valid paths; **no runtime YAML modification or generation**.
- Reuse TASK-006 settings, masked targets and one multitask PRF metric, VAL-only `joint/average_macro_f1` with mode max, standard loggers (including MLflow), callbacks and `summary.txt`, and `logs/audio/<feature>/` as log base. Make conservative changes only if a specific family needs distinct length handling; document it.
- Existing `configs/audio/egemaps/` ten YAMLs stay intact unless an independently verified defect requires minimal correction.
- Keep exactly one universal **Bash** launcher. Its `--list`, `--feature FEATURE`, optional `--model MODEL`, and `--all` discover actual folders/configs and validate complete fold sets; no hardcoded eGeMAPS-only `--all`. Do not loop seeds or create new Python or per-model launchers. No rewriting/moving logs or monkey-patching Chimera-ML.
- No TSCMamba architecture fallback; no Mantis download or guessed representation.
- TEST never used for tuning, training, or early stopping. No new test-per-epoch behavior.

## Verification and deliverables
- Static validate every new and existing YAML for parse, registry model name, true input dimensions, exactly folds 0–4, logging paths and supported feature family. Run `bash -n`, `--list` (non-training), `py_compile` when relevant, `git diff --check`.
- Update `scripts/runs/README.md`, `docs/evidence/TASK-007_AUDIO_FEATURE_MATRIX_EN.md` and `docs/PROGRESS_EN.md`.
- **Strict: do not execute Chimera training, smoke, MLflow, data preprocessing jobs, or clean logs.** Do not create numerical training results. User alone will launch the series.
- Work in `codex/task-007-audio-feature-configs`, commit + normal push. **No PR/merge**. Eight-heading handoff and stop.

This task extends configuration *coverage*, not architecture complexity or research training.
