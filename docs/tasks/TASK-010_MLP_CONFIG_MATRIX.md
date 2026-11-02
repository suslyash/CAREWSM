# TASK-010 — MLP only on non-temporal (T=1) audio features

## Goal and exact model matrix
Correct the previous TASK-010 scope. **MLP is only for native static / turn-level features with T=1**. Do not add MLP configs for temporal families. The desired experiment matrix is:

- **Temporal (T>1)**: `egemaps` (registry `egemaps_lld`), `disvoice_prosody_dynamic`, `disvoice_phonation_dynamic` — **MPTSNet and MD-Former only**, each on five folds. Retain their 30 existing YAMLs unchanged.
- **Static (T=1)**: `egemaps_func`, `disvoice_prosody_static`, `disvoice_phonation_static`, `disvoice_articulation_static`, `wav2vec_xlsr300m`, `trillsson4_1`, `emotionmsp_trajectories`, `emotionmsp_embeds` — **MLP as primary baseline, MD-Former optionally for comparison**, each on folds 0–4. Keep the 40 existing MD-Former YAMLs.
- **Remove the existing 40 MPTSNet YAMLs on static T=1 features** from the launchable configuration matrix; they have no valid temporal/periodicity interpretation. Instead add precisely **40 new literal MLP YAMLs** to those eight folders, `mlp_fold0.yaml` … `mlp_fold4.yaml`. End state **110 YAMLs**, not 165. Do not run MPTSNet on T=1.
- MD-Former on T=1 is only a static-feature comparison; **never** present its result as evidence of temporal modeling.

## Implementation
1. Inspect `AGENTS.md`, TASK-007/008 evidence and actual `carewsm_pooled_mlp` registration and feature dimensions. Use only proven registry names and dimensions.
2. Each MLP YAML is explicit: numeric fold 0–4, correct feature family and `input_dim`, `carewsm_pooled_mlp`, fixed seed 5, full canonical CARE+WSM TRAIN/VAL, observed-label masking, existing optimizer/trainer defaults, `logs/audio/<feature>/`, console/file + MLflow loggers, snapshot and summary, VAL-only checkpoint/early stopping on `joint/average_macro_f1` with `mode: max`.
3. Preserve the **single Bash launcher** `scripts/run_audio_experiments.sh`; no Python launcher or dynamic YAML generation. `--list`, `--check`, `--feature`, `--model`, `--all` must reflect exactly the allowed matrix. `--feature egemaps_func --model mlp` selects five explicit configs; `--all` only 110 allowed configs. Explicit unsupported pairs (`egemaps --model mlp`, `egemaps_func --model mptsnet`) must be rejected with a clear error, not silently ignored.
4. Update any hardcoded read-only preflight model mapping and check exactly 11 folders × 2 eligible models × five folds. Clearly label static `T=1`, temporal `T>1`; no claim that MLP models time.
5. Make only necessary changes. Do not change existing MD-Former and temporal configurations, model internals or canonical splits unless there is a verified blocker.

## Verification / handoff
- YAML parse/config validation, precise 110-file/family-model-fold matrix check, Bash `-n`, preflight `--check --all`, `--check --feature egemaps_func --model mlp`; verify that forbidden model-feature combinations reject. `git diff --check`.
- **No training, smoke, MLflow run, TEST, preprocessing or log deletion**. Owner manually runs experiments.
- Update `scripts/runs/README.md`, `docs/evidence/TASK-010_MLP_CONFIG_MATRIX_EN.md`, `docs/PROGRESS_EN.md`.
- Implement on `codex/task-010-mlp-configs`, commit + normal push; no PR/merge; eight-heading handoff and stop.
