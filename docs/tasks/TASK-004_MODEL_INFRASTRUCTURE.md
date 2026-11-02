# TASK-004 — Three native audio sequence classifiers in Chimera-ML

## Goal
Implement the **actual three project classifier families** (not generic renamed Transformer/Mamba placeholders) as minimal reproducible Chimera-ML model components:
1. TSCMamba-inspired: temporal + spectral views and Mamba/SSM sequence modelling;
2. MPTSNet-inspired (AAAI 2025): FFT-derived periods, multiscale local convolution and global dependency modelling;
3. MD-Former-inspired (2025): multiscale dual temporal/cross-channel Transformer branches.

Use the official author repositories/papers wherever available to trace key blocks and configurations; document adaptations and any missing/unavailable author code. Preserve a small pooled MLP reference as baseline only, not a fourth research family.

## Preflight and scope
Read `AGENTS.md`, project contracts, `docs/MODEL_SHORTLIST.md`, TASK-002 and TASK-003 evidence and actual `chimera-ml==0.3.0` API. Start from newest `origin/main`; branch `codex/task-004-model-infrastructure`. Work in `src/audio/models/` (and minimal `src/audio/loss/`, `src/audio/metrics/`, `configs/audio/` as needed). Keep scope AUDIO ONLY.

TASK-004 is **model/loss/metrics infrastructure plus one synthetic batch forward/backward smoke**, not a full training experiment or feature screening. Do not trigger Mantis installation/checkpoint downloads or block native work waiting on Mantis. Native input is mandatory; later frozen Mantis sequences must be accepted through the same representation interface.

## Required model/data contract
- Register all three model builders with Chimera-ML model registry through the existing `src/chimera_plugin.py`. Config-driven `build_model` must instantiate each.
- Consume actual `carewsm_joint` loader batches (`inputs["native"]`, variable lengths, sequence masks, labels and observed-label masks). Verify real batch shape/keys; do not assume tensor format from memory. Handle padding safely. Different feature dimensions (egemaps vs embeddings) must be configurable.
- Each model: shared temporal representation -> **separate task-specific feature layers** -> independent Depression/Parkinson binary logits. Missing targets masked. No pseudo temporal expansion of static arrays.
- Implement native architectural mechanisms substantively: TSCMamba temporal/spectral multiview with actual Mamba path (error clearly if required Mamba dependency unavailable, never silently substitute a Transformer); MPTSNet explicit periodicity/FFT + multiscale CNN/dependency path; MD-Former dual branch with temporal + cross-channel structure. Keep the adaptations feasible for 25–100+ channel inputs and variable lengths; guard zero-length or all-padding sequences.
- Reference pooled MLP can be included for shape/ablations. Masked loss: BCEWithLogits per task only over observed labels, equal 0.5/0.5 combination. Optional positive weights are computed TRAIN-only, not hard-coded or derived from VAL/TEST. Leave advanced gradient conflict balancing (PCGrad/CAGrad/STCH/PAGB) to a later task unless trivial hooks.
- Implement/verify all six required **binary per-task** metrics: Macro F1, Weighted F1, Macro Recall, Weighted Recall, Patient Recall, Healthy Recall. Separate CARE/WSM metric computations and their arithmetic mean (Average) without pooling. Define zero-division and one-class behavior, count observed labels. Do not use TEST for model selection.
- Training callback/logger references from WSM: `mlflow_logger`, `console_file_logger`, `checkpoint_callback`, `snapshot_callback`, `early_stopping_callback`, `wsm_summary_callback`. Reuse standard Chimera components; don't duplicate them. If necessary, adapt the WSM summary callback later in a separate task.

## Verification / evidence
- Run one no-training real-data loader batch from registered `carewsm_joint`; then synthetic or single mini-batch forward + masked-loss backward for each available classifier. Check finite logits/gradients and mask invariance on padded frames and missing labels. If Mamba cannot be instantiated due missing official dependency, provide explicit BLOCKED evidence for **that model only**, not a fake replacement.
- Verify registration and YAML instantiation through Chimera-ML 0.3.0 and include a minimal runnable smoke command/config per model or one parameterized command.
- Brief focused checks only (no large test programme), `py_compile`, `git diff --check`. Evidence `docs/evidence/TASK-004_MODEL_INFRASTRUCTURE_EN.md`: official inspirations, key blocks, adaptations, available/blocked dependencies, count and shape checks, commands, no results claimed.
- Update `docs/PROGRESS_EN.md` and nothing else among owner/manager docs. No real fold training, no TEST metric optimisation, no model selection, no Optuna or broad Mantis cache.

## Git handoff
Commit + normal push; no PR/merge. Return eight heading handoff per `AGENTS.md`. Stop for manager review.

**Manager will review/merge and then open the next atomic task.**
