# TASK-006 — First real full-fold native audio experiment

## Objective
Move from 8-sample infrastructure smoke to **one scientifically interpretable full-fold experiment**: joint CARE+WSM fold-0 TRAIN/VAL, native `egemaps_lld`, MPTSNet (or pooled MLP only if MPTSNet has a demonstrated compute blocker). Keep audio-only.

This is **one configuration**, not a sweep, architecture ranking or feature-search campaign. Do not trigger Mantis or TSCMamba installation. Do not change the fixed folds.

## Method
- Use installed Chimera-ML 0.3.0, existing decorator-based registries, `carewsm_multitask_prf`, masked multitask BCE, standard loggers/callbacks and summary collector. Respect `AGENTS.md` formatting and WSM conventions.
- Consume **all eligible fold-0** CARE+WSM TRAIN and VAL rows. Remove `max_train_samples` and `max_val_samples` from experimental config; preserve separate CARE/WSM VAL views. Never add DEV independently.
- Choose a reproducible, **time-respecting** way of handling native sequences of 10k+ frames while keeping video-wide temporal evidence, sequence masks and original sample identities. Make the policy explicit in config and evidence and apply consistently across TRAIN/VAL. Do not use first-N-frame truncation; avoid inadvertently triggering arbitrary 256-step pooling just because sample limits are set. Any pooling/resampling must not learn from VAL/TEST. If any preprocessing learns statistics, fit on TRAIN only.
- Check full-dataset and per-task observed-label counts; ensure that both heads contribute where labels exist. Missing labels do not become negatives. Joint model learns both tasks; neither disease is silently skipped.
- Train one reasonably bounded fold-0 run on available GPU (prefer CUDA if supported), with epochs/early stopping/budget explicitly stated before launch in YAML. No large Optuna or exhaustive runs. Save VAL-selected checkpoint based **only on VAL**, never TEST.
- Log all six metrics for each disease on joint VAL and separately CARE/WSM VAL, plus mathematically defined CARE/WSM arithmetic Average only when each underlying metric has observed support; undefined remains undefined, not zero. Record sample/profile/observed counts and class support. Ensure metrics are not overwritten in Chimera logger, `summary.txt`, `all_summary.txt`, or MLflow; use meaningful audio/model/feature/fold run naming and tracked IDs.
- **Do not inspect or use TEST during model selection.** This first run may stop after full-fold VAL evidence; keep TEST evaluation for the subsequent fixed-checkpoint reporting task, with same checkpoint on both CARE and WSM TEST.
- Important known issue from TASK-005: the metric implementation currently may compute `Average` from `nan` for one corpus; fix narrowly if necessary. The standard six-metric report must be correct and task/corpus-separated. Do not change target semantics.
- Keep TSCMamba explicitly BLOCKED without `mamba_ssm`, no Transformer fallback.

## Deliverables and checks
- One **meaningful** `configs/audio/` YAML reflecting `audio`, architecture, `native`, `egemaps_lld`, joint CARE+WSM, fold 0, and full-fold purpose. Keep console/file + MLflow, checkpoint, snapshot, early stopping, summary callback.
- Minimal code adaptations in `src/audio/` only when needed for full-fold support and reasonable compute; no duplicate framework or generic refactor. No changes to feature source artifacts or canonical splits.
- `docs/evidence/TASK-006_FIRST_FULL_FOLD_EN.md` with exact command, resolved config, dataset and per-task/class counts, sequence policy, runtime/hardware, seed, epoch-level VAL evidence, selected checkpoint, metric validity, MLflow/run paths and limitations. Numeric metrics must originate from actual full-fold validation; if run is not completed, state BLOCKED with logs and do not invent results.
- One bounded real training attempt; no repeated debugging experiments or full five-fold sweep. `py_compile`, `git diff --check`, one concise end-to-end verification. Update `docs/PROGRESS_EN.md` only among management docs.
- Commit and normal push branch `codex/task-006-first-full-fold`; **no PR/merge**. Eight-heading handoff and stop.

## Follow-on
On manager PASS, proceed to fixed-checkpoint separate CARE/WSM TEST statistics and subsequent controlled native-vs-Mantis comparisons (once Mantis is installed and pinned), plus the remaining architectures.
