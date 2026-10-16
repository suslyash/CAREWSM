# Stage 7 Final Freeze Dossier — TASK-007A

## Status and boundary

TASK-007A is a freeze/preflight task only. Stage 7 is active; Final Test is not
authorized. No production training, inference, performance evaluation, Test
loader iteration, Test metric inspection, model/source/cache/script/dependency
change, tuning, or method reselection occurred.

The manager-frozen compared-method set is exactly:

1. frozen temporal audio baseline;
2. full optimized R4 trial012;
3. equal-parameter shared-fusion trial012.

The frozen seed set is `42, 43, 44, 45, 46`. Seeds 42–44 are accepted evidence
and were not rerun. This task creates only seed45/46 configuration clones.

## Frozen accepted seed42–44 identities

### Audio

Configs: `audio/00_frozen_baseline.yaml`,
`audio/01_frozen_baseline_seed43.yaml`, and
`audio/02_frozen_baseline_seed44.yaml`.

| seed | DEV D/P/Mean | selected checkpoint SHA256 |
|---:|---|---|
| 42 | 0.7479183895 / 0.8277353635 / 0.7878268765 | `0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2` |
| 43 | 0.708699 / 0.811771 / 0.760235 | `1a65f8dd605a6d942bed828f126e79a1f02930dd578176ba31f49c4dc10899dd` |
| 44 | 0.734396 / 0.809382 / 0.771889 | `7832d20a5431666365e2214423633f2be87bb27f37406d5a5dda8adda21e9852` |

The frozen audio feature contract remains the canonical
`/media/maxim/Databases/WSM_NEW/features` cache, with feature dimension 768,
and the new configs retain the reference data parameters exactly.

### Full R4 trial012

Configs: `fusion/37_r4_ra_stch_optuna_selected_seed42.yaml`,
`fusion/38_r4_ra_stch_optuna_confirm_seed43.yaml`, and
`fusion/39_r4_ra_stch_optuna_confirm_seed44.yaml`.

| seed | DEV D/P/Mean | selected checkpoint SHA256 |
|---:|---|---|
| 42 | 0.759025 / 0.879259 / 0.8191424538 | `104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a` |
| 43 | 0.718527 / 0.804363 / 0.761445 | `6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e` |
| 44 | 0.728217 / 0.832060 / 0.780139 | `b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17` |

### Shared fusion

Configs: `ablations/55_shared_fusion_seed42.yaml`,
`ablations/56_shared_fusion_seed43.yaml`, and
`ablations/57_shared_fusion_seed44.yaml`.

| seed | selected epoch | DEV D/P/Mean | run identity | checkpoint SHA256 |
|---:|---:|---|---|---|
| 42 | 11 | 0.765324 / 0.883606 / 0.824465 | `stage6_shared_fusion_trial012_seed42_2026-09-28_22-44_wsm_av_r3_disease_query_model_5db549e8` | `21504702976a960ffea02a68778ebc3bf7e6cc15ea8c9866f182ff04cbd30785` |
| 43 | 10 | 0.724496 / 0.818152 / 0.771324 | `stage6_shared_fusion_trial012_seed43_2026-09-28_22-51_wsm_av_r3_disease_query_model_454bf80b` | `ec746e519e551d69d59954dffff0a73968fb4ab555d43020b7722e635cc477ee` |
| 44 | 11 | 0.731448 / 0.838372 / 0.784910 | `stage6_shared_fusion_trial012_seed44_2026-09-28_22-57_wsm_av_r3_disease_query_model_a398d7d8` | `215e3a61c5dcb8ff7dd92ed5ca9336d66405c3f1ba6cc045841f08ec990d10a8` |

All existing identities above are historical accepted evidence. They were
audited from committed PROGRESS/history and were not executed in this task.

## New seed45/46 config freeze

The six new configs are:

- `audio/03_final_seed45.yaml`, `audio/04_final_seed46.yaml`;
- `fusion/40_r4_trial012_final_seed45.yaml`,
  `fusion/41_r4_trial012_final_seed46.yaml`;
- `fusion/42_shared_fusion_final_seed45.yaml`,
  `fusion/43_shared_fusion_final_seed46.yaml`.

Programmatic YAML equivalence passed for every pair. After removing only the
top-level `seed` and `experiment_info.params.run_name`, each new file is
identical to its required reference. The required run names and seeds are
present. All six pass `chimera-ml validate-config`.

Every new config retains `dev/mean_score` as the max-mode checkpoint and
early-stopping selector. Test protocol configuration remains present because
the project contract requires epoch-level monitoring, but no Test loader was
iterated and no Test value was inspected in TASK-007A.

## Cache and component preflight

The semantic pseudo cache remains:

`/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt`

Its SHA256 is
`17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`.
Frozen coverage is TRAIN rows 6325, accepted missing depression 376/2665
(all positive), and accepted missing Parkinson 1801/3660 (212 positive,
1589 negative). No cache was generated or modified.

All six registered DataModules, models, losses, and callbacks instantiated
without training. Audio trainable parameters were 3,033,416. Full R4 and
shared fusion each retained exactly 295,239 trainable parameters. Shape-only
TRAIN and DEV batch acquisition passed once for the audio family and once for
the fusion family. No optimizer was stepped and no metric was calculated.

## Frozen statistical plan

For each method, D Score, P Score, and Mean_Score will be summarized over
seeds42–46 by arithmetic mean, sample standard deviation (`ddof=1`),
minimum, maximum, range, and the descriptive two-sided 95% Student-t interval

`mean +/- 2.7764451051977987 * s / sqrt(5)`.

Same-seed paired deltas will be computed for full R4 minus audio, shared
fusion minus audio, and shared fusion minus full R4. Each five-value D/P/Mean
delta vector will report its values, mean, sample standard deviation,
Mean wins/ties/losses, and the same descriptive interval. No significance
winner is predeclared.

For every selected checkpoint, DEV-only Brier and ECE-15 will be reported per
task and method, with five-seed mean, sample standard deviation, and the same
95% interval. No threshold fitting or recalibration is authorized.

## Final Test firewall and next order

Final Test is not authorized in TASK-007A. New-seed Test values must not be
inspected, transcribed, compared, or used. All fifteen selected checkpoint
identities must be frozen before the separate final-Test task reads one final
TEST_NONE/SOFT/HARD result per checkpoint. No model, config, threshold, or
method change is allowed after that task begins.

If this freeze passes manager review, the exact next production order is:

1. audio seed45;
2. full R4 seed45;
3. shared fusion seed45;
4. audio seed46;
5. full R4 seed46;
6. shared fusion seed46.

No retry for metric improvement, extra seed, or post-firewall config change is
authorized.
