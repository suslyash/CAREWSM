# Stage 7 Final DEV Evidence — TASK-007B

## Result and boundary

`STAGE-7 FIVE-SEED DEV CHECKPOINT FREEZE COMPLETE`

The recovered accepted temporal-audio seed42 checkpoint is present at the
historical path below and its SHA256 was verified. The existing adapter strict-
loaded the epoch-4 payload with the archived graph compatibility path, all
adapter parameters remained frozen, and evaluation mode was retained. The exact
TASK-004H AV evaluation path now reproduces the accepted DEV scores within
0.0005; the unified fifteen-checkpoint freeze is complete. No production run
or source/config/model change was authorized.

TASK-007B otherwise preserved the frozen method set (temporal audio, full R4
trial012, equal-parameter shared fusion) and seeds 42–46. The six authorized
runs completed once, in order: audio45, R4-45, shared45, audio46, R4-46,
shared46. No retry, sweep, extra seed, config change, source/cache/dependency
change, or metric-driven stopping occurred. Test inspection is explicitly
`false`; no Test loader, Test metric key, raw run output, or Test log was
inspected.

## New run and checkpoint identities

| method | seed | MLflow/run identity | selected epoch | checkpoint path | SHA256 |
|---|---:|---|---:|---|---|
| audio | 45 | `bf1a681a7896499b884257cacee025c5` / `stage7_audio_final_seed45_2026-09-29_15-59_audio_mamba_segment_model_e1210d8f` | 6 | `logs/wsm_mm_pd_dep_v1/stage7_audio_final_seed45_2026-09-29_15-59_audio_mamba_segment_model_e1210d8f/checkpoints/epoch=6_dev_mean_score=0.7482.pt` | `1881bc526e9d17d77f424d2b053b5ea53f7ba5d85b82cb7643176c764da5f370` |
| full R4 | 45 | `14b0d142e78b4887bddac1a7dd286471` / `stage7_r4_trial012_final_seed45_2026-09-29_16-08_wsm_av_r3_disease_query_model_9a555979` | 20 | `logs/wsm_mm_pd_dep_v1/stage7_r4_trial012_final_seed45_2026-09-29_16-08_wsm_av_r3_disease_query_model_9a555979/checkpoints/epoch=20_dev_mean_score=0.7918.pt` | `cd30baf76392ab04e8b54d64ad67b260bae71216313942d968ad564a3422fc37` |
| shared fusion | 45 | `77dc95d7b8a84ed283dc76d22ef54e83` / `stage7_shared_fusion_final_seed45_2026-09-29_16-17_wsm_av_r3_disease_query_model_2eb9d5be` | 14 | `logs/wsm_mm_pd_dep_v1/stage7_shared_fusion_final_seed45_2026-09-29_16-17_wsm_av_r3_disease_query_model_2eb9d5be/checkpoints/epoch=14_dev_mean_score=0.7872.pt` | `ccd80ca0be94d25901eb6ad628671c4ebda24ca745a22af1136f5b6c6c4ea606` |
| audio | 46 | `fa12953b90bf46869fa012a6b42dcc95` / `stage7_audio_final_seed46_2026-09-29_16-25_audio_mamba_segment_model_57104699` | 10 | `logs/wsm_mm_pd_dep_v1/stage7_audio_final_seed46_2026-09-29_16-25_audio_mamba_segment_model_57104699/checkpoints/epoch=10_dev_mean_score=0.7576.pt` | `546ca0929e746692f18106922cecb10ea4781cca2243d31855e68d7d63983263` |
| full R4 | 46 | `7cfc409f33e24713be127c295e79a279` / `stage7_r4_trial012_final_seed46_2026-09-29_16-36_wsm_av_r3_disease_query_model_52637a4b` | 30 | `logs/wsm_mm_pd_dep_v1/stage7_r4_trial012_final_seed46_2026-09-29_16-36_wsm_av_r3_disease_query_model_52637a4b/checkpoints/epoch=30_dev_mean_score=0.7978.pt` | `18ee9c9e298fab13f807404cd625360c3e6c063a73ee4340d6f474207044f5e1` |
| shared fusion | 46 | `15ef29a1583a40a9a9c8fb6f63445574` / `stage7_shared_fusion_final_seed46_2026-09-29_16-47_wsm_av_r3_disease_query_model_8d38837e` | 25 | `logs/wsm_mm_pd_dep_v1/stage7_shared_fusion_final_seed46_2026-09-29_16-47_wsm_av_r3_disease_query_model_8d38837e/checkpoints/epoch=25_dev_mean_score=0.7867.pt` | `b99c153723265641057bf25c0aa7a57c01e5dc8cac638a1bcd40f9c87bb3041b` |

## Existing accepted identities

The following accepted identities were not retrained. Full paths and SHA256
were verified locally for all 15 listed entries, including the recovered audio seed42:

- audio seed42: `logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt`; SHA `0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2`; historical run `multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77`; MLflow ID `not recorded`.
- audio seed43: `logs/wsm_mm_pd_dep_v1/frozen_audio_wavlm_l9_pool4_seed43_2026-09-28_13-12_audio_mamba_segment_model_08192c1b/checkpoints/epoch=5_dev_mean_score=0.7602.pt`; SHA `1a65f8dd605a6d942bed828f126e79a1f02930dd578176ba31f49c4dc10899dd`; MLflow `4119e9a654ec481ca42606f7f207b969`.
- audio seed44: `logs/wsm_mm_pd_dep_v1/frozen_audio_wavlm_l9_pool4_seed44_2026-09-28_13-21_audio_mamba_segment_model_8720f109/checkpoints/epoch=4_dev_mean_score=0.7719.pt`; SHA `7832d20a5431666365e2214423633f2be87bb27f37406d5a5dda8adda21e9852`; MLflow `ab9a3a73a0e94a2c889a69b25038d141`.
- full R4 seed42: `logs/wsm_mm_pd_dep_v1/optuna_r4_ra_stch_base_seed42_2026-09-28_01-54_wsm_av_r3_disease_query_model_r4-ra-stch-optuna-v1-18c6-012_0ce9c7a4/checkpoints/epoch=11_dev_mean_score=0.8191.pt`; SHA `104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a`; MLflow ID not recorded in accepted evidence.
- full R4 seed43: `logs/wsm_mm_pd_dep_v1/r4_ra_stch_optuna_trial012_confirm_seed43_2026-09-28_12-01_wsm_av_r3_disease_query_model_12038580/checkpoints/epoch=10_dev_mean_score=0.7614.pt`; SHA `6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e`; MLflow ID not recorded in accepted evidence.
- full R4 seed44: `logs/wsm_mm_pd_dep_v1/r4_ra_stch_optuna_trial012_confirm_seed44_2026-09-28_12-07_wsm_av_r3_disease_query_model_41bc602b/checkpoints/epoch=11_dev_mean_score=0.7801.pt`; SHA `b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17`; MLflow ID not recorded in accepted evidence.
- shared fusion seed42: `logs/wsm_mm_pd_dep_v1/stage6_shared_fusion_trial012_seed42_2026-09-28_22-44_wsm_av_r3_disease_query_model_5db549e8/checkpoints/epoch=11_dev_mean_score=0.8245.pt`; SHA `21504702976a960ffea02a68778ebc3bf7e6cc15ea8c9866f182ff04cbd30785`; MLflow `2fa7446c90dc44e381e5f7b6aca9720b`.
- shared fusion seed43: `logs/wsm_mm_pd_dep_v1/stage6_shared_fusion_trial012_seed43_2026-09-28_22-51_wsm_av_r3_disease_query_model_454bf80b/checkpoints/epoch=10_dev_mean_score=0.7713.pt`; SHA `ec746e519e551d69d59954dffff0a73968fb4ab555d43020b7722e635cc477ee`; MLflow `f6ef444b8e704065a679bb5294a39248`.
- shared fusion seed44: `logs/wsm_mm_pd_dep_v1/stage6_shared_fusion_trial012_seed44_2026-09-28_22-57_wsm_av_r3_disease_query_model_a398d7d8/checkpoints/epoch=11_dev_mean_score=0.7849.pt`; SHA `215e3a61c5dcb8ff7dd92ed5ca9336d66405c3f1ba6cc045841f08ec990d10a8`; MLflow `93391d46396a4475a69fe48e3ea84098`.

## New DEV evidence

Selected only by maximum `dev/mean_score`; the unified DEV pass used no Test
loader. Observed counts were D/P `621/312`.

| method | seed | D UAR/MF1/Score | P UAR/MF1/Score | Mean |
|---|---:|---|---|---:|
| audio | 45 | 0.697572/0.697224/0.697398 | 0.799034/0.799034/0.799034 | 0.748216 |
| full R4 | 45 | 0.725350/0.724803/0.725077 | 0.849758/0.867178/0.858468 | 0.791772 |
| shared fusion | 45 | 0.739916/0.739549/0.739733 | 0.825880/0.843464/0.834672 | 0.787202 |
| audio | 46 | 0.720495/0.719880/0.720187 | 0.801035/0.788813/0.794924 | 0.757556 |
| full R4 | 46 | 0.725537/0.725273/0.725405 | 0.859420/0.880909/0.870164 | 0.797785 |
| shared fusion | 46 | 0.727451/0.727451/0.727451 | 0.833264/0.858518/0.845891 | 0.786671 |

Five-seed DEV summaries use the frozen multiplier `2.7764451051977987` and
sample standard deviation. Values are ordered D, P, Mean; each tuple is
`mean / std / min / max / range / CI-low / CI-high`.

- audio: `0.721720/0.020065/0.697398/0.747918/0.050520/0.696805/0.746634`; `0.808569/0.012800/0.794924/0.827735/0.032811/0.792676/0.824463`; `0.765145/0.015234/0.748216/0.787827/0.039611/0.746230/0.784059`.
- full R4: `0.731250/0.015928/0.718527/0.759025/0.040498/0.711473/0.751027`; `0.848863/0.030547/0.804363/0.879259/0.074896/0.810934/0.886792`; `0.790057/0.021364/0.761445/0.819142/0.057697/0.763530/0.816583`.
- shared fusion: `0.737690/0.016475/0.724496/0.765324/0.040828/0.717234/0.758147`; `0.844139/0.024284/0.818152/0.883606/0.065454/0.813986/0.874292`; `0.790914/0.019857/0.771324/0.824465/0.053141/0.766258/0.815571`.

Paired Mean deltas, seeds42–46, are R4-audio
`0.031316/0.001210/0.008250/0.043556/0.040229`, shared-audio
`0.036638/0.011089/0.013021/0.038986/0.029115`, and shared-R4
`0.005323/0.009879/0.004771/-0.004570/-0.011114`. Their mean/std/95% CI
are respectively `0.024912/0.019122/[0.001169,0.048655]`,
`0.025769/0.013058/[0.009556,0.041984]`, and
`0.000858/0.008506/[-0.009704,0.011419]`. Mean wins/ties/losses are
R4-audio `5/0/0`, shared-audio `5/0/0`, shared-R4 `3/0/2`.
These are descriptive only and do not change the finalist set.

## Audio42 evaluation-path diagnostic and calibration

The manager-established source byte-identity result is preserved: the accepted
TASK-004H source files and current main are byte-identical for the adapter,
audio model, AV DataModule, manifest/segment index, and AV R3 model. Diagnostic
A reconstructed the exact WSMAVFusionDataModule path, accessed only dm.val_dataset,
and evaluated all 933 DEV rows with the recovered adapter. base_logits equaled
legacy_audio_class_logits[:,:,1] - legacy_audio_class_logits[:,:,0] with maximum
absolute difference 0.0; classification mismatch count was 0. Raw-logit
compute_sparse_two_task_metrics and independent legacy-argmax confusion counts
were identical within 1e-12: D
0.7480392157/0.7477975633/0.7479183895, P
0.8209799862/0.8344907407/0.8277353635, Mean 0.7878268765.

Diagnostic B used WSMAudioSegmentDataModule(test_filters=()) and the native DEV
task loaders. Direct historical-model logits and gathered adapter logits matched
with maximum absolute difference 0.0 for both depression and Parkinson. Native
metrics reproduced the same accepted D/P/Mean values.

Diagnostic C compared 12 stable canonical DEV rows (six depression and six
Parkinson) across the AV and native DataModules. Task IDs, labels, observed
target placement, exact feature paths, feature-file SHA256 values, sanitized
temporal tensor shapes/contents, and mask lengths all matched; all 12 rows
passed every equality flag. No raw feature values were recorded.

Diagnostic D verified current seed43 and seed44 checkpoint SHAs and reproduced
accepted native DEV scores within 0.0005: seed43 D/P/Mean
0.7086990/0.8117710/0.7602350; seed44
0.7343964/0.8093819/0.7718891.

Runtime identity: Python 3.12.3, torch 2.10.0+cu128, NumPy 2.5.1,
CUDA 12.8, cuDNN 91002, CPU device, float32. No package/environment change
occurred. The exact prior temporary evaluation code was unavailable, so the
narrower mechanism (probability/logit mixing versus task-column ordering) is
not claimed. A/B proves the prior 0.4367 result was an evaluation-path failure,
not a checkpoint, source, feature-artifact, target-identity, or current
seed43/44 evaluator failure.

The prior failed audio42 attempt is superseded and its Brier/ECE values are
discarded. Correct calibration used sigmoid(raw base_logits) only for
probabilities, observed DEV rows, 15 equal-width bins, no threshold fitting,
no recalibration, and no class balancing. Audio42 observed D/P counts were
621/312; valid Brier D/P 0.2117752880/0.1241006106; valid ECE-15 D/P
0.1664041658/0.1066767589.

All 15 checkpoint paths and SHA256 values are now frozen: the 14 identities
already verified in the prior evidence plus the recovered audio42 identity.
Five-seed calibration summaries below use the five DEV values, arithmetic mean,
sample standard deviation, and multiplier 2.7764451051977987 (two-sided 95%
Student-t CI), with D/P observed counts 621/312 per seed.

| method | task | Brier mean/std/95% CI | ECE-15 mean/std/95% CI |
|---|---|---|---|
| audio | D | 0.2170451701/0.0154621974/[0.1978463162,0.2362440239] | 0.1639118302/0.0352236709/[0.1201758661,0.2076477942] |
| audio | P | 0.1459662288/0.0219627056/[0.1186959280,0.1732365297] | 0.1321086706/0.0316775441/[0.0927758046,0.1714415367] |
| full R4 | D | 0.1838423908/0.0093544491/[0.1722273081,0.1954574735] | 0.0853467572/0.0248444032/[0.0544983561,0.1161951582] |
| full R4 | P | 0.0879230276/0.0138090356/[0.0707768452,0.1050692099] | 0.0781538793/0.0245852325/[0.0476272813,0.1086804774] |
| shared fusion | D | 0.1798457295/0.0089288407/[0.1687591094,0.1909323496] | 0.0848528604/0.0178286394/[0.0627156807,0.1069900402] |
| shared fusion | P | 0.0887637645/0.0126071922/[0.0731098679,0.1044176611] | 0.0846235798/0.0184776064/[0.0616806011,0.1075665585] |

The exact completion result is now:
STAGE-7 FIVE-SEED DEV CHECKPOINT FREEZE COMPLETE.

No method, config, threshold, source, cache, or family changed after TASK-007A.
Stage 7 remains active. Final Test remains locked.
