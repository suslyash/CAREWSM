# Stage 7 Final Test Evidence — TASK-007C

## Result and boundary

STAGE-7 FINAL TEST EVALUATION INCOMPLETE

The mandatory DEV-only preflight stopped before any Test loader or Test
dataset iteration. The exact blocker is the frozen shared-fusion seed43
checkpoint/config reproduction:

- checkpoint:
  logs/wsm_mm_pd_dep_v1/stage6_shared_fusion_trial012_seed43_2026-09-28_22-51_wsm_av_r3_disease_query_model_454bf80b/checkpoints/epoch=10_dev_mean_score=0.7713.pt
- checkpoint SHA256:
  ec746e519e551d69d59954dffff0a73968fb4ab555d43020b7722e635cc477ee
- config:
  configs/wsm_mm_pd_dep_v1/ablations/56_shared_fusion_seed43.yaml
- frozen DEV D/P/Mean:
  0.724496/0.818152/0.771324
- current exact configured shared-mode reproduction:
  0.7229314130/0.8181521375/0.7705417752
- absolute D/Mean mismatches:
  0.0015645870/0.0007822248, exceeding 0.0005.

A narrow DEV-only diagnostic also evaluated the same checkpoint with
task-aware fusion enabled; it produced D/P/Mean
0.7183358212/0.8181521375/0.7682439793 and therefore does not resolve the
mismatch. No source/config/checkpoint/cache change is authorized.

## Firewall status

Evaluator script:
scripts/common/evaluate_stage7_final_test.py

The script exposes --help, contains 15 frozen entries, verifies checkpoint and
config SHA256 values plus the pseudo-cache SHA256, implements raw-logit
thresholding, historical two-class margin/argmax equivalence, sparse masks,
Student-t arithmetic, atomic output, and overwrite refusal. Static and
synthetic checks passed. The all-15 DEV preflight was attempted with the exact
semantic RAMPS DataModule. Audio and R4 entries, and shared seed42, passed
before the shared seed43 mismatch stopped execution.

No firewall commit was eligible to authorize Test because the DEV gate failed.
No standardized Final Test invocation occurred; invocation count is 0.
The external report was not created, so report SHA256 is not applicable.

## Frozen roles and boundaries

Pre-Test roles remain unchanged and were not revised:

1. primary paper candidate: equal-parameter shared fusion;
2. secondary multimodal reference: full R4 trial012;
3. baseline: temporal audio.

No training, Test evaluation, recalibration, threshold fitting, model selection,
role revision, retry, or Test-driven decision occurred. The prior historical
Test-line visibility procedural exception remains separate:
FINAL TEST VALUES NOT USED FOR TASK-007B SELECTION OR RECOVERY.

Stage 7 remains active pending manager review. Final Test remains locked by the
failed pre-Test DEV gate.


## Corrective config-native DEV firewall

The first preflight blocker is resolved at the evaluator runtime layer only. Installed Chimera semantics were inspected read-only: `cuda` is selected when available, `mixed_precision=true` enables `torch.amp.autocast(device_type="cuda", enabled=True)`, and the environment's default CUDA autocast dtype is float16. The corrected evaluator reads the frozen fusion config's `train.device` and `train.mixed_precision`, mirrors that gate, and converts outputs to CPU float32 before metric calculation. No model, config, cache, source, or dependency was changed.

For shared seed43, canonical DEV-only A/B/C results were:

- A CPU float32/no autocast: `0.7229314130 / 0.8181521375 / 0.7705417752`.
- B CUDA float32/no autocast: `0.7229314130 / 0.8181521375 / 0.7705417752`; zero sign disagreements vs A, max raw-logit delta `1.9669533e-06`.
- C CUDA config-native float16 autocast: `0.7244960383 / 0.8181521375 / 0.7713240879`; one sign disagreement vs A, max raw-logit delta `0.0028064251`.

C matches the accepted frozen seed43 DEV triple within `0.0005`, while A retains the mismatch. Shared seed42 and seed44 controls also passed under config-native semantics: `0.7653242327 / 0.8836056059 / 0.8244649193` and `0.7314476158 / 0.8383722433 / 0.7849099296`, respectively, each within `0.0005` of frozen evidence.

The mandatory full evaluator DEV preflight then passed all 15 frozen entries (3 methods × 5 seeds), all checkpoint/config/pseudo-cache SHA checks, audio42 margin/argmax equivalence, synthetic checks, and overwrite refusal. All reproduced D/P/Mean triples were within `0.0005`; the evaluator reported `test_iteration: false`. The corrected firewall is now eligible for commit/push.

Final Test remains unrun: invocation count `0`, no Test loader/dataset iteration, and no external report. The final Test command is authorized only after this corrective firewall commit is visible on `origin`; no Test result or method claim exists yet.


## Final Test evidence after the corrective firewall

Exact completion result: `STAGE-7 FINAL TEST EVALUATION COMPLETE`.

- Corrective firewall/evaluator commit pushed before Test: `bfaae7f812904dd9ae9a7460b394117867db5439`.
- Evaluator script SHA256: `b691c9563bb6ffd63316b664949dcf1bef02d8d3f4e770110c2e70adbc4cee1e`.
- External report: `/media/maxim/Programs/Features/WSM/stage7_final_test/final_test_v1.json`.
- External report SHA256: `59f763f04843c8e54d5d3f9893d3abe0530b47a32ea11f5b9585390edeb515bc`.
- Runtime metadata recorded in the report: Python 3.12.3; torch 2.10.0+cu128; NumPy 2.5.1; CUDA 12.8; cuDNN 91002. Fusion evaluation used the frozen config-native CUDA float16 autocast semantics established by the corrective firewall; aggregate report metrics were cast to CPU float32.
- The 15 checkpoint/config/run identities are preserved by the exact frozen DEV ledger encoded in the evaluator and are repeated in each report row; no identity was changed.
- Exactly one standardized Final Test invocation occurred after the pushed firewall. No training, recalibration, threshold fitting, retry, tuning, or post-Test role/config/checkpoint revision occurred.

Protocol membership counts: `TEST_NONE=1364`, `TEST_SOFT=1208`, `TEST_HARD=1014`.

### Per-seed aggregate Test rows

D/P entries are `UAR / MF1 / Score`; the final column is Mean_Score.

#### test_none

| Method | Seed | Depression UAR/MF1/Score | Parkinson UAR/MF1/Score | Mean | Checkpoint/config trace |
|---|---:|---:|---:|---:|---|
| temporal audio | 42 | 0.768681/0.763094/0.765887 | 0.865406/0.840764/0.853085 | 0.809486 | `0873c7cb5e32…` / `955ad447dd94…` |
| temporal audio | 43 | 0.730448/0.733508/0.731978 | 0.870589/0.832216/0.851403 | 0.791690 | `1a65f8dd605a…` / `e4df4c8bb9bf…` |
| temporal audio | 44 | 0.777767/0.782953/0.780360 | 0.840002/0.846956/0.843479 | 0.811919 | `7832d20a5431…` / `d9c9e596df5f…` |
| temporal audio | 45 | 0.724572/0.724890/0.724731 | 0.849410/0.809085/0.829247 | 0.776989 | `1881bc526e9d…` / `8bb3c4109ddc…` |
| temporal audio | 46 | 0.767695/0.755935/0.761815 | 0.889479/0.838395/0.863937 | 0.812876 | `546ca0929e74…` / `71ba2fc42ff6…` |
| full R4 trial012 | 42 | 0.749184/0.749769/0.749477 | 0.848712/0.858579/0.853646 | 0.801561 | `104b79e40983…` / `04ed5fcb2a1a…` |
| full R4 trial012 | 43 | 0.813524/0.815573/0.814549 | 0.819828/0.839975/0.829902 | 0.822225 | `6af4a4ed2104…` / `9409f6e581eb…` |
| full R4 trial012 | 44 | 0.837944/0.838105/0.838025 | 0.799886/0.815453/0.807670 | 0.822847 | `b06fec6912b2…` / `5f13b1c72b0d…` |
| full R4 trial012 | 45 | 0.767634/0.767120/0.767377 | 0.793884/0.793144/0.793514 | 0.780446 | `cd30baf76392…` / `99cc0c9c2994…` |
| full R4 trial012 | 46 | 0.783609/0.787306/0.785458 | 0.786087/0.804730/0.795408 | 0.790433 | `18ee9c9e298f…` / `c37eb7c0819b…` |
| equal-parameter shared fusion | 42 | 0.757123/0.756877/0.757000 | 0.838672/0.851825/0.845248 | 0.801124 | `21504702976a…` / `4c8d2d18b4ee…` |
| equal-parameter shared fusion | 43 | 0.809871/0.810312/0.810092 | 0.817306/0.839029/0.828168 | 0.819130 | `ec746e519e55…` / `62e6631c4538…` |
| equal-parameter shared fusion | 44 | 0.833403/0.833244/0.833324 | 0.781322/0.784889/0.783106 | 0.808215 | `215e3a61c5dc…` / `5a9cef2d7900…` |
| equal-parameter shared fusion | 45 | 0.786689/0.785999/0.786344 | 0.800073/0.803115/0.801594 | 0.793969 | `ccd80ca0be94…` / `30236d8f06ea…` |
| equal-parameter shared fusion | 46 | 0.805555/0.813255/0.809405 | 0.796081/0.815430/0.805755 | 0.807580 | `b99c15372326…` / `00310e540f89…` |

### Five-seed summaries

| Protocol | Method | D mean [95% CI] | P mean [95% CI] | Mean mean [95% CI] |
|---|---|---:|---:|---:|
| test_none | temporal audio | 0.752954 [0.723614,0.782294] | 0.848230 [0.832241,0.864219] | 0.800592 [0.781012,0.820172] |
| test_none | full R4 trial012 | 0.790977 [0.746733,0.835221] | 0.816028 [0.784314,0.847742] | 0.803502 [0.780018,0.826987] |
| test_none | equal-parameter shared fusion | 0.799233 [0.763389,0.835076] | 0.812774 [0.782704,0.842844] | 0.806003 [0.794417,0.817590] |

Summary calculation details: each cell uses five seed values, arithmetic mean, sample standard deviation (`ddof=1`), min, max, range, and the frozen multiplier `2.7764451051977987`; full min/max/range/std values are reproducible from the per-seed rows above.

### Paired same-seed comparisons

| Pair | Metric | Five deltas (42,43,44,45,46) | Mean | Sample std | 95% CI | W/T/L for Mean |
|---|---|---|---:|---:|---:|---:|
| full R4 trial012 − temporal audio | D | -0.016411, 0.082571, 0.057665, 0.042646, 0.023643 | 0.038023 | 0.037288 | [-0.008276,0.084322] | — |
| full R4 trial012 − temporal audio | P | 0.000561, -0.021501, -0.035809, -0.035733, -0.068529 | -0.032202 | 0.025165 | [-0.063448,-0.000956] | — |
| full R4 trial012 − temporal audio | Mean | -0.007925, 0.030535, 0.010928, 0.003457, -0.022443 | 0.002910 | 0.019924 | [-0.021829,0.027649] | 3/0/2 |
| equal-parameter shared fusion − temporal audio | D | -0.008888, 0.078114, 0.052964, 0.061613, 0.047590 | 0.046279 | 0.032933 | [0.005387,0.087170] | — |
| equal-parameter shared fusion − temporal audio | P | -0.007837, -0.023235, -0.060374, -0.027653, -0.058182 | -0.035456 | 0.022970 | [-0.063977,-0.006936] | — |
| equal-parameter shared fusion − temporal audio | Mean | -0.008362, 0.027439, -0.003705, 0.016980, -0.005296 | 0.005411 | 0.015863 | [-0.014285,0.025108] | 2/0/3 |
| equal-parameter shared fusion − full R4 trial012 | D | 0.007523, -0.004457, -0.004701, 0.018967, 0.023947 | 0.008256 | 0.013143 | [-0.008064,0.024575] | — |
| equal-parameter shared fusion − full R4 trial012 | P | -0.008398, -0.001734, -0.024564, 0.008080, 0.010347 | -0.003254 | 0.014110 | [-0.020773,0.014266] | — |
| equal-parameter shared fusion − full R4 trial012 | Mean | -0.000437, -0.003095, -0.014633, 0.013523, 0.017147 | 0.002501 | 0.012937 | [-0.013563,0.018565] | 2/0/3 |

### DEV-to-Test descriptive Mean deltas

| Protocol | Method | Frozen DEV five-seed Mean | Test five-seed Mean | Test − DEV |
|---|---|---:|---:|---:|
| test_none | temporal audio | 0.765144575 | 0.800592239 | +0.035447664 |
| test_none | full R4 trial012 | 0.790056691 | 0.803502485 | +0.013445794 |
| test_none | equal-parameter shared fusion | 0.790914400 | 0.806003456 | +0.015089056 |

#### test_soft

| Method | Seed | Depression UAR/MF1/Score | Parkinson UAR/MF1/Score | Mean | Checkpoint/config trace |
|---|---:|---:|---:|---:|---|
| temporal audio | 42 | 0.781461/0.775970/0.778716 | 0.863621/0.839571/0.851596 | 0.815156 | `0873c7cb5e32…` / `955ad447dd94…` |
| temporal audio | 43 | 0.752115/0.754909/0.753512 | 0.872253/0.831131/0.851692 | 0.802602 | `1a65f8dd605a…` / `e4df4c8bb9bf…` |
| temporal audio | 44 | 0.794414/0.798231/0.796322 | 0.841036/0.847734/0.844385 | 0.820354 | `7832d20a5431…` / `d9c9e596df5f…` |
| temporal audio | 45 | 0.749585/0.750457/0.750021 | 0.857408/0.815141/0.836274 | 0.793148 | `1881bc526e9d…` / `8bb3c4109ddc…` |
| temporal audio | 46 | 0.803643/0.798463/0.801053 | 0.892854/0.841371/0.867112 | 0.834083 | `546ca0929e74…` / `71ba2fc42ff6…` |
| full R4 trial012 | 42 | 0.759360/0.757326/0.758343 | 0.858541/0.870757/0.864649 | 0.811496 | `104b79e40983…` / `04ed5fcb2a1a…` |
| full R4 trial012 | 43 | 0.825096/0.824832/0.824964 | 0.818778/0.843862/0.831320 | 0.828142 | `6af4a4ed2104…` / `9409f6e581eb…` |
| full R4 trial012 | 44 | 0.857478/0.854491/0.855984 | 0.802712/0.819144/0.810928 | 0.833456 | `b06fec6912b2…` / `5f13b1c72b0d…` |
| full R4 trial012 | 45 | 0.774540/0.771895/0.773217 | 0.789632/0.790435/0.790033 | 0.781625 | `cd30baf76392…` / `99cc0c9c2994…` |
| full R4 trial012 | 46 | 0.798442/0.799500/0.798971 | 0.794406/0.815624/0.805015 | 0.801993 | `18ee9c9e298f…` / `c37eb7c0819b…` |
| equal-parameter shared fusion | 42 | 0.767962/0.764983/0.766473 | 0.847576/0.863487/0.855531 | 0.811002 | `21504702976a…` / `4c8d2d18b4ee…` |
| equal-parameter shared fusion | 43 | 0.820138/0.818400/0.819269 | 0.820108/0.846296/0.833202 | 0.826236 | `ec746e519e55…` / `62e6631c4538…` |
| equal-parameter shared fusion | 44 | 0.853370/0.850259/0.851814 | 0.782765/0.785913/0.784339 | 0.818077 | `215e3a61c5dc…` / `5a9cef2d7900…` |
| equal-parameter shared fusion | 45 | 0.795062/0.791983/0.793522 | 0.798941/0.805642/0.802291 | 0.797907 | `ccd80ca0be94…` / `30236d8f06ea…` |
| equal-parameter shared fusion | 46 | 0.830540/0.834875/0.832708 | 0.802494/0.826186/0.814340 | 0.823524 | `b99c15372326…` / `00310e540f89…` |

### Five-seed summaries

| Protocol | Method | D mean [95% CI] | P mean [95% CI] | Mean mean [95% CI] |
|---|---|---:|---:|---:|
| test_soft | temporal audio | 0.775925 [0.746616,0.805233] | 0.850212 [0.836083,0.864341] | 0.813068 [0.793383,0.832754] |
| test_soft | full R4 trial012 | 0.802296 [0.753472,0.851120] | 0.820389 [0.784586,0.856192] | 0.811343 [0.785432,0.837253] |
| test_soft | equal-parameter shared fusion | 0.812757 [0.771212,0.854303] | 0.817941 [0.783750,0.852131] | 0.815349 [0.801248,0.829450] |

Summary calculation details: each cell uses five seed values, arithmetic mean, sample standard deviation (`ddof=1`), min, max, range, and the frozen multiplier `2.7764451051977987`; full min/max/range/std values are reproducible from the per-seed rows above.

### Paired same-seed comparisons

| Pair | Metric | Five deltas (42,43,44,45,46) | Mean | Sample std | 95% CI | W/T/L for Mean |
|---|---|---|---:|---:|---:|---:|
| full R4 trial012 − temporal audio | D | -0.020372, 0.071452, 0.059662, 0.023196, -0.002082 | 0.026371 | 0.039196 | [-0.022297,0.075039] | — |
| full R4 trial012 − temporal audio | P | 0.013054, -0.020372, -0.033457, -0.046241, -0.062097 | -0.029823 | 0.028514 | [-0.065227,0.005582] | — |
| full R4 trial012 − temporal audio | Mean | -0.003659, 0.025540, 0.013102, -0.011522, -0.032090 | -0.001726 | 0.022277 | [-0.029386,0.025934] | 2/0/3 |
| equal-parameter shared fusion − temporal audio | D | -0.012243, 0.065757, 0.055492, 0.043501, 0.031655 | 0.036832 | 0.030267 | [-0.000750,0.074414] | — |
| equal-parameter shared fusion − temporal audio | P | 0.003936, -0.018490, -0.060046, -0.033983, -0.052772 | -0.032271 | 0.025959 | [-0.064504,-0.000038] | — |
| equal-parameter shared fusion − temporal audio | Mean | -0.004154, 0.023634, -0.002277, 0.004759, -0.010559 | 0.002281 | 0.013125 | [-0.014017,0.018578] | 2/0/3 |
| equal-parameter shared fusion − full R4 trial012 | D | 0.008129, -0.005695, -0.004170, 0.020305, 0.033737 | 0.010461 | 0.016727 | [-0.010308,0.031231] | — |
| equal-parameter shared fusion − full R4 trial012 | P | -0.009118, 0.001882, -0.026589, 0.012258, 0.009325 | -0.002448 | 0.015818 | [-0.022090,0.017193] | — |
| equal-parameter shared fusion − full R4 trial012 | Mean | -0.000494, -0.001907, -0.015380, 0.016281, 0.021531 | 0.004006 | 0.014907 | [-0.014503,0.022515] | 2/0/3 |

### DEV-to-Test descriptive Mean deltas

| Protocol | Method | Frozen DEV five-seed Mean | Test five-seed Mean | Test − DEV |
|---|---|---:|---:|---:|
| test_soft | temporal audio | 0.765144575 | 0.813068329 | +0.047923753 |
| test_soft | full R4 trial012 | 0.790056691 | 0.811342575 | +0.021285884 |
| test_soft | equal-parameter shared fusion | 0.790914400 | 0.815348966 | +0.024434566 |

#### test_hard

| Method | Seed | Depression UAR/MF1/Score | Parkinson UAR/MF1/Score | Mean | Checkpoint/config trace |
|---|---:|---:|---:|---:|---|
| temporal audio | 42 | 0.794493/0.786820/0.790656 | 0.883176/0.848053/0.865615 | 0.828135 | `0873c7cb5e32…` / `955ad447dd94…` |
| temporal audio | 43 | 0.772319/0.774505/0.773412 | 0.888623/0.837311/0.862967 | 0.818189 | `1a65f8dd605a…` / `e4df4c8bb9bf…` |
| temporal audio | 44 | 0.806129/0.809945/0.808037 | 0.855499/0.855499/0.855499 | 0.831768 | `7832d20a5431…` / `d9c9e596df5f…` |
| temporal audio | 45 | 0.768700/0.767829/0.768265 | 0.869070/0.816850/0.842960 | 0.805612 | `1881bc526e9d…` / `8bb3c4109ddc…` |
| temporal audio | 46 | 0.804714/0.797439/0.801077 | 0.899365/0.849013/0.874189 | 0.837633 | `546ca0929e74…` / `71ba2fc42ff6…` |
| full R4 trial012 | 42 | 0.762184/0.758727/0.760456 | 0.855251/0.862857/0.859054 | 0.809755 | `104b79e40983…` / `04ed5fcb2a1a…` |
| full R4 trial012 | 43 | 0.828496/0.827441/0.827968 | 0.848121/0.863222/0.855671 | 0.841820 | `6af4a4ed2104…` / `9409f6e581eb…` |
| full R4 trial012 | 44 | 0.869716/0.864687/0.867202 | 0.833767/0.836804/0.835285 | 0.851243 | `b06fec6912b2…` / `5f13b1c72b0d…` |
| full R4 trial012 | 45 | 0.777391/0.774104/0.775747 | 0.809760/0.797743/0.803752 | 0.789749 | `cd30baf76392…` / `99cc0c9c2994…` |
| full R4 trial012 | 46 | 0.798304/0.799390/0.798847 | 0.829657/0.837717/0.833687 | 0.816267 | `18ee9c9e298f…` / `c37eb7c0819b…` |
| equal-parameter shared fusion | 42 | 0.771905/0.767158/0.769531 | 0.857181/0.865940/0.861561 | 0.815546 | `21504702976a…` / `4c8d2d18b4ee…` |
| equal-parameter shared fusion | 43 | 0.824848/0.822011/0.823430 | 0.848121/0.863222/0.855671 | 0.839551 | `ec746e519e55…` / `62e6631c4538…` |
| equal-parameter shared fusion | 44 | 0.866569/0.861576/0.864073 | 0.803720/0.795529/0.799624 | 0.831848 | `215e3a61c5dc…` / `5a9cef2d7900…` |
| equal-parameter shared fusion | 45 | 0.798641/0.794614/0.796627 | 0.825204/0.820324/0.822764 | 0.809696 | `ccd80ca0be94…` / `30236d8f06ea…` |
| equal-parameter shared fusion | 46 | 0.823424/0.828416/0.825920 | 0.840399/0.850852/0.845626 | 0.835773 | `b99c15372326…` / `00310e540f89…` |

### Five-seed summaries

| Protocol | Method | D mean [95% CI] | P mean [95% CI] | Mean mean [95% CI] |
|---|---|---:|---:|---:|
| test_hard | temporal audio | 0.788289 [0.766950,0.809628] | 0.860246 [0.845660,0.874832] | 0.824268 [0.808625,0.839910] |
| test_hard | full R4 trial012 | 0.806044 [0.753113,0.858975] | 0.837490 [0.810052,0.864927] | 0.821767 [0.790908,0.852626] |
| test_hard | equal-parameter shared fusion | 0.815916 [0.772020,0.859812] | 0.837049 [0.805236,0.868863] | 0.826483 [0.810207,0.842758] |

Summary calculation details: each cell uses five seed values, arithmetic mean, sample standard deviation (`ddof=1`), min, max, range, and the frozen multiplier `2.7764451051977987`; full min/max/range/std values are reproducible from the per-seed rows above.

### Paired same-seed comparisons

| Pair | Metric | Five deltas (42,43,44,45,46) | Mean | Sample std | 95% CI | W/T/L for Mean |
|---|---|---|---:|---:|---:|---:|
| full R4 trial012 − temporal audio | D | -0.030201, 0.054556, 0.059165, 0.007483, -0.002230 | 0.017755 | 0.038320 | [-0.029826,0.065336] | — |
| full R4 trial012 − temporal audio | P | -0.006561, -0.007296, -0.020214, -0.039208, -0.040502 | -0.022756 | 0.016533 | [-0.043285,-0.002228] | — |
| full R4 trial012 − temporal audio | Mean | -0.018381, 0.023630, 0.019476, -0.015863, -0.021366 | -0.002501 | 0.022093 | [-0.029933,0.024931] | 2/0/3 |
| equal-parameter shared fusion − temporal audio | D | -0.021125, 0.050018, 0.056036, 0.028363, 0.024843 | 0.027627 | 0.030387 | [-0.010103,0.065357] | — |
| equal-parameter shared fusion − temporal audio | P | -0.004054, -0.007296, -0.055875, -0.020196, -0.028563 | -0.023197 | 0.020766 | [-0.048982,0.002588] | — |
| equal-parameter shared fusion − temporal audio | Mean | -0.012589, 0.021361, 0.000081, 0.004083, -0.001860 | 0.002215 | 0.012353 | [-0.013123,0.017553] | 3/0/2 |
| equal-parameter shared fusion − full R4 trial012 | D | 0.009076, -0.004539, -0.003129, 0.020880, 0.027073 | 0.009872 | 0.014092 | [-0.007626,0.027370] | — |
| equal-parameter shared fusion − full R4 trial012 | P | 0.002507, 0.000000, -0.035661, 0.019012, 0.011939 | -0.000441 | 0.021101 | [-0.026641,0.025760] | — |
| equal-parameter shared fusion − full R4 trial012 | Mean | 0.005791, -0.002269, -0.019395, 0.019946, 0.019506 | 0.004716 | 0.016447 | [-0.015706,0.025137] | 3/0/2 |

### DEV-to-Test descriptive Mean deltas

| Protocol | Method | Frozen DEV five-seed Mean | Test five-seed Mean | Test − DEV |
|---|---|---:|---:|---:|
| test_hard | temporal audio | 0.765144575 | 0.824267573 | +0.059122998 |
| test_hard | full R4 trial012 | 0.790056691 | 0.821766905 | +0.031710215 |
| test_hard | equal-parameter shared fusion | 0.790914400 | 0.826482635 | +0.035568235 |

### Interpretation and boundary

- Frozen pre-Test roles remain unchanged: shared fusion is the primary paper candidate, full R4 is the secondary multimodal reference, and temporal audio is the baseline.
- Test point estimates do not revise roles. No claim that shared fusion statistically outperforms full R4 is made: every paired Mean 95% CI includes zero.
- These are reporting/generalization results only. They do not establish missing-label correctness or comorbidity recovery, and do not revise any unsupported Stage-6 mechanism claim.
- The prior historical Test-line visibility exception is separate from this standardized evaluation and was not used for this report.
- Stage 7 remains pending manager closure; no further experiment is authorized by this result.


### Complete five-seed summary statistics (mean / sample std / min / max / range / 95% CI)

#### test_none

| Method | Metric | Mean | SD | Min | Max | Range | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|
| audio | D | 0.752954220 | 0.023629389 | 0.724731048 | 0.780359717 | 0.055628669 | [0.723614458,0.782293982] |
| audio | P | 0.848230259 | 0.012876941 | 0.829247335 | 0.863937154 | 0.034689819 | [0.832241424,0.864219093] |
| audio | Mean | 0.800592239 | 0.015769043 | 0.776989191 | 0.812876063 | 0.035886872 | [0.781012387,0.820172092] |
| R4 | D | 0.790976991 | 0.035632891 | 0.749476612 | 0.838024817 | 0.088548205 | [0.746732913,0.835221068] |
| R4 | P | 0.816027979 | 0.025541472 | 0.793514361 | 0.853645728 | 0.060131367 | [0.784314052,0.847741906] |
| R4 | Mean | 0.803502485 | 0.018913944 | 0.780445739 | 0.822847289 | 0.042401550 | [0.780017722,0.826987248] |
| shared | D | 0.799232789 | 0.028867444 | 0.756999840 | 0.833323622 | 0.076323782 | [0.763389123,0.835076455] |
| shared | P | 0.812774124 | 0.024217744 | 0.783105673 | 0.845248167 | 0.062142494 | [0.782703823,0.842844424] |
| shared | Mean | 0.806003456 | 0.009331527 | 0.793968973 | 0.819129645 | 0.025160672 | [0.794416835,0.817590078] |
#### test_soft

| Method | Metric | Mean | SD | Min | Max | Range | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|
| audio | D | 0.775924741 | 0.023604200 | 0.750020855 | 0.801053087 | 0.051032232 | [0.746616256,0.805233226] |
| audio | P | 0.850211917 | 0.011379112 | 0.836274362 | 0.867112342 | 0.030837981 | [0.836082883,0.864340950] |
| audio | Mean | 0.813068329 | 0.015854168 | 0.793147608 | 0.834082715 | 0.040935106 | [0.793382780,0.832753878] |
| R4 | D | 0.802295956 | 0.039321574 | 0.758343451 | 0.855984450 | 0.097640999 | [0.753471774,0.851120138] |
| R4 | P | 0.820389193 | 0.028834837 | 0.790033494 | 0.864649257 | 0.074615764 | [0.784586014,0.856192373] |
| R4 | Mean | 0.811342575 | 0.020867379 | 0.781625358 | 0.833456191 | 0.051830832 | [0.785432301,0.837252848] |
| shared | D | 0.812757146 | 0.033459433 | 0.766472575 | 0.851814286 | 0.085341711 | [0.771211773,0.854302519] |
| shared | P | 0.817940786 | 0.027535964 | 0.784338892 | 0.855531353 | 0.071192461 | [0.783750370,0.852131202] |
| shared | Mean | 0.815348966 | 0.011356850 | 0.797906817 | 0.826235527 | 0.028328710 | [0.801247574,0.829450358] |
#### test_hard

| Method | Metric | Mean | SD | Min | Max | Range | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|
| audio | D | 0.788289192 | 0.017185831 | 0.768264585 | 0.808036638 | 0.039772053 | [0.766950164,0.809628219] |
| audio | P | 0.860245955 | 0.011747116 | 0.842959868 | 0.874189048 | 0.031229180 | [0.845659983,0.874831926] |
| audio | Mean | 0.824267573 | 0.012597735 | 0.805612226 | 0.837632877 | 0.032020650 | [0.808625420,0.839909727] |
| R4 | D | 0.806043962 | 0.042628844 | 0.760455813 | 0.867201518 | 0.106745705 | [0.753113261,0.858974664] |
| R4 | P | 0.837489848 | 0.022097369 | 0.803751521 | 0.859053863 | 0.055302342 | [0.810052341,0.864927356] |
| R4 | Mean | 0.821766905 | 0.024852758 | 0.789749434 | 0.851243400 | 0.061493965 | [0.790908130,0.852625680] |
| shared | D | 0.815916137 | 0.035352380 | 0.769531489 | 0.864072625 | 0.094541136 | [0.772020360,0.859811915] |
| shared | P | 0.837049133 | 0.025621594 | 0.799624192 | 0.861560634 | 0.061936442 | [0.805235723,0.868862544] |
| shared | Mean | 0.826482635 | 0.013108037 | 0.809695500 | 0.839550501 | 0.029855001 | [0.810206857,0.842758413] |
