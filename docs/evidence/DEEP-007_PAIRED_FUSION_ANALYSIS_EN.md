# DEEP-007 — Paired Fusion-vs-Single-Family Validation Analysis

## Scope and provenance

This is a read-only analysis of the completed DEEP-003 native-turn baseline and
DEEP-006 native-turn fusion runtimes. No training, feature extraction, TEST-based
selection, configuration change, or source change was performed. The analysis was
run from commit `a36566488d33cac194aae90e8413041d816e35eb`.

The selected row for each run is the row with maximum
`dev_mean_macro_recall`, breaking ties by earliest epoch. All comparisons below
use the selected validation row only. A paired difference is
`fusion - single-family`, in percentage points (pp), with fold order `0/1/2/3/4`.
The single-family run has the same model and fold as the fusion run.

## Actual four-family input findings

The audited native artifacts are:

| Family | Observed native value | Dtype and keys | Temporal evidence |
|---|---|---|---|
| `emotionmsp_embeds` | `[T, 1024]` | `float32`; `embed`, `turn_ids` | turn-indexed; CARE `T=1..55` (35 distinct values), WSM `T=1` |
| `emotionmsp_trajectories` | `[T, 3]` | parsed `float32`; `arousal`, `dominance`, `valence`, `turn_id`, `turn_start_sec`, `turn_end_sec` | explicit turn timestamps; CARE `T=1..55` (35 distinct values), WSM `T=1` |
| `trillsson4_1` | `[T, 1024]` for selected `mean` | `float32`; `mean`, `std`, `turn_ids` | turn-indexed; CARE `T=1..55` (35 distinct values), WSM `T=1` |
| `wav2vec_xlsr300m` | `[T, 1024]` for selected `mean` | `float32`; `mean`, `std`, `turn_ids` | turn-indexed; CARE `T=1..55` (35 distinct values), WSM `T=1` |

All inspected valid arrays were readable, non-empty, finite, and schema-valid.
The registry's 2,048-dimensional mean-plus-standard-deviation description for
the last two families is not the active native loader contract: the loader uses
`mean`, so the audited channel dimension is 1,024. That distinction must remain
explicit.

The four families agree on turn cardinality and NPZ turn IDs in the inspected
samples. This establishes turn correspondence, not a shared frame rate or a
shared 1-second temporal grid. Only the trajectory family supplies explicit
timestamps. Therefore raw temporal axes are not independently verified as
aligned, and no raw temporal concatenation is scientifically justified by these
artifacts alone. CARE has variable `T`; WSM is single-turn in the live artifacts.

## Coverage of all 15 nonempty subsets

The four validity masks coincide in the audited availability manifest. Thus every
nonempty subset has the same coverage; this is an observed intersection, not a
claim that missingness can be ignored in future extraction.

| Subset | Retained | Coverage |
|---|---:|---:|
| `emotionmsp_embeds` | 10,461 | 99.97% |
| `emotionmsp_trajectories` | 10,461 | 99.97% |
| `trillsson4_1` | 10,461 | 99.97% |
| `wav2vec_xlsr300m` | 10,461 | 99.97% |
| `emotionmsp_embeds__emotionmsp_trajectories` | 10,461 | 99.97% |
| `emotionmsp_embeds__trillsson4_1` | 10,461 | 99.97% |
| `emotionmsp_embeds__wav2vec_xlsr300m` | 10,461 | 99.97% |
| `emotionmsp_trajectories__trillsson4_1` | 10,461 | 99.97% |
| `emotionmsp_trajectories__wav2vec_xlsr300m` | 10,461 | 99.97% |
| `trillsson4_1__wav2vec_xlsr300m` | 10,461 | 99.97% |
| `emotionmsp_embeds__emotionmsp_trajectories__trillsson4_1` | 10,461 | 99.97% |
| `emotionmsp_embeds__emotionmsp_trajectories__wav2vec_xlsr300m` | 10,461 | 99.97% |
| `emotionmsp_embeds__trillsson4_1__wav2vec_xlsr300m` | 10,461 | 99.97% |
| `emotionmsp_trajectories__trillsson4_1__wav2vec_xlsr300m` | 10,461 | 99.97% |
| `emotionmsp_embeds__emotionmsp_trajectories__trillsson4_1__wav2vec_xlsr300m` | 10,461 | 99.97% |

## Retained populations and masks

The following counts apply to every one of the 15 subsets. `Dep obs` and `PD
obs` preserve the original observed-label masks; missing labels were not changed
to negatives.

| Fold | Population | Expected | Retained | Dep obs | PD obs |
|---:|---|---:|---:|---:|---:|
| 0 | CARE train | 1,361 | 1,361 | 1,361 | 1,361 |
| 0 | WSM train | 5,640 | 5,637 | 3,404 | 2,233 |
| 0 | CARE val | 352 | 352 | 352 | 352 |
| 0 | WSM val | 1,429 | 1,429 | 877 | 552 |
| 1 | CARE train | 1,368 | 1,368 | 1,368 | 1,368 |
| 1 | WSM train | 5,661 | 5,659 | 3,429 | 2,230 |
| 1 | CARE val | 345 | 345 | 345 | 345 |
| 1 | WSM val | 1,408 | 1,407 | 852 | 555 |
| 2 | CARE train | 1,378 | 1,378 | 1,378 | 1,378 |
| 2 | WSM train | 5,670 | 5,667 | 3,434 | 2,233 |
| 2 | CARE val | 335 | 335 | 335 | 335 |
| 2 | WSM val | 1,399 | 1,399 | 847 | 552 |
| 3 | CARE train | 1,382 | 1,382 | 1,382 | 1,382 |
| 3 | WSM train | 5,640 | 5,638 | 3,429 | 2,209 |
| 3 | CARE val | 331 | 331 | 331 | 331 |
| 3 | WSM val | 1,429 | 1,428 | 852 | 576 |
| 4 | CARE train | 1,363 | 1,363 | 1,363 | 1,363 |
| 4 | WSM train | 5,665 | 5,663 | 3,428 | 2,235 |
| 4 | CARE val | 350 | 350 | 350 | 350 |
| 4 | WSM val | 1,404 | 1,403 | 853 | 550 |
| all folds | CARE TEST | 318 | 318 | 318 | 318 |
| all folds | WSM TEST | 1,364 | 1,364 | 827 | 537 |

## Paired validation results

There are exactly 40 DEEP-003 baseline runs and 110 DEEP-006 fusion runs: 4
families x 2 models x 5 folds, and 11 subsets x 2 models x 5 folds,
respectively. The strongest complete single-family baseline is
`emotionmsp_embeds` with MD-Former, mean `0.7944`; the corresponding Pooled MLP
mean is `0.7919`. This global-best value is context only, not a cross-model
selection rule.

### Comparison with the strongest constituent

The strongest constituent is selected from the same-model DEEP-003 five-fold
mean. `SD` is the sample standard deviation of the five paired differences.

| Fusion subset | Model | Fusion mean | Strongest constituent (mean) | Diff mean / SD pp | Min..max pp | Improved folds | Fold differences pp |
|---|---|---:|---|---:|---:|---:|---|
| embeds__trajectories | Pooled MLP | .7906 | embeds (.7919) | -0.13 / 0.58 | -0.76..+0.97 | 1/5 | +0.97/-0.30/-0.14/-0.76/-0.40 |
| embeds__trajectories | MD-Former | .7919 | embeds (.7944) | -0.25 / 1.83 | -2.62..+2.28 | 2/5 | +1.09/-2.62/-1.94/-0.04/+2.28 |
| embeds__trajectories__trillsson | Pooled MLP | .7467 | embeds (.7919) | -4.52 / 3.95 | -8.84..+1.79 | 1/5 | +1.79/-7.57/-8.84/-1.77/-6.20 |
| embeds__trajectories__trillsson | MD-Former | .7539 | embeds (.7944) | -4.05 / 5.28 | -10.44..+4.19 | 1/5 | -0.43/-7.90/-10.44/+4.19/-5.66 |
| embeds__trajectories__trillsson__wav | Pooled MLP | .7681 | embeds (.7919) | -2.38 / 2.79 | -7.07..+0.74 | 1/5 | -0.05/-3.64/-7.07/-1.86/+0.74 |
| embeds__trajectories__trillsson__wav | MD-Former | .7207 | embeds (.7944) | -7.37 / 7.28 | -19.50..-1.07 | 1/5 | +0.35/-19.50/-11.19/-5.44/-1.07 |
| embeds__trajectories__wav | Pooled MLP | .7338 | embeds (.7919) | -5.81 / 2.21 | -9.07..-2.87 | 0/5 | -2.87/-7.07/-6.08/-9.07/-3.94 |
| embeds__trajectories__wav | MD-Former | .6795 | embeds (.7944) | -11.49 / 4.67 | -20.47..-8.16 | 0/5 | -8.16/-20.47/-11.68/-8.82/-8.32 |
| embeds__trillsson | Pooled MLP | .7952 | embeds (.7919) | +0.33 / 5.99 | -9.05..+8.32 | 3/5 | +8.32/-2.73/-9.05/+4.56/+0.55 |
| embeds__trillsson | MD-Former | .7788 | embeds (.7944) | -1.56 / 6.70 | -10.28..+8.04 | 3/5 | +1.74/-7.94/-10.28/+8.04/+0.62 |
| embeds__trillsson__wav | Pooled MLP | .7662 | embeds (.7919) | -2.57 / 3.93 | -8.76..+3.24 | 1/5 | +3.24/-4.25/-8.76/-1.10/-1.99 |
| embeds__trillsson__wav | MD-Former | .7010 | embeds (.7944) | -9.34 / 5.40 | -17.86..-3.43 | 0/5 | -4.22/-17.86/-12.69/-3.43/-8.52 |
| embeds__wav | Pooled MLP | .7128 | embeds (.7919) | -7.91 / 2.16 | -11.73..-6.00 | 0/5 | -6.56/-6.39/-11.73/-6.00/-8.85 |
| embeds__wav | MD-Former | .6723 | embeds (.7944) | -12.21 / 1.56 | -13.99..-10.04 | 0/5 | -11.01/-13.99/-13.90/-12.09/-10.04 |
| trajectories__trillsson | Pooled MLP | .7712 | trillsson (.7802) | -0.90 / 1.09 | -2.25..+0.24 | 2/5 | -2.17/+0.24/+0.08/-2.25/-0.39 |
| trajectories__trillsson | MD-Former | .7547 | trillsson (.7601) | -0.54 / 2.54 | -4.44..+2.11 | 3/5 | +1.14/+2.11/+1.14/-4.44/-2.64 |
| trajectories__trillsson__wav | Pooled MLP | .7622 | trillsson (.7802) | -1.80 / 1.74 | -4.16..+1.00 | 1/5 | -4.16/-0.92/+1.00/-2.22/-2.69 |
| trajectories__trillsson__wav | MD-Former | .7526 | trillsson (.7601) | -0.75 / 2.38 | -2.80..+3.66 | 1/5 | -2.08/+3.66/-0.16/-2.80/-2.35 |
| trajectories__wav | Pooled MLP | .7018 | wav (.6832) | +1.87 / 4.27 | -3.71..+7.64 | 3/5 | +7.64/-3.71/+0.03/-0.63/+6.01 |
| trajectories__wav | MD-Former | .6571 | wav (.6791) | -2.21 / 2.91 | -6.44..+1.96 | 1/5 | -3.10/-6.44/+1.96/-3.44/-0.01 |
| trillsson__wav | Pooled MLP | .7785 | trillsson (.7802) | -0.17 / 2.69 | -3.98..+2.93 | 3/5 | -3.98/+1.85/+1.07/-2.71/+2.93 |
| trillsson__wav | MD-Former | .7304 | trillsson (.7601) | -2.96 / 3.64 | -9.81..+0.94 | 1/5 | +0.94/-2.54/-1.11/-9.81/-2.30 |

### Every constituent comparison

The following is the complete 56-comparison audit. It reports every constituent,
not only the strongest one. Values are paired fold differences in pp, followed
by their five-fold mean.

| Subset / model | Constituent | Fold differences pp | Mean pp |
|---|---|---|---:|
| embeds__trajectories / MLP | embeds | +0.97/-0.30/-0.14/-0.76/-0.40 | -0.13 |
| embeds__trajectories / MLP | trajectories | +25.58/+32.77/+28.18/+25.17/+32.61 | +28.86 |
| embeds__trajectories / MD | embeds | +1.09/-2.62/-1.94/-0.04/+2.28 | -0.25 |
| embeds__trajectories / MD | trajectories | +27.59/+32.42/+29.14/+23.99/+32.83 | +29.19 |
| embeds__trillsson / MLP | embeds | +8.32/-2.73/-9.05/+4.56/+0.55 | +0.33 |
| embeds__trillsson / MLP | trillsson | +0.99/+2.29/-0.65/+1.86/+3.00 | +1.50 |
| embeds__trillsson / MD | embeds | +1.74/-7.94/-10.28/+8.04/+0.62 | -1.56 |
| embeds__trillsson / MD | trillsson | +0.48/+3.00/+0.94/+4.25/+0.67 | +1.87 |
| embeds__wav / MLP | embeds | -6.56/-6.39/-11.73/-6.00/-8.85 | -7.91 |
| embeds__wav / MLP | wav | +3.74/+7.98/-1.72/-0.11/+4.94 | +2.97 |
| embeds__wav / MD | embeds | -11.01/-13.99/-13.90/-12.09/-10.04 | -12.21 |
| embeds__wav / MD | wav | -0.36/-0.65/+1.11/-3.82/+0.33 | -0.68 |
| trajectories__trillsson / MLP | trajectories | +29.77/+28.29/+20.00/+26.38/+30.17 | +26.92 |
| trajectories__trillsson / MLP | trillsson | -2.17/+0.24/+0.08/-2.25/-0.39 | -0.90 |
| trajectories__trillsson / MD | trajectories | +28.90/+26.21/+21.00/+23.38/+27.86 | +25.47 |
| trajectories__trillsson / MD | trillsson | +1.14/+2.11/+1.14/-4.44/-2.64 | -0.54 |
| trajectories__wav / MLP | trajectories | +21.95/+14.99/+18.34/+19.41/+25.23 | +19.98 |
| trajectories__wav / MLP | wav | +7.64/-3.71/+0.03/-0.63/+6.01 | +1.87 |
| trajectories__wav / MD | trajectories | +12.75/+15.26/+18.03/+12.32/+20.17 | +15.71 |
| trajectories__wav / MD | wav | -3.10/-6.44/+1.96/-3.44/-0.01 | -2.21 |
| trillsson__wav / MLP | trillsson | -3.98/+1.85/+1.07/-2.71/+2.93 | -0.17 |
| trillsson__wav / MLP | wav | +13.65/+11.20/+2.68/+5.88/+14.27 | +9.54 |
| trillsson__wav / MD | trillsson | +0.94/-2.54/-1.11/-9.81/-2.30 | -2.96 |
| trillsson__wav / MD | wav | +12.85/-0.14/+2.68/+2.25/+8.02 | +5.13 |
| embeds__trajectories__trillsson / MLP | embeds | +1.79/-7.57/-8.84/-1.77/-6.20 | -4.52 |
| embeds__trajectories__trillsson / MLP | trajectories | +26.40/+25.50/+19.48/+24.16/+26.81 | +24.47 |
| embeds__trajectories__trillsson / MLP | trillsson | -5.54/-2.55/-0.44/-4.47/-3.75 | -3.35 |
| embeds__trajectories__trillsson / MD | embeds | -0.43/-7.90/-10.44/+4.19/-5.66 | -4.05 |
| embeds__trajectories__trillsson / MD | trajectories | +26.07/+27.14/+20.64/+28.22/+24.89 | +25.39 |
| embeds__trajectories__trillsson / MD | trillsson | -1.69/+3.04/+0.78/+0.40/-5.61 | -0.62 |
| embeds__trajectories__wav / MLP | embeds | -2.87/-7.07/-6.08/-9.07/-3.94 | -5.81 |
| embeds__trajectories__wav / MLP | trajectories | +21.74/+26.00/+22.24/+16.86/+29.07 | +23.18 |
| embeds__trajectories__wav / MLP | wav | +7.43/+7.30/+3.93/-3.18/+9.85 | +5.07 |
| embeds__trajectories__wav / MD | embeds | -8.16/-20.47/-11.68/-8.82/-8.32 | -11.49 |
| embeds__trajectories__wav / MD | trajectories | +18.34/+14.57/+19.40/+15.21/+22.23 | +17.95 |
| embeds__trajectories__wav / MD | wav | +2.49/-7.13/+3.33/-0.55/+2.05 | +0.04 |
| embeds__trillsson__wav / MLP | embeds | +3.24/-4.25/-8.76/-1.10/-1.99 | -2.57 |
| embeds__trillsson__wav / MLP | trillsson | -4.09/+0.77/-0.36/-3.80/+0.46 | -1.40 |
| embeds__trillsson__wav / MLP | wav | +13.54/+10.12/+1.25/+4.79/+11.80 | +8.30 |
| embeds__trillsson__wav / MD | embeds | -4.22/-17.86/-12.69/-3.43/-8.52 | -9.34 |
| embeds__trillsson__wav / MD | trillsson | -5.48/-6.92/-1.47/-7.22/-8.47 | -5.91 |
| embeds__trillsson__wav / MD | wav | +6.43/-4.52/+2.32/+4.84/+1.85 | +2.18 |
| trajectories__trillsson__wav / MLP | trajectories | +27.78/+27.13/+20.92/+26.41/+27.87 | +26.02 |
| trajectories__trillsson__wav / MLP | trillsson | -4.16/-0.92/+1.00/-2.22/-2.69 | -1.80 |
| trajectories__trillsson__wav / MLP | wav | +13.47/+8.43/+2.61/+6.37/+8.65 | +7.91 |
| trajectories__trillsson__wav / MD | trajectories | +25.68/+27.76/+19.70/+25.02/+28.15 | +25.26 |
| trajectories__trillsson__wav / MD | trillsson | -2.08/+3.66/-0.16/-2.80/-2.35 | -0.75 |
| trajectories__trillsson__wav / MD | wav | +9.83/+6.06/+3.63/+9.26/+7.97 | +7.35 |
| all four / MLP | embeds | -0.05/-3.64/-7.07/-1.86/+0.74 | -2.38 |
| all four / MLP | trajectories | +24.56/+29.43/+21.25/+24.07/+33.75 | +26.61 |
| all four / MLP | trillsson | -7.38/+1.38/+1.33/-4.56/+3.19 | -1.21 |
| all four / MLP | wav | +10.25/+10.73/+2.94/+4.03/+14.53 | +8.50 |
| all four / MD | embeds | +0.35/-19.50/-11.19/-5.44/-1.07 | -7.37 |
| all four / MD | trajectories | +26.85/+15.54/+19.89/+18.59/+29.48 | +22.07 |
| all four / MD | trillsson | -0.91/-8.56/+0.03/-9.23/-1.02 | -3.94 |
| all four / MD | wav | +11.00/-6.16/+3.82/+2.83/+9.30 | +4.16 |

## Diagnostic metric audit

For diagnostic review, the three highest fusion groups by five-fold validation
mean were inspected against their strongest constituent: `embeds__trillsson /`
Pooled MLP (`.7952` vs embeds), `embeds__trajectories /` MD-Former (`.7919` vs
embeds), and `embeds__trajectories /` Pooled MLP (`.7906` vs embeds). At each
selected epoch, the exact per-corpus/per-task metrics were checked for CARE and
WSM, Depression and Parkinson: Macro Recall, Patient Recall, Healthy Recall, and
Macro F1. The audit found no missing or non-finite values.

The principal exact diagnostic observations are:

| Fusion/model/fold | Corpus/task | Fusion MR/PR/HR/F1 | Paired baseline MR/PR/HR/F1 |
|---|---|---|---|
| embeds__trillsson / MLP / 0 | CARE Depression | .8097/.7209/.8985/.8061 | .8439/.7442/.9436/.8536 |
| embeds__trillsson / MLP / 0 | CARE Parkinson | .8331/.7353/.9308/.7838 | .6266/.2941/.9591/.6468 |
| embeds__trillsson / MLP / 0 | WSM Depression | .7511/.7778/.7244/.7447 | .7768/.7425/.8110/.7767 |
| embeds__trillsson / MLP / 0 | WSM Parkinson | .9232/.9083/.9381/.9235 | .7371/.5764/.8978/.7435 |
| embeds__trillsson / MLP / 1 | CARE Depression | .7991/.7912/.8071/.7685 | .8825/.8791/.8858/.8592 |
| embeds__trillsson / MLP / 1 | CARE Parkinson | .7750/.6154/.9346/.7603 | .8683/.7692/.9673/.8642 |
| embeds__trillsson / MLP / 2 | CARE Depression | .6591/.3908/.9274/.6777 | .7741/.6207/.9274/.7897 |
| embeds__trillsson / MLP / 2 | CARE Parkinson | .5612/.1290/.9934/.5852 | .8305/.6774/.9836/.8562 |
| embeds__trillsson / MLP / 3 | CARE Depression | .9111/.9146/.9076/.8855 | .8420/.7683/.9157/.8392 |
| embeds__trillsson / MLP / 3 | CARE Parkinson | .6942/.4688/.9197/.6766 | .6172/.2812/.9532/.6332 |
| embeds__trillsson / MLP / 4 | CARE Depression | .8955/.8333/.9577/.9013 | .8058/.7000/.9115/.8102 |
| embeds__trillsson / MLP / 4 | CARE Parkinson | .8297/.7353/.9241/.7745 | .8595/.8235/.8956/.7625 |
| embeds__trillsson / MLP / 0..4 | WSM tasks | exact values checked | exact values checked |
| embeds__trajectories / MD / 0 | CARE Depression | .8692/.8023/.9361/.8692 | .8790/.8256/.9323/.8745 |
| embeds__trajectories / MD / 0 | CARE Parkinson | .6603/.4118/.9088/.6427 | .6466/.3529/.9403/.6525 |
| embeds__trajectories / MD / 1 | CARE Parkinson | .8105/.6667/.9542/.8070 | .9163/.8718/.9608/.8860 |
| embeds__trajectories / MD / 2 | CARE Parkinson | .7508/.5806/.9211/.7156 | .8012/.6452/.9572/.7927 |
| embeds__trajectories / MD / 3 | CARE Parkinson | .5927/.2188/.9666/.6143 | .6289/.2812/.9766/.6631 |
| embeds__trajectories / MD / 4 | CARE Parkinson | .8580/.8235/.8924/.7586 | .7455/.6176/.8734/.6773 |

For the remaining WSM rows and the third diagnostic group, the same four metrics
were present and finite in both sides of every selected fold; the full runtime
summary rows are the source of record. The compact diagnostic table above retains
the exact low-recall cases that determine the scientific interpretation.

Before evaluating diagnostics, the following descriptive flags were fixed:

* `patient-collapse`: Patient Recall `< 0.50`;
* `healthy-collapse`: Healthy Recall `< 0.50`;
* `class-imbalance`: `abs(Patient Recall - Healthy Recall) >= 0.40`.

These are audit flags, not selection criteria. The fusion examples include
patient-collapse and class-imbalance: for example, CARE Parkinson fold 2 of
`embeds__trillsson / MLP` has PR `.1290`, HR `.9934`, and MR `.5612`; CARE
Parkinson fold 3 of `embeds__trajectories / MD` has PR `.2188`, HR `.9666`, and
MR `.5927`. Thus a higher aggregate recall cannot be interpreted as uniformly
balanced disease performance.

## Representation and model readiness

The minimal defensible static contract is independent arithmetic mean pooling of
each observed `[T,D]` family over its valid turns, producing `[1,D]`, followed by
ordered concatenation. It is a pooled-static representation, not raw-native
fusion. The proposed component dimensions are:

| Component | Dimension |
|---|---:|
| `emotionmsp_embeds_mean` | 1,024 |
| `emotionmsp_trajectories_mean` | 3 |
| `trillsson4_1_mean` | 1,024 |
| `wav2vec_xlsr300m_mean` | 1,024 |

Under this explicit contract, proposed names and dimensions are:

| Name | Dimension |
|---|---:|
| `emotionmsp_embeds` | 1,024 |
| `emotionmsp_trajectories` | 3 |
| `trillsson4_1` | 1,024 |
| `wav2vec_xlsr300m` | 1,024 |
| `emotionmsp_embeds__emotionmsp_trajectories` | 1,027 |
| `emotionmsp_embeds__trillsson4_1` | 2,048 |
| `emotionmsp_embeds__wav2vec_xlsr300m` | 2,048 |
| `emotionmsp_trajectories__trillsson4_1` | 1,027 |
| `emotionmsp_trajectories__wav2vec_xlsr300m` | 1,027 |
| `trillsson4_1__wav2vec_xlsr300m` | 2,048 |
| `emotionmsp_embeds__emotionmsp_trajectories__trillsson4_1` | 2,051 |
| `emotionmsp_embeds__emotionmsp_trajectories__wav2vec_xlsr300m` | 2,051 |
| `emotionmsp_embeds__trillsson4_1__wav2vec_xlsr300m` | 3,072 |
| `emotionmsp_trajectories__trillsson4_1__wav2vec_xlsr300m` | 2,051 |
| `emotionmsp_embeds__emotionmsp_trajectories__trillsson4_1__wav2vec_xlsr300m` | 3,075 |

Existing Pooled MLP and MD-Former can consume this representation through their
existing static input path, subject to the existing config/data contract. No
source change is justified by the audit. They cannot consume raw variable-length
four-family sequences as a single static vector without a new temporal batching
and alignment contract. Therefore no temporal-native model comparison is
justified at this stage. A future temporal comparison requires timestamp evidence
or an explicitly authorized turn-alignment contract and must not repeat static
vectors along a fabricated time axis.

## Integrity checks and ambiguities

* Run counts were exactly 40 baseline and 110 fusion; all identifiers were unique
  within their series and each expected model/fold group was present.
* Selected epochs had corresponding checkpoint artifacts, and all selected
  validation metrics inspected were finite. TEST values were not used for any
  selection or comparison.
* Recomputing the arithmetic mean of the four selected validation components gave
  40/40 baseline and 110/110 fusion values within absolute error `0.00015` of
  `dev_mean_macro_recall`; maximum error was `0.000075` in each series. The small
  discrepancy is consistent with four-decimal summary rounding. Strict `1e-5`
  equality is therefore not an appropriate invariant for these summaries.
* Profiles, split identities, fold identities, corpus identities, and observed
  label masks were retained from the canonical metadata. The fixed CARE TEST and
  WSM TEST populations were unchanged.
* The main scientific ambiguity is temporal: equal turn IDs do not prove equal
  timestamps across all families. A second ambiguity is whether future work wants
  `mean` only or the registry's `mean+std`; these are different representation
  contracts and dimensions.
* The paired results do not establish a performance claim. `embeds__trillsson /
  Pooled MLP` is only `+0.33 pp` over its strongest constituent on average and
  improves 3/5 folds, with a 5-fold SD of `5.99 pp`; it is not a consistent
  improvement. Adding trajectories to the strong embeds baseline is generally
  neutral-to-negative, while combinations involving the weak trajectories-only
  baseline can look positive without demonstrating incremental value over the
  strongest constituent. No significance test or TEST-based conclusion was made.
