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
timestamps. Therefore the approved native turn alignment must not be interpreted
as frame-synchronous or second-level timestamp alignment. Native turn concatenation
is justified by identical ordered turn IDs; a separate frame-level contract is not
established by these artifacts. CARE has variable `T`; WSM is single-turn in the live artifacts.

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
| embeds__trajectories | Pooled MLP | .7906 | embeds (.7919) | -0.13 / 0.65 | -0.76..+0.97 | 1/5 | +0.97/-0.30/-0.14/-0.76/-0.40 |
| embeds__trajectories | MD-Former | .7919 | embeds (.7944) | -0.25 / 2.04 | -2.62..+2.28 | 2/5 | +1.09/-2.62/-1.94/-0.04/+2.28 |
| embeds__trajectories__trillsson | Pooled MLP | .7467 | embeds (.7919) | -4.52 / 4.42 | -8.84..+1.79 | 1/5 | +1.79/-7.57/-8.84/-1.77/-6.20 |
| embeds__trajectories__trillsson | MD-Former | .7539 | embeds (.7944) | -4.05 / 5.90 | -10.44..+4.19 | 1/5 | -0.43/-7.90/-10.44/+4.19/-5.66 |
| embeds__trajectories__trillsson__wav | Pooled MLP | .7681 | embeds (.7919) | -2.38 / 3.12 | -7.07..+0.74 | 1/5 | -0.05/-3.64/-7.07/-1.86/+0.74 |
| embeds__trajectories__trillsson__wav | MD-Former | .7207 | embeds (.7944) | -7.37 / 8.14 | -19.50..-1.07 | 1/5 | +0.35/-19.50/-11.19/-5.44/-1.07 |
| embeds__trajectories__wav | Pooled MLP | .7338 | embeds (.7919) | -5.81 / 2.47 | -9.07..-2.87 | 0/5 | -2.87/-7.07/-6.08/-9.07/-3.94 |
| embeds__trajectories__wav | MD-Former | .6795 | embeds (.7944) | -11.49 / 5.22 | -20.47..-8.16 | 0/5 | -8.16/-20.47/-11.68/-8.82/-8.32 |
| embeds__trillsson | Pooled MLP | .7952 | embeds (.7919) | +0.33 / 6.69 | -9.05..+8.32 | 3/5 | +8.32/-2.73/-9.05/+4.56/+0.55 |
| embeds__trillsson | MD-Former | .7788 | embeds (.7944) | -1.56 / 7.49 | -10.28..+8.04 | 3/5 | +1.74/-7.94/-10.28/+8.04/+0.62 |
| embeds__trillsson__wav | Pooled MLP | .7662 | embeds (.7919) | -2.57 / 4.40 | -8.76..+3.24 | 1/5 | +3.24/-4.25/-8.76/-1.10/-1.99 |
| embeds__trillsson__wav | MD-Former | .7010 | embeds (.7944) | -9.34 / 6.03 | -17.86..-3.43 | 0/5 | -4.22/-17.86/-12.69/-3.43/-8.52 |
| embeds__wav | Pooled MLP | .7128 | embeds (.7919) | -7.91 / 2.41 | -11.73..-6.00 | 0/5 | -6.56/-6.39/-11.73/-6.00/-8.85 |
| embeds__wav | MD-Former | .6723 | embeds (.7944) | -12.21 / 1.75 | -13.99..-10.04 | 0/5 | -11.01/-13.99/-13.90/-12.09/-10.04 |
| trajectories__trillsson | Pooled MLP | .7712 | trillsson (.7802) | -0.90 / 1.22 | -2.25..+0.24 | 2/5 | -2.17/+0.24/+0.08/-2.25/-0.39 |
| trajectories__trillsson | MD-Former | .7547 | trillsson (.7601) | -0.54 / 2.84 | -4.44..+2.11 | 3/5 | +1.14/+2.11/+1.14/-4.44/-2.64 |
| trajectories__trillsson__wav | Pooled MLP | .7622 | trillsson (.7802) | -1.80 / 1.95 | -4.16..+1.00 | 1/5 | -4.16/-0.92/+1.00/-2.22/-2.69 |
| trajectories__trillsson__wav | MD-Former | .7526 | trillsson (.7601) | -0.75 / 2.66 | -2.80..+3.66 | 1/5 | -2.08/+3.66/-0.16/-2.80/-2.35 |
| trajectories__wav | Pooled MLP | .7018 | wav (.6832) | +1.87 / 4.78 | -3.71..+7.64 | 3/5 | +7.64/-3.71/+0.03/-0.63/+6.01 |
| trajectories__wav | MD-Former | .6571 | wav (.6791) | -2.21 / 3.26 | -6.44..+1.96 | 1/5 | -3.10/-6.44/+1.96/-3.44/-0.01 |
| trillsson__wav | Pooled MLP | .7785 | trillsson (.7802) | -0.17 / 3.01 | -3.98..+2.93 | 3/5 | -3.98/+1.85/+1.07/-2.71/+2.93 |
| trillsson__wav | MD-Former | .7304 | trillsson (.7601) | -2.96 / 4.07 | -9.81..+0.94 | 1/5 | +0.94/-2.54/-1.11/-9.81/-2.30 |

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

The three leading fusion groups by five-fold validation mean were inspected at
selected epochs against their strongest matched single-family baseline. The
following table contains all 60 required corpus/task/fold comparisons. Values are
ordered `Macro Recall / Patient Recall / Healthy Recall / Macro F1`; both sides
are the actual selected-epoch validation values from the runtime summaries.

| Fusion group / model | Fold | Corpus/task | Fusion MR/PR/HR/F1 | Matched baseline MR/PR/HR/F1 |
|---|---:|---|---|---|
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 0 | CARE Depression | 0.8097/0.7209/0.8985/0.8061 | 0.8439/0.7442/0.9436/0.8536 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 0 | CARE Parkinson | 0.8331/0.7353/0.9308/0.7838 | 0.6266/0.2941/0.9591/0.6468 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 0 | WSM Depression | 0.7511/0.7778/0.7244/0.7447 | 0.7768/0.7425/0.8110/0.7767 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 0 | WSM Parkinson | 0.9232/0.9083/0.9381/0.9235 | 0.7371/0.5764/0.8978/0.7435 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 1 | CARE Depression | 0.7991/0.7912/0.8071/0.7685 | 0.8825/0.8791/0.8858/0.8592 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 1 | CARE Parkinson | 0.7750/0.6154/0.9346/0.7603 | 0.8683/0.7692/0.9673/0.8642 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 1 | WSM Depression | 0.7308/0.7159/0.7456/0.7271 | 0.7030/0.6348/0.7712/0.7039 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 1 | WSM Parkinson | 0.9135/0.9035/0.9235/0.9127 | 0.8741/0.8246/0.9235/0.8777 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 2 | CARE Depression | 0.6591/0.3908/0.9274/0.6777 | 0.7741/0.6207/0.9274/0.7897 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 2 | CARE Parkinson | 0.5612/0.1290/0.9934/0.5852 | 0.8305/0.6774/0.9836/0.8562 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 2 | WSM Depression | 0.6771/0.5610/0.7932/0.6800 | 0.7092/0.6192/0.7992/0.7117 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 2 | WSM Parkinson | 0.9063/0.8559/0.9567/0.9111 | 0.8517/0.8428/0.8607/0.8497 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 3 | CARE Depression | 0.9111/0.9146/0.9076/0.8855 | 0.8420/0.7683/0.9157/0.8392 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 3 | CARE Parkinson | 0.6942/0.4688/0.9197/0.6766 | 0.6172/0.2812/0.9532/0.6332 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 3 | WSM Depression | 0.7182/0.7297/0.7067/0.7113 | 0.7391/0.7558/0.7224/0.7317 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 3 | WSM Parkinson | 0.8966/0.9277/0.8654/0.8915 | 0.8394/0.8715/0.8073/0.8340 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 4 | CARE Depression | 0.8955/0.8333/0.9577/0.9013 | 0.8058/0.7000/0.9115/0.8102 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 4 | CARE Parkinson | 0.8297/0.7353/0.9241/0.7745 | 0.8595/0.8235/0.8956/0.7625 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 4 | WSM Depression | 0.7619/0.6948/0.8291/0.7641 | 0.7792/0.7529/0.8055/0.7773 |
| emotionmsp_embeds__trillsson4_1 / Pooled MLP | 4 | WSM Parkinson | 0.8571/0.8509/0.8634/0.8548 | 0.8780/0.8772/0.8789/0.8753 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 0 | CARE Depression | 0.8692/0.8023/0.9361/0.8692 | 0.8790/0.8256/0.9323/0.8745 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 0 | CARE Parkinson | 0.6603/0.4118/0.9088/0.6427 | 0.6466/0.3529/0.9403/0.6525 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 0 | WSM Depression | 0.7915/0.7995/0.7835/0.7873 | 0.7624/0.7886/0.7362/0.7561 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 0 | WSM Parkinson | 0.7825/0.7074/0.8576/0.7860 | 0.7720/0.7205/0.8235/0.7732 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 1 | CARE Depression | 0.9159/0.9341/0.8976/0.8880 | 0.9084/0.9231/0.8937/0.8810 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 1 | CARE Parkinson | 0.8105/0.6667/0.9542/0.8070 | 0.9163/0.8718/0.9608/0.8860 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 1 | WSM Depression | 0.7224/0.6145/0.8304/0.7265 | 0.7312/0.6261/0.8363/0.7354 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 1 | WSM Parkinson | 0.8479/0.8333/0.8624/0.8462 | 0.8458/0.7895/0.9021/0.8494 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 2 | CARE Depression | 0.8019/0.7126/0.8911/0.7996 | 0.8525/0.7816/0.9234/0.8525 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 2 | CARE Parkinson | 0.7508/0.5806/0.9211/0.7156 | 0.8012/0.6452/0.9572/0.7927 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 2 | WSM Depression | 0.7619/0.7326/0.7913/0.7603 | 0.7246/0.6599/0.7893/0.7257 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 2 | WSM Parkinson | 0.8511/0.9127/0.7895/0.8395 | 0.8648/0.8690/0.8607/0.8613 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 3 | CARE Depression | 0.8122/0.8171/0.8072/0.7724 | 0.7766/0.5854/0.9679/0.8077 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 3 | CARE Parkinson | 0.5927/0.2188/0.9666/0.6143 | 0.6289/0.2812/0.9766/0.6631 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 3 | WSM Depression | 0.7207/0.7878/0.6535/0.7062 | 0.7393/0.6657/0.8130/0.7415 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 3 | WSM Parkinson | 0.8340/0.8514/0.8165/0.8300 | 0.8165/0.8072/0.8257/0.8151 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 4 | CARE Depression | 0.8028/0.7556/0.8500/0.7845 | 0.8152/0.7111/0.9192/0.8210 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 4 | CARE Parkinson | 0.8580/0.8235/0.8924/0.7586 | 0.7455/0.6176/0.8734/0.6773 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 4 | WSM Depression | 0.7729/0.7442/0.8016/0.7712 | 0.7633/0.7151/0.8114/0.7635 |
| emotionmsp_embeds__emotionmsp_trajectories / MD-Former | 4 | WSM Parkinson | 0.8795/0.8553/0.9037/0.8800 | 0.8981/0.8553/0.9410/0.9017 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 0 | CARE Depression | 0.8249/0.7326/0.9173/0.8262 | 0.8439/0.7442/0.9436/0.8536 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 0 | CARE Parkinson | 0.6319/0.3235/0.9403/0.6391 | 0.6266/0.2941/0.9591/0.6468 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 0 | WSM Depression | 0.7712/0.6802/0.8622/0.7754 | 0.7768/0.7425/0.8110/0.7767 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 0 | WSM Parkinson | 0.7952/0.6987/0.8916/0.8009 | 0.7371/0.5764/0.8978/0.7435 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 1 | CARE Depression | 0.8868/0.8681/0.9055/0.8709 | 0.8825/0.8791/0.8858/0.8592 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 1 | CARE Parkinson | 0.8203/0.6667/0.9739/0.8391 | 0.8683/0.7692/0.9673/0.8642 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 1 | WSM Depression | 0.7391/0.6696/0.8087/0.7409 | 0.7030/0.6348/0.7712/0.7039 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 1 | WSM Parkinson | 0.8693/0.8333/0.9052/0.8710 | 0.8741/0.8246/0.9235/0.8777 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 2 | CARE Depression | 0.7551/0.5747/0.9355/0.7755 | 0.7741/0.6207/0.9274/0.7897 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 2 | CARE Parkinson | 0.7916/0.6129/0.9704/0.8048 | 0.8305/0.6774/0.9836/0.8562 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 2 | WSM Depression | 0.7271/0.6192/0.8350/0.7313 | 0.7092/0.6192/0.7992/0.7117 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 2 | WSM Parkinson | 0.8862/0.8777/0.8947/0.8848 | 0.8517/0.8428/0.8607/0.8497 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 3 | CARE Depression | 0.8321/0.7927/0.8715/0.8124 | 0.8420/0.7683/0.9157/0.8392 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 3 | CARE Parkinson | 0.6099/0.2500/0.9699/0.6363 | 0.6172/0.2812/0.9532/0.6332 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 3 | WSM Depression | 0.7386/0.7645/0.7126/0.7299 | 0.7391/0.7558/0.7224/0.7317 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 3 | WSM Parkinson | 0.8266/0.8795/0.7737/0.8188 | 0.8394/0.8715/0.8073/0.8340 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 4 | CARE Depression | 0.7624/0.6556/0.8692/0.7595 | 0.8058/0.7000/0.9115/0.8102 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 4 | CARE Parkinson | 0.8878/0.9118/0.8639/0.7482 | 0.8595/0.8235/0.8956/0.7625 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 4 | WSM Depression | 0.7834/0.7180/0.8487/0.7859 | 0.7792/0.7529/0.8055/0.7773 |
| emotionmsp_embeds__emotionmsp_trajectories / Pooled MLP | 4 | WSM Parkinson | 0.8726/0.8509/0.8944/0.8726 | 0.8780/0.8772/0.8789/0.8753 |

The descriptive imbalance flags were fixed before evaluation and were not used
for selection: `patient-collapse` means Patient Recall `< 0.50`,
`healthy-collapse` means Healthy Recall `< 0.50`, and `class-imbalance` means
`abs(Patient Recall - Healthy Recall) >= 0.40`. For example, CARE Parkinson fold
2 of `embeds__trillsson / Pooled MLP` has PR `.1290`, HR `.9934`, and MR `.5612`;
CARE Parkinson fold 3 of `embeds__trajectories / MD-Former` has PR `.2188`, HR
`.9666`, and MR `.5927`. These are descriptive warnings, not feature-selection
criteria.

## Representation and model readiness

The completed DEEP-004 through DEEP-006 experiments use native turn-aligned
fusion. For each family the Dataset loads `[T,D]` values and ordered `turn_ids`,
requires identical ordered turn-ID sequences, and concatenates corresponding
turns to `[T,D1+D2]`. This is turn-level alignment, not frame-synchronous or
second-level timestamp alignment. The audited artifacts support matching turn
IDs, but do not claim a common frame rate or shared wall-clock timestamps.

The Dataset already implements this alignment and variable-length batching. Both
Pooled MLP and MD-Former completed real experiments on the resulting fused
native-turn sequences. The 110-run comparison is therefore scientifically
interpretable as native turn-aligned feature fusion and must not be retrospectively
reclassified as invalid.

A static control could independently pool each family over its turns and then
concatenate the pooled vectors, with dimensions 1,024, 3, 1,024, and 1,024 and
subset dimensions 1,027, 2,048, 2,051, 3,072, or 3,075 as applicable. That is a
distinct future pooled-static experiment, not the representation used here and
not an authorized replacement. Raw turn alignment and frame-level timestamp
synchronization remain different scientific claims.

Existing Pooled MLP and MD-Former consumed the approved native-turn fused input
without source changes in DEEP-006. No model-readiness blocker was identified.

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
* Turn IDs establish the approved native turn alignment, but they do not establish
  frame-level timestamp synchronization across all families. A second ambiguity is whether future work wants
  `mean` only or the registry's `mean+std`; these are different representation
  contracts and dimensions.
* The paired results do not establish a performance claim. `embeds__trillsson /
  Pooled MLP` is only `+0.33 pp` over its strongest constituent on average and
  improves 3/5 folds, with the corrected sample SD of `6.69 pp`; it is not a consistent
  improvement. Adding trajectories to the strong embeds baseline is generally
  neutral-to-negative, while combinations involving the weak trajectories-only
  baseline can look positive without demonstrating incremental value over the
  strongest constituent. No significance test or TEST-based conclusion was made.
