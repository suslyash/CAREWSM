# CAREWSM audio experiment report

Date: 2026-11-04
Branch: codex/manager-report-2026-11-04

## Executive summary

Complete audio screening matrix. Each fold used joint CARE+WSM data; checkpoint selection used only joint/average_macro_f1; CARE TEST and WSM TEST were evaluated after every epoch. Aggregates use each fold best validation epoch.

- Canonical runs: **110/110**.
- summary.csv files: **111**; duplicate historical runs excluded: **1**.
- Values are percentages; mean +/- std is across folds.
- Average is arithmetic CARE/WSM mean, not pooled predictions.

## Ranking by TEST Average Macro F1

| Rank | Configuration | Depression | PD | Overall | Mean best epoch | VAL monitor |
|---:|---|---:|---:|---:|---:|---:|
| 1 | emotionmsp_embeds/mlp | 76.35 | 73.21 | 74.78 | 34.2 | 0.7941 |
| 2 | emotionmsp_embeds/mdformer | 76.10 | 70.80 | 73.45 | 16.2 | 0.7882 |
| 3 | trillsson4_1/mlp | 73.97 | 70.61 | 72.29 | 19.0 | 0.7621 |
| 4 | trillsson4_1/mdformer | 70.52 | 68.89 | 69.70 | 20.0 | 0.7559 |
| 5 | wav2vec_xlsr300m/mlp | 64.23 | 65.27 | 64.75 | 34.2 | 0.7118 |
| 6 | wav2vec_xlsr300m/mdformer | 56.55 | 64.48 | 60.52 | 23.8 | 0.6668 |
| 7 | disvoice_articulation_static/mlp | 55.30 | 57.40 | 56.35 | 33.2 | 0.6186 |
| 8 | disvoice_phonation_dynamic/mdformer | 62.47 | 48.44 | 55.46 | 14.0 | 0.5344 |
| 9 | disvoice_phonation_dynamic/mptsnet | 64.79 | 45.72 | 55.26 | 14.4 | 0.5279 |
| 10 | disvoice_prosody_static/mdformer | 56.69 | 52.57 | 54.63 | 23.6 | 0.5937 |
| 11 | egemaps/mptsnet | 47.76 | 61.41 | 54.58 | 23.2 | 0.5063 |
| 12 | egemaps_func/mdformer | 43.83 | 58.54 | 51.18 | 30.2 | 0.4918 |
| 13 | disvoice_prosody_static/mlp | 51.01 | 50.88 | 50.94 | 20.4 | 0.5253 |
| 14 | disvoice_phonation_static/mdformer | 50.83 | 49.15 | 49.99 | 26.0 | 0.5354 |
| 15 | egemaps/mdformer | 43.92 | 55.82 | 49.87 | 16.8 | 0.4862 |
| 16 | disvoice_articulation_static/mdformer | 47.96 | 51.06 | 49.51 | 16.4 | 0.5220 |
| 17 | disvoice_prosody_dynamic/mdformer | 49.93 | 45.12 | 47.52 | 9.2 | 0.4529 |
| 18 | egemaps_func/mlp | 43.54 | 48.46 | 46.00 | 19.6 | 0.4657 |
| 19 | disvoice_phonation_static/mlp | 44.26 | 45.06 | 44.66 | 1.0 | 0.4265 |
| 20 | emotionmsp_trajectories/mlp | 41.61 | 44.81 | 43.21 | 2.4 | 0.4279 |
| 21 | disvoice_prosody_dynamic/mptsnet | 39.10 | 45.12 | 42.11 | 1.0 | 0.4111 |
| 22 | emotionmsp_trajectories/mdformer | 39.15 | 45.05 | 42.10 | 1.0 | 0.4102 |


## Full aggregate TEST results

Values are mean +/- std across folds, in percent.

| Configuration | Task | CARE Macro F1 | WSM Macro F1 | Average Macro F1 | CARE Macro Recall | WSM Macro Recall | Average Macro Recall | CARE Patient Recall | WSM Patient Recall | Average Patient Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| emotionmsp_embeds/mlp | Depression | 76.39 +/- 2.85 | 76.31 +/- 2.36 | 76.35 +/- 2.09 | 74.32 +/- 2.76 | 76.09 +/- 2.33 | 75.20 +/- 1.93 | 54.43 +/- 6.37 | 69.25 +/- 3.45 | 61.84 +/- 4.14 |
| emotionmsp_embeds/mlp | PD | 70.25 +/- 3.07 | 76.17 +/- 2.83 | 73.21 +/- 1.99 | 67.77 +/- 2.48 | 73.51 +/- 2.63 | 70.64 +/- 1.27 | 39.41 +/- 4.46 | 52.03 +/- 5.41 | 45.72 +/- 1.75 |
| emotionmsp_embeds/mdformer | Depression | 72.87 +/- 2.34 | 79.33 +/- 2.34 | 76.10 +/- 1.77 | 70.60 +/- 2.10 | 79.34 +/- 1.90 | 74.97 +/- 1.39 | 45.36 +/- 3.65 | 75.34 +/- 3.92 | 60.35 +/- 0.71 |
| emotionmsp_embeds/mdformer | PD | 67.56 +/- 8.42 | 74.03 +/- 4.56 | 70.80 +/- 4.47 | 65.76 +/- 8.22 | 72.28 +/- 3.60 | 69.02 +/- 4.21 | 35.88 +/- 15.62 | 52.93 +/- 3.67 | 44.41 +/- 7.16 |
| trillsson4_1/mlp | Depression | 71.55 +/- 7.35 | 76.39 +/- 3.89 | 73.97 +/- 2.90 | 70.75 +/- 7.63 | 76.27 +/- 3.75 | 73.51 +/- 3.11 | 53.81 +/- 14.50 | 70.51 +/- 4.73 | 62.16 +/- 7.79 |
| trillsson4_1/mlp | PD | 73.04 +/- 3.97 | 68.17 +/- 2.87 | 70.61 +/- 1.71 | 72.13 +/- 3.85 | 67.89 +/- 3.14 | 70.01 +/- 2.64 | 49.41 +/- 9.16 | 50.08 +/- 8.48 | 49.74 +/- 8.34 |
| trillsson4_1/mdformer | Depression | 68.63 +/- 13.05 | 72.41 +/- 2.65 | 70.52 +/- 7.06 | 67.87 +/- 11.92 | 72.75 +/- 1.81 | 70.31 +/- 6.40 | 46.60 +/- 20.17 | 70.09 +/- 6.30 | 58.34 +/- 10.93 |
| trillsson4_1/mdformer | PD | 70.85 +/- 4.74 | 66.92 +/- 5.48 | 68.89 +/- 4.02 | 70.17 +/- 3.49 | 68.02 +/- 4.51 | 69.10 +/- 3.34 | 46.47 +/- 10.48 | 55.79 +/- 13.00 | 51.13 +/- 11.57 |
| wav2vec_xlsr300m/mlp | Depression | 62.32 +/- 8.49 | 66.14 +/- 2.15 | 64.23 +/- 4.37 | 62.17 +/- 6.45 | 66.16 +/- 2.29 | 64.16 +/- 3.43 | 30.31 +/- 13.45 | 58.21 +/- 7.57 | 44.26 +/- 9.47 |
| wav2vec_xlsr300m/mlp | PD | 61.00 +/- 2.15 | 69.55 +/- 3.89 | 65.27 +/- 1.81 | 61.09 +/- 2.99 | 69.36 +/- 3.23 | 65.23 +/- 1.94 | 30.00 +/- 8.67 | 53.39 +/- 5.77 | 41.69 +/- 5.90 |
| wav2vec_xlsr300m/mdformer | Depression | 47.11 +/- 5.18 | 65.99 +/- 1.47 | 56.55 +/- 2.10 | 48.81 +/- 5.44 | 66.38 +/- 1.94 | 57.59 +/- 1.99 | 15.26 +/- 5.57 | 61.49 +/- 11.40 | 38.38 +/- 7.11 |
| wav2vec_xlsr300m/mdformer | PD | 58.14 +/- 2.58 | 70.83 +/- 2.65 | 64.48 +/- 2.26 | 57.80 +/- 2.67 | 70.89 +/- 2.89 | 64.34 +/- 2.59 | 22.35 +/- 8.47 | 55.49 +/- 13.60 | 38.92 +/- 10.33 |
| disvoice_articulation_static/mlp | Depression | 45.84 +/- 3.77 | 64.76 +/- 3.70 | 55.30 +/- 1.32 | 49.20 +/- 3.21 | 64.77 +/- 3.77 | 56.98 +/- 0.91 | 9.48 +/- 4.27 | 57.13 +/- 6.90 | 33.30 +/- 4.24 |
| disvoice_articulation_static/mlp | PD | 55.52 +/- 4.21 | 59.27 +/- 3.43 | 57.40 +/- 1.72 | 55.06 +/- 3.44 | 63.83 +/- 1.47 | 59.44 +/- 1.32 | 14.71 +/- 8.58 | 64.58 +/- 5.83 | 39.64 +/- 6.22 |
| disvoice_phonation_dynamic/mdformer | Depression | 56.80 +/- 2.01 | 68.13 +/- 1.63 | 62.47 +/- 1.41 | 57.07 +/- 1.50 | 67.97 +/- 1.50 | 62.52 +/- 1.20 | 25.36 +/- 4.16 | 59.04 +/- 2.47 | 42.20 +/- 3.31 |
| disvoice_phonation_dynamic/mdformer | PD | 51.13 +/- 5.44 | 45.75 +/- 4.58 | 48.44 +/- 4.71 | 53.06 +/- 4.33 | 50.30 +/- 2.15 | 51.68 +/- 2.80 | 10.00 +/- 15.51 | 8.55 +/- 14.23 | 9.27 +/- 14.84 |
| disvoice_phonation_dynamic/mptsnet | Depression | 61.82 +/- 2.07 | 67.76 +/- 0.86 | 64.79 +/- 1.08 | 61.24 +/- 1.91 | 67.67 +/- 0.92 | 64.45 +/- 1.12 | 35.05 +/- 5.00 | 59.76 +/- 3.94 | 47.41 +/- 4.13 |
| disvoice_phonation_dynamic/mptsnet | PD | 47.18 +/- 0.00 | 44.26 +/- 1.32 | 45.72 +/- 0.66 | 50.00 +/- 0.00 | 50.61 +/- 0.64 | 50.30 +/- 0.32 | 0.00 +/- 0.00 | 1.22 +/- 1.28 | 0.61 +/- 0.64 |
| disvoice_prosody_static/mdformer | Depression | 60.60 +/- 1.70 | 52.78 +/- 2.41 | 56.69 +/- 1.85 | 60.31 +/- 1.17 | 53.99 +/- 1.38 | 57.15 +/- 1.18 | 22.93 +/- 2.78 | 30.82 +/- 8.22 | 26.87 +/- 5.09 |
| disvoice_prosody_static/mdformer | PD | 45.18 +/- 1.67 | 59.95 +/- 1.74 | 52.57 +/- 1.46 | 45.66 +/- 1.77 | 62.03 +/- 1.50 | 53.84 +/- 1.26 | 1.25 +/- 2.80 | 53.69 +/- 4.62 | 27.47 +/- 1.83 |
| egemaps/mptsnet | Depression | 38.35 +/- 1.74 | 57.16 +/- 6.28 | 47.76 +/- 2.43 | 44.43 +/- 3.99 | 59.29 +/- 4.06 | 51.86 +/- 1.26 | 0.62 +/- 1.38 | 28.95 +/- 12.05 | 14.79 +/- 6.49 |
| egemaps/mptsnet | PD | 61.79 +/- 9.87 | 61.02 +/- 3.52 | 61.41 +/- 6.44 | 60.27 +/- 8.68 | 60.15 +/- 2.85 | 60.21 +/- 5.64 | 22.94 +/- 18.53 | 24.06 +/- 7.29 | 23.50 +/- 12.63 |
| egemaps_func/mdformer | Depression | 40.87 +/- 0.43 | 46.78 +/- 5.90 | 43.83 +/- 3.01 | 48.48 +/- 1.54 | 53.15 +/- 2.01 | 50.81 +/- 0.66 | 1.03 +/- 1.78 | 12.84 +/- 8.86 | 6.93 +/- 5.17 |
| egemaps_func/mdformer | PD | 60.27 +/- 7.65 | 56.82 +/- 7.33 | 58.54 +/- 7.43 | 58.92 +/- 5.44 | 57.55 +/- 4.30 | 58.24 +/- 4.83 | 20.59 +/- 12.82 | 20.45 +/- 12.28 | 20.52 +/- 12.46 |
| disvoice_prosody_static/mlp | Depression | 55.10 +/- 12.43 | 46.92 +/- 9.28 | 51.01 +/- 10.61 | 57.63 +/- 7.26 | 53.38 +/- 3.42 | 55.51 +/- 5.04 | 18.54 +/- 17.13 | 15.52 +/- 14.83 | 17.03 +/- 15.60 |
| disvoice_prosody_static/mlp | PD | 43.74 +/- 1.30 | 58.01 +/- 9.71 | 50.88 +/- 5.17 | 51.76 +/- 10.21 | 59.99 +/- 6.57 | 55.87 +/- 2.03 | 32.50 +/- 44.72 | 50.63 +/- 9.98 | 41.57 +/- 26.67 |
| disvoice_phonation_static/mdformer | Depression | 44.58 +/- 3.34 | 57.09 +/- 5.28 | 50.83 +/- 3.81 | 48.97 +/- 1.78 | 59.33 +/- 3.04 | 54.15 +/- 1.65 | 8.08 +/- 7.28 | 28.80 +/- 11.99 | 18.44 +/- 9.24 |
| disvoice_phonation_static/mdformer | PD | 48.35 +/- 1.56 | 49.94 +/- 7.56 | 49.15 +/- 4.23 | 49.60 +/- 1.23 | 52.95 +/- 5.10 | 51.28 +/- 2.35 | 5.45 +/- 6.21 | 21.52 +/- 20.71 | 13.49 +/- 12.94 |
| egemaps/mdformer | Depression | 39.91 +/- 1.19 | 47.93 +/- 8.85 | 43.92 +/- 3.90 | 47.62 +/- 2.75 | 53.91 +/- 4.00 | 50.76 +/- 1.16 | 0.21 +/- 0.46 | 15.34 +/- 13.18 | 7.77 +/- 6.73 |
| egemaps/mdformer | PD | 53.80 +/- 11.53 | 57.84 +/- 5.33 | 55.82 +/- 8.24 | 56.53 +/- 13.24 | 58.27 +/- 4.18 | 57.40 +/- 8.62 | 15.88 +/- 30.82 | 20.45 +/- 12.53 | 18.17 +/- 21.53 |
| disvoice_articulation_static/mdformer | Depression | 40.48 +/- 0.53 | 55.44 +/- 16.58 | 47.96 +/- 8.07 | 49.00 +/- 1.07 | 60.37 +/- 9.50 | 54.68 +/- 4.29 | 0.00 +/- 0.00 | 33.59 +/- 30.72 | 16.80 +/- 15.36 |
| disvoice_articulation_static/mdformer | PD | 46.86 +/- 0.28 | 55.26 +/- 11.79 | 51.06 +/- 5.76 | 49.40 +/- 0.55 | 59.84 +/- 9.28 | 54.62 +/- 4.37 | 0.00 +/- 0.00 | 36.79 +/- 34.08 | 18.40 +/- 17.04 |
| disvoice_prosody_dynamic/mdformer | Depression | 47.28 +/- 3.63 | 52.58 +/- 8.71 | 49.93 +/- 6.08 | 50.98 +/- 1.08 | 56.02 +/- 3.47 | 53.50 +/- 1.99 | 10.72 +/- 5.99 | 25.57 +/- 15.41 | 18.14 +/- 10.55 |
| disvoice_prosody_dynamic/mdformer | PD | 47.25 +/- 0.00 | 42.98 +/- 0.00 | 45.12 +/- 0.00 | 50.00 +/- 0.00 | 50.00 +/- 0.00 | 50.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 |
| egemaps_func/mlp | Depression | 40.97 +/- 1.10 | 46.12 +/- 12.08 | 43.54 +/- 6.03 | 46.95 +/- 4.29 | 53.91 +/- 5.35 | 50.43 +/- 0.71 | 4.12 +/- 5.83 | 14.81 +/- 20.33 | 9.46 +/- 12.96 |
| egemaps_func/mlp | PD | 50.17 +/- 4.09 | 46.76 +/- 6.11 | 48.46 +/- 4.93 | 51.55 +/- 2.13 | 51.79 +/- 3.05 | 51.67 +/- 2.46 | 3.53 +/- 4.83 | 4.81 +/- 7.84 | 4.17 +/- 6.09 |
| disvoice_phonation_static/mlp | Depression | 44.46 +/- 0.32 | 44.06 +/- 0.76 | 44.26 +/- 0.54 | 50.67 +/- 0.11 | 51.61 +/- 0.25 | 51.14 +/- 0.18 | 4.47 +/- 0.47 | 8.38 +/- 1.22 | 6.43 +/- 0.84 |
| disvoice_phonation_static/mlp | PD | 47.14 +/- 0.00 | 42.99 +/- 0.00 | 45.06 +/- 0.00 | 50.00 +/- 0.00 | 50.00 +/- 0.00 | 50.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 |
| emotionmsp_trajectories/mlp | Depression | 40.80 +/- 0.23 | 42.42 +/- 1.65 | 41.61 +/- 0.72 | 49.59 +/- 0.46 | 46.33 +/- 1.55 | 47.96 +/- 0.94 | 0.00 +/- 0.00 | 14.57 +/- 7.25 | 7.28 +/- 3.62 |
| emotionmsp_trajectories/mlp | PD | 47.14 +/- 0.05 | 42.47 +/- 0.63 | 44.81 +/- 0.34 | 49.93 +/- 0.10 | 48.71 +/- 1.76 | 49.32 +/- 0.93 | 0.00 +/- 0.00 | 0.30 +/- 0.41 | 0.15 +/- 0.21 |
| disvoice_prosody_dynamic/mptsnet | Depression | 40.93 +/- 0.00 | 37.27 +/- 0.00 | 39.10 +/- 0.00 | 50.00 +/- 0.00 | 50.00 +/- 0.00 | 50.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 |
| disvoice_prosody_dynamic/mptsnet | PD | 47.25 +/- 0.00 | 42.98 +/- 0.00 | 45.12 +/- 0.00 | 50.00 +/- 0.00 | 50.00 +/- 0.00 | 50.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 |
| emotionmsp_trajectories/mdformer | Depression | 41.00 +/- 0.00 | 37.30 +/- 0.00 | 39.15 +/- 0.00 | 50.00 +/- 0.00 | 50.00 +/- 0.00 | 50.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 |
| emotionmsp_trajectories/mdformer | PD | 47.18 +/- 0.00 | 42.93 +/- 0.00 | 45.05 +/- 0.00 | 50.00 +/- 0.00 | 50.00 +/- 0.00 | 50.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 |


## Integrity and interpretation

| Check | Result |
|---|---|
| Canonical groups | 22 |
| Canonical fold runs | 110 |
| Groups with all five folds | 22/22 |
| Duplicate historical runs excluded | 1 |
| Failed markers in master log | 0 |
| Active training process | none |
| Selection monitor | joint/average_macro_f1 |
| TEST used for selection | no |

## Interpretation

1. Ranking is descriptive evidence; promotion must be confirmed from validation and fold artifacts.
2. Near-zero Patient Recall with near-constant Healthy Recall indicates majority-class/collapse behavior, even if Weighted F1 looks acceptable.
3. Zero or very small fold std can be real when all folds produce the same collapsed prediction pattern; inspect Patient Recall, Healthy Recall and observed counts.
4. Static families are marked STATIC_INPUT / TEMPORAL_INTERPRETATION_LIMITED by preflight; their results are not evidence of genuine temporal dynamics.

## Reproducibility locations

- Run artifacts: logs/audio/<feature>/<model>/<run>/
- Per-epoch summaries: summary.csv
- Metric histories: metrics_history.json
- Checkpoints: <run>/checkpoints/
- MLflow: mlruns/
- Launcher: scripts/run_audio_experiments.sh
