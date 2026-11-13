# Issue #109 — Mean-pooled Static Gated-64 MTL completion

- Source branch base: `510e370d545a9ec2ae96d9bf6a5fb7466edf7e9e` (origin/main at branch creation).
- Final implementation source: `8ae2dc2` (runner verification fix included).
- Exact setup: joint CARE+WSM; four families in order A/B/C/D = emotionmsp_embeds (1024), emotionmsp_trajectories (3), trillsson4_1 (1024), wav2vec_xlsr300m (1024); mean pooling; Static Gated Fusion hidden 64, dropout 0.1; seed 5; AdamW 1e-4; 50 epochs/patience 7; VAL monitor only.
- Production result: 20 unique slots (4 methods × 5 folds), all FINISHED in isolated MLflow. Two superseded FINISHED runs are retained as the required smoke and verification-retry history; they are excluded from the canonical 20-row table.

## Overall UAR (four corpus-task validation Macro Recalls)

| Method | VAL mean ± sample SD | TEST mean ± sample SD | Status |
|---|---:|---:|---|
| Equal (historical comparator) | 0.8309 ± 0.0404 | 0.7822 ± 0.0224 | old 145-run evidence |
| STCH (historical comparator) | 0.8277 ± 0.0407 | 0.7842 ± 0.0174 | old 145-run evidence |
| PCGrad (historical comparator) | 0.8278 ± 0.0394 | 0.7821 ± 0.0225 | old 145-run evidence |
| DB-MTL (historical comparator) | 0.8197 ± 0.0318 | 0.7872 ± 0.0229 | old 145-run evidence |
| progress | 0.8296 ± 0.0411 | 0.7822 ± 0.0241 | new 5-fold evidence |
| ra_stch | 0.8266 ± 0.0392 | 0.7837 ± 0.0147 | new 5-fold evidence |
| gradnorm | 0.8267 ± 0.0432 | 0.7833 ± 0.0209 | new 5-fold evidence |
| cagrad | 0.8251 ± 0.0404 | 0.7751 ± 0.0193 | new 5-fold evidence |

## New method trade-offs

The canonical fold table contains all six metrics for CARE/WSM validation and descriptive TEST for Depression/Parkinson: Macro F1, Weighted F1, Macro Recall, Weighted Recall, Patient Recall and Healthy Recall. The aggregate CSV contains mean and sample SD for every component. Validation ranking is the only comparison basis; TEST is not used for selection, tuning, retry decisions or interpretation of the winner.

| New method | VAL CARE Depression MR | VAL CARE Parkinson MR | VAL WSM Depression MR | VAL WSM Parkinson MR |
|---|---:|---:|---:|---:|
| progress | 0.8461 | 0.8312 | 0.7465 | 0.8945 |
| ra_stch | 0.8524 | 0.8145 | 0.7426 | 0.8970 |
| gradnorm | 0.8555 | 0.8119 | 0.7415 | 0.8978 |
| cagrad | 0.8618 | 0.8041 | 0.7393 | 0.8951 |

## Evidence and integrity

- `logs/audio_turns_combined_mtl_completion/reports/all_results_by_fold.csv`: canonical 20 rows.
- `logs/audio_turns_combined_mtl_completion/reports/aggregate_results.csv`: recomputed five-fold means and sample SD.
- `logs/audio_turns_combined_mtl_completion/reports/provenance_run_ledger.csv`: source/config/checkpoint/MLflow provenance.
- `logs/audio_turns_combined_mtl_completion/reports/run_ledger.csv`: resumable status and failed-attempt history.
- Every selected checkpoint is from validation monitor `val_mean_corpora_tasks/macro_recall`; no TEST value enters selection.
- Heavy checkpoints and both MLflow databases are intentionally not committed.
