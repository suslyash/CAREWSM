# Historical TEST metrics export for final LaTeX tables

## Outcome

This offline export publishes the original validation-selected TEST metrics for 41 report blocks and 205 fold entries: DEEP-003 (8), DEEP-006 (22), DEEP-008 A/F (2), weighted-F (1), and Stage G (8). It contains 123 aggregate task rows (Depression, PD, Average). No model was retrained and no inference was rerun.

## Method

Each source summary was read at its original validation-selected epoch, selected by the finite dev_mean_macro_recall value with earliest-epoch tie handling. The exporter verifies Training complete, exactly one non-last checkpoint for that epoch, and one FINISHED MLflow run ID. TEST values are read only from that selected row.

For every fold, CARE and WSM task values are retained for Macro Recall, Patient Recall, and Healthy Recall. The Average corpus is the arithmetic CARE/WSM mean within that fold. The Average task is the arithmetic Depression/PD mean within that fold; its Average corpus combines all four fold-level corpus/task values. Means and sample standard deviations are then computed over the five fold-level values (ddof=1). No standard deviations are averaged.

## Provenance and exclusions

Corrected Progress, RA-STCH, and GradNorm use the corrected Stage G namespace. Equal uses the completed weighted-F control run, while Static-STCH, PCGrad, CAGrad, and DB-MTL use their original valid Stage G runs. Superseded pre-correction Progress/RA-STCH/GradNorm artifacts are not included. Runtime logs, checkpoints, predictions, patient-level data, and MLflow databases remain uncommitted.

## Reproduction

From the repository root, run:

    .venv/bin/python scripts/export_historical_test_metrics.py

The script writes the three CSVs in docs/evidence/tables/ and fails on missing folds, missing checkpoints, non-finite metrics, incomplete summaries, duplicate MLflow identities, or non-FINISHED runs.
