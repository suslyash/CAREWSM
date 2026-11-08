# Historical TEST metrics export for final LaTeX tables

## Outcome

This offline export publishes the original validation-selected TEST metrics for 41 report blocks and 205 fold entries: DEEP-003 (8), DEEP-006 (22), DEEP-008 A/F (2), weighted-F (1), and Stage G (8). It contains 123 aggregate task rows (Depression, PD, Average). No model was retrained and no inference was rerun.

## Method

Each source summary was read at its original validation-selected epoch, selected by the finite dev_mean_macro_recall value with earliest-epoch tie handling. The exporter verifies Training complete, exactly one completed runtime candidate, exactly one non-last checkpoint for that epoch, the checkpoint filename epoch and rounded validation value, and one FINISHED MLflow run ID. TEST values are read only from that selected row.

TEST recalls must be finite and in [0, 1]. Binary Macro Recall is checked against the arithmetic mean of Patient Recall and Healthy Recall within source precision. For every fold, CARE and WSM task values are retained for Macro Recall, Patient Recall, and Healthy Recall. The Average corpus is the arithmetic CARE/WSM mean within that fold. The Average task is the arithmetic Depression/PD mean within that fold; its Average corpus combines all four fold-level corpus/task values. Means and sample standard deviations are then computed over the five fold-level values (ddof=1). No standard deviations are averaged.

## Provenance discrepancy resolution

The accepted Stage G ledger and the first export differed for three fold-zero records. Read-only inspection covered the matching runtime directories, train.log, summary.txt, selected checkpoint names, MLflow SQLite records, run-name tags, experiment ID, status, timestamps, metric counts, and MLflow artifact config snapshots.

- Static-STCH fold 0: ledger ID 276bb158801e4cd1b85aa99949d07071 is a FINISHED but incomplete 12:02 record with zero logged metrics and no completed runtime directory. The complete 12:06 runtime is 03a0d4c7d99741f5a6f4de369943d840, with summary, selected checkpoint, Training complete, and 111 metrics.
- Corrected GradNorm fold 0: ledger ID be8535dd860f4e6da92d04d78a2f65d1 is a FINISHED but incomplete 14:49 record with zero logged metrics. The complete 14:50 runtime is 707670be07684a98ab30807e9d175b88, with summary, selected checkpoint, Training complete, and 120 metrics.
- CAGrad fold 0: ledger ID f0eade557acd4d5a82b2093953026315 is a FINISHED 12:09 record with only 18 metrics. The complete 12:11 runtime is fe6f132700db4677a17939f08eb891fe, with summary, selected checkpoint, Training complete, and 119 metrics.

The regenerated manifest retains the complete runtime identities because they are the only records supported by complete source artifacts and full selected TEST evidence. No TEST value was changed to match a report; the source summaries and checkpoints remain authoritative.

## Provenance and exclusions

Corrected Progress, RA-STCH, and GradNorm use the corrected Stage G namespace. Equal reuses the exact weighted-F fold run directory, selected checkpoint, MLflow run ID, and TEST values. Static-STCH, PCGrad, CAGrad, and DB-MTL use their original valid complete Stage G runs. Superseded pre-correction Progress/RA-STCH/GradNorm artifacts are not included. Ambiguous completed runtime matches fail closed; incomplete candidates are not treated as completed runs. Runtime logs, checkpoints, predictions, patient-level data, and MLflow databases remain uncommitted.

## Verification

The regenerated export passes: 41 blocks, 205 block-fold entries, 123 aggregate rows, five folds per block, 200 unique physical MLflow runs, eight Stage G methods, and 15 corrected replacement runs. Targeted synthetic regression tests cover VAL-only selection, earliest tie handling, fold-first sample SD, component-SD misuse detection, and ambiguous-run failure. Maximum output numerical drift against the prior export is below the documented 5e-11 rounding level; TEST values remain sourced from the selected summaries.

## Reproduction

From the repository root, run:

    .venv/bin/python scripts/export_historical_test_metrics.py

The script writes the three CSVs in docs/evidence/tables/ and fails on missing folds, ambiguous completed runtime candidates, missing checkpoints, checkpoint/VAL mismatches, non-finite or out-of-range recalls, inconsistent binary recalls, duplicate MLflow identities, or non-FINISHED runs.
