# REFACTOR-000 — Documentation contract evidence

This documentation-only task establishes the simplified CAREWSM development contract before further code refactoring.

## Contract changes

- Research-oriented direct PyTorch/Chimera-ML code is mandatory; needless abstraction is prohibited.
- The active data layer is exactly one Dataset and one DataModule in the two specified files.
- The availability manifest is joined with authoritative split metadata for path fields such as `label_id` and `subgroup`.
- Evaluation names are `train`, `val_care`, `val_wsm`, `test_care`, and `test_wsm`.
- Epoch INFO logging is concise and includes counts, loss, and the three recall outputs; all six metrics remain calculated.
- `CAREWSMSummaryCallback`, wide semicolon-delimited `summary.txt`, `dev_mean_macro_recall`, and the standard `logs/<experiment_name>/<run_name>/` layout are specified.
- Historical 110 runs and their old `joint/average_macro_f1` selection are explicitly preserved as historical evidence.
- REFACTOR-001 through REFACTOR-005 precede TASK-011, which remains paused.

## Preservation

Historical task prompts, reports, and the fold-results CSV were not deleted or rewritten. No Python, YAML, shell script, test, experiment, or data file was changed.

## Review correction

The manager-gate section now treats TASK-001 as historical, requires explicit authorization for each atomic task, places REFACTOR-001 after independent REFACTOR-000 review and merge, and keeps TASK-011 paused through the refactor sequence. The coding-style rule now applies to all active and future implementation tasks, including corrective refactoring. The Mantis rule now refers to future audited schema/timestamp evidence and deferred manager authorization rather than describing TASK-001 as a future gate.
