# NEXT TASK

## TASK-001 — fast audio feature integrity and coverage audit

**Status: READY**

Prompt:

`docs/tasks/TASK-001_FEATURE_AUDIT.md`

Purpose:

- discover the exact operational 11-family feature registry from live data/code;
- verify which artifacts actually extracted successfully;
- classify missing/corrupt/empty/non-finite/schema-invalid artifacts;
- build deterministic per-sample/per-family availability metadata;
- verify the fixed split/fold/label/sample mapping needed for later filtering;
- record temporal/static/timestamp semantics needed for the Mantis 1.0 s / 0.5 s representation task.

Do not train models, derive Mantis embeddings, implement Optuna, or run transfer experiments before manager review.
