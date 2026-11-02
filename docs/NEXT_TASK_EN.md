# NEXT TASK

## TASK-001 — fast audio feature integrity and coverage audit

**Status: READY**

Authoritative implementation prompt:

`docs/tasks/TASK-001_FEATURE_AUDIT.md`

Purpose:

- discover the exact 11 feature families from real extraction outputs/code;
- validate whether artifacts actually extracted successfully;
- identify failed/missing/corrupt/empty/non-finite artifacts;
- generate deterministic feature-availability/filtering metadata;
- verify enough split/label/sample identity facts to make later filtering safe;
- do this without training or feature re-extraction.

Do not start the Chimera DataModule or model implementation until the manager reviews TASK-001.
