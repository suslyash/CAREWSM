# WSM Stage 5 Detailed Plan

Status: **CLOSED TO OPTIMIZATION — R4 trial 012 retained as the leading matched-seed candidate; final promotion pending Stage-6 diagnostic support**.

This is a stage-specific detail file extracted from the former monolithic `docs/PLAN.md`.
Read it only when the active task belongs to this stage or when historical plan detail is explicitly needed.
Current project status is authoritative in [../PROGRESS_EN.md](../PROGRESS_EN.md).

### Stage 5 — RAMPS [CLOSED TO OPTIMIZATION]

- R1: disease teachers calibrated only on the corresponding observed-task DEV data; soft cross-corpus targets, warm-up, separate thresholds, stop-gradient teacher.
- R2: uncertainty, multimodal agreement, OOD distance, coverage curves; semantic agreement only after the base reliability mechanism.
- R3: disease-query fusion, task-conditioned modality gates, availability masks, auxiliary unimodal agreement heads.
- R4: equal weights, static STCH, PAGB/progress comparator, and RA-STCH.

R-full includes only components that improve DEV and do not worsen calibration/negative transfer.

Gate: missing head receives a non-zero direct gradient on the other corpus; observed truth wins; accepted class balance/coverage logged; gradients finite; gains not explained only by parameter count; final composition frozen before Test.


## Post-closure comparator correction

The original negative robustness interpretation was based on comparing R4 seeds42/43/44 against a seed42-only audio reference. The owner-authorized frozen-audio confirmation later established matched audio seeds43/44. Under matched seeds, R4 DEV Mean exceeds audio on 3/3 seeds and in three-seed mean, while audio has lower seed variance. Therefore the negative robustness interpretation is superseded, but Stage-5 optimization remains closed and no confirmation seed may become a tuning target. See `docs/PROGRESS_EN.md` MANAGER-DECISION-059 for the authoritative current interpretation.
