# Legacy WSM implementation boundary

The existing WSM-derived implementation remains in place as **untouched
legacy candidates**. This directory is documentation only; TASK-001B does not
move, delete, import, register, or activate those components.

Legacy candidates include the copied modules under `src/common/`, `src/audio/`,
`src/fusion/`, `src/video/`, and `src/text/`, the historical files under
`configs/`, and retained orchestration under `scripts/`. Their KEEP / ADAPT /
LEGACY / DELETE classifications are recorded in the accepted
[TASK-001A audit](../../docs/TASK-001A_CORPUS_FEATURE_REPO_AUDIT_EN.md).

The active CAREWSM namespace is `src/carewsm/`. Any future reuse or migration
requires a separate reviewed task and an explicit data-contract decision.
