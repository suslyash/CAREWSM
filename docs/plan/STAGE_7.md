# WSM Stage 7 Detailed Plan

Status: **COMPLETE — five-seed DEV freeze and one-shot Final Test reporting accepted**.

This is a stage-specific detail file extracted from the former monolithic `docs/PLAN.md`.
Read it only when the active task belongs to this stage or when historical plan detail is explicitly needed.
Current project status is authoritative in [../PROGRESS_EN.md](../PROGRESS_EN.md).

### Stage 7 — Final evaluation

Manager-frozen compared methods before final-seed execution:

1. frozen temporal audio;
2. full R4 trial012;
3. equal-parameter shared-fusion trial012.

Frozen seeds for all compared methods: `42,43,44,45,46`. Existing accepted 42-44 checkpoints are reused. Only seeds45/46 require new training after TASK-007A passes review.

Statistical plan is to be frozen before new runs: five-seed mean, sample standard deviation, and two-sided 95% Student-t confidence interval (df=4), plus paired same-seed deltas/95% CIs for audio vs full R4, audio vs shared fusion, and full R4 vs shared fusion.

All five checkpoints for every method are now DEV-selected and frozen. Final Test reporting is authorized only through TASK-007C.

Pre-Test role freeze:
- primary paper candidate: equal-parameter shared fusion;
- secondary multimodal reference: full R4 trial012;
- frozen baseline: temporal audio.

The primary role is a DEV-only parsimony decision: shared and full have the same 295239 trainable parameters, shared has a slightly higher five-seed DEV Mean, and TASK-006I did not support a task-aware-fusion contribution. The shared-vs-full paired DEV CI includes zero, so no statistical-superiority claim is made.

Once TASK-007C exposes Test values, no model/config/checkpoint/threshold/candidate-role revision is allowed.

After freeze:

1. train final compared methods on at least 5 seeds;
2. choose every checkpoint only by DEV/Mean_Score;
3. run separate TEST_NONE/SOFT/HARD evaluation once;
4. report per-task UAR/MF1/Score and Mean_Score;
5. report mean, standard deviation, confidence interval, and paired comparison where applicable;
6. trace every table entry to config, commit, checkpoint, and MLflow run.

Gate: no Test-driven revision; complete traceability; claims match annotation evidence.

## Final accepted Stage-7 outcome

- DEV checkpoint freeze: `STAGE-7 FIVE-SEED DEV CHECKPOINT FREEZE COMPLETE`.
- Final Test: `STAGE-7 FINAL TEST EVALUATION COMPLETE`.
- Exactly 15 frozen checkpoints: 3 methods × 5 seeds.
- Exactly one standardized Final Test invocation after the pushed evaluator firewall.
- Primary pre-Test role remains equal-parameter shared fusion; full R4 remains secondary reference; temporal audio remains baseline.
- Shared has the highest five-seed Mean point estimate on TEST_NONE, TEST_SOFT, and TEST_HARD, but every predeclared paired Mean 95% CI includes zero. No statistical-superiority claim is authorized.
- No post-Test revision occurred.
- No further Stage-7 experiment or Test invocation is authorized.
