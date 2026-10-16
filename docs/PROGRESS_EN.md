# WSM Project Progress

## 1. Current State

Plan initialized: 2026-09-23.

Current stage: **Research program — COMPLETE**.

Stage 5: **CLOSED TO OPTIMIZATION — R4 RETAINED AS LEADING MATCHED-SEED CANDIDATE**.

Stage 7: **COMPLETE — FIVE-SEED DEV + ONE-SHOT FINAL TEST EVIDENCE FROZEN**.

Final Test authorized: **completed once under TASK-007C; no further Test invocation authorized**.

Current atomic task: **TASK-012A-C1 COMPLETE WITH DOCUMENTED HISTORICAL GUARD LIMITATIONS — manager review pending**.

Expected Codex branch: `codex/task-012a-shared-progress-postclosure-test`.

`TASK-007C` is accepted and merged. Exact result: `STAGE-7 FINAL TEST EVALUATION COMPLETE`. Stage 7 remains COMPLETE. Shared fusion remains the primary parsimonious paper candidate, full R4 the secondary multimodal reference, and temporal audio the baseline. T1 remains standalone text evidence only; T2 remains blocked before implementation/training. TASK-007D completed the final ledger at [FINAL_RESEARCH_LEDGER_EN.md](FINAL_RESEARCH_LEDGER_EN.md). TASK-010A is an explicit manager-authorized, DEV-only post-closure extension; it does not revise a frozen role, ledger, paper, or Final-Test result.

## TASK-012A post-closure Shared + Progress evaluation

- Outcome: complete; this is a post-closure comparative extension, not a pristine first-use Final Test. Candidate recipe/checkpoints were frozen before Test; no Test-based tuning or follow-up run occurred.
- Branch: `codex/task-012a-shared-progress-postclosure-test`; firewall commit `00e4569d988d6fbd95194d9d4c0e1ff4c20f1a1b` pushed before Test. Final evidence commit is pending manager review.
- Five frozen seeds were verified with exact config/checkpoint SHA256, model `wsm_av_r3_disease_query_model`, shared fusion, Progress balancing, 552775 trainable parameters, and canonical pseudo cache SHA `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945` (accepted D/P `376/1801`).
- DEV reconstruction passed all five Score/Mean gates within `0.0005`; maximum absolute difference was `2.02e-7`. Protocol membership was TEST_NONE `1364`, TEST_SOFT `1208`, TEST_HARD `1014`.
- Exactly one Phase-B invocation completed 15 evaluations (5 seeds × 3 protocols), once each. External artifacts: `logs/task012a_postclosure_shared_progress_test/dev_preflight.json`, `test_results.json`, and `test_invocation.marker`; no raw predictions/logits/probabilities/labels/metadata were retained.
- Evidence dossier: [POSTCLOSURE_SHARED_PROGRESS_TEST_EXTENSION_EN.md](POSTCLOSURE_SHARED_PROGRESS_TEST_EXTENSION_EN.md). Historical Stage-7 evidence remains unchanged; the TASK-011A raw-summary Test-column exposure caveat is retained and strict quarantine compliance is not claimed. `src/audio` stayed unchanged and Test did not drive any decision.
- TASK-012A-C1 audit: retained artifacts hash/schema/arithmetic/identity checks passed without inference or dataset access. Original Test results and negative Shared deltas remain unchanged. The original evaluator’s post-evaluation output-overwrite and weak preflight gates are documented; exact historical autocast dtype and UTC command/push timestamps were not retained. A permanent closed-task Test interlock and four synthetic post-evaluation checks were added; no CLI bypass exists.
- TASK-012A is now locked as a completed comparative extension pending manager review. The original Stage-7 evaluation remains closed; no inference, training, tuning, retry, or further Test pass is authorized.

## TASK-011A post-closure Progress Optuna extension

- Branch: `codex/task-011a-progress-optuna-shared-r4`, based on `eff19d9987e79d99f29b0f5c7fff257341fe350d`.
- Firewall commit pushed before production: `a64068939bff12a198b73cf3d33131536a881d34`.
- Production contract complete: exactly 62 invocations — two untuned seed42 baselines, Shared + Progress `30/30`, and Full R4 + Progress `30/30`. No retry, extra seed, second sweep, or post-firewall source/config change occurred.
- Baseline DEV Mean: Shared `0.810388`; Full R4 `0.808575`; matched delta Shared `+0.001813`. Both baseline trainable counts were `295239`.
- Objective winners: Shared trial 28 Mean `0.8336994328`, D/P Score `0.749256/0.918143`, params `552775`; Full R4 trial 8 Mean `0.8193449680`, D/P Score `0.740769/0.897921`, params `346759`. Balanced winner: `NONE` for both frozen gates.
- Canonical pseudo cache SHA256: `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`; accepted D/P `376/1801`. Evidence dossier: [POSTCLOSURE_PROGRESS_OPTUNA_SHARED_R4_EN.md](POSTCLOSURE_PROGRESS_OPTUNA_SHARED_R4_EN.md).
- Cross-family interpretation is DEV-only equal-budget family optimization; no causal architecture claim, leaderboard, promotion, demotion, confirmation seeds, or Final Test.
- Procedural deviation: one raw summary was opened and rendered Test columns. Test values were not used for any decision or analysis and no Test numbers are reproduced. Strict quarantine compliance is not claimed.
- Status: **TASK-011A COMPLETE WITH DOCUMENTED TEST-QUARANTINE PROCEDURAL DEVIATION**.
- **DEV-ONLY SELECTION EVIDENCE RETAINED; STRICT QUARANTINE COMPLIANCE NOT MET**.

## TASK-010A corrected evidence (scientific review PASS)

- Source gate: `SOURCE GATE RESOLVED`; manager-authorized primary/official sources and pinned revisions are recorded in [analysis/TASK010A_METHOD_SPEC.md](analysis/TASK010A_METHOD_SPEC.md).
- Scope: four actual-parameter gradient methods (PCGrad, CAGrad, GradNorm, DB-MTL) on the historical fixed R4 balancing composition only; no architecture, fusion, pseudo-cache, or `src/audio` change.
- DEV firewall: `wsm_ramps_semantic_dev_only_datamodule` returns validation keys exactly `['dev']`; no Test loader, Test metric, or Final-Test operation was invoked.
- Frozen cache verification: SHA256 `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`; accepted D/P `376/1801`, classes D `376/0`, P `212/1589`.
- Model/loss construction: all four method implementations instantiate against the unchanged 403079 trainable-parameter R4 architecture. TRAIN/DEV-only PCGrad shape/forward/backward smoke passed with no optimizer step; pseudo target and reliability tensors were detached.
- Deterministic CPU tests: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_task010a_gradient_mtl.py` — `13 passed`; includes Equal component/gradient parity, method-specific hand/reference checks, and actual R4 gradient-injection smokes.
- Configs: twelve validated configs under `configs/wsm_mm_pd_dep_v1/mtl_extension/`; [analysis/task010a_config_equivalence.json](analysis/task010a_config_equivalence.json) reports no forbidden difference versus the seed-matched Equal references.
- Production evidence: authoritative FIX1 production is complete after firewall commit 858efb026bbfb8c5a49469218380ea28fe2bbf5a: exactly twelve runs in order PCGrad 42/43/44, CAGrad 42/43/44, GradNorm 42/43/44, and DB-MTL 42/43/44. No retry, sweep, extra seed, or tuning occurred; source/config remained frozen after the firewall. No Test loader, Test metric, or Final Test was used. Final scientific evidence is commit 4828412f62299abdab908596802f77f50d49a30c; manager scientific review is PASS, and FIX2 documentation correction is commit 71577004cdd0e8dc29722cbea31f2fdb27aaaa4b.

### TASK-010A-FIX1 — corrected gradient-MTL comparator evidence complete; scientific review PASS

The previous TASK-010A evidence is explicitly superseded: the old loss inferred
ownership from autograd None/non-None behavior and reported 51 shared / 0
D-only / 0 P-only, which is structurally invalid for R4. The superseded
comparison and CSV are preserved at
[analysis/TASK010A_MTL_COMPARISON_SUPERSEDED_EN.md](analysis/TASK010A_MTL_COMPARISON_SUPERSEDED_EN.md)
and [analysis/task010a_mtl_runs_SUPERSEDED.csv](analysis/task010a_mtl_runs_SUPERSEDED.csv).

FIX1 binds the loss to the named R4 model structure: 14 fully shared tensors,
18 Depression-only tensors, 18 Parkinson-only tensors, and one
row-partitioned task_queries tensor. The callback binds and asserts this
registry before fit. Added deterministic ownership, row-locality, shared
gradient, Equal-parity, and zero-gradient misclassification regression tests;
the FIX1 suite passed `13 passed`. All twelve configs use only `_fix1` run
names; scientific settings remained unchanged.

The corrective firewall commit `858efb026bbfb8c5a49469218380ea28fe2bbf5a`
was pushed before production. The exact twelve authorized FIX1 runs then
completed in order: PCGrad seeds42/43/44, CAGrad seeds42/43/44, GradNorm
seeds42/43/44, and DB-MTL seeds42/43/44. No retry, sweep, extra seed,
metric-driven tuning, or post-firewall source/config change occurred.

Final corrected evidence is recorded in
[analysis/TASK010A_MTL_COMPARISON_EN.md](analysis/TASK010A_MTL_COMPARISON_EN.md)
and [analysis/task010a_mtl_runs.csv](analysis/task010a_mtl_runs.csv). The
selected-checkpoint ledger records DEV-only maximum `dev/mean_score`, selected
epochs, checkpoint SHA256, run directories, runtime, exact structural counts,
and finite method diagnostics. Corrected three-seed DEV Mean values are:
PCGrad `0.7668203333`, CAGrad `0.7631166667`, GradNorm `0.7736443333`, and
DB-MTL `0.7563373333`. The best scalar comparator remains frozen Progress at
`0.7847510000`; these results are descriptive only and do not promote,
demote, or select a method.

Manager scientific review status: PASS. No rerun is required or authorized;
the corrected evidence is complete as descriptive DEV-only post-closure
evidence. No Test loader was iterated and no Test metrics, Final-Test outputs,
or Test artifacts were inspected. src/audio stayed unchanged. Source/config
scopes were frozen after the firewall; main/master were untouched. FIX2 is a
documentation-only correction that closes the stale pending/active status and
the method-family classification issue, with no new scientific result, role
change, promotion, or demotion. This extension does not reopen the closed
research programme or authorize a follow-up task.

## TASK-007D closure

- Outcome: complete; `WSM RESEARCH EVIDENCE PROGRAM COMPLETE`.
- Final Test: complete exactly once; no further Test invocation is authorized.
- Frozen roles: shared fusion primary parsimonious candidate; full R4 secondary reference; temporal audio baseline; T1 standalone text evidence; T2 blocked before implementation/training.
- The Final-Test point estimates do not authorize statistical superiority; every predeclared paired Mean 95% CI includes zero.
- `src/audio` remained unchanged. No compute, Test, recomputation, or new claim was performed.
- Closure boundary: `NO FURTHER EXPERIMENT AUTHORIZED BY THE CURRENT PROGRAMME`.

## 2. Default Context Policy

This file is the compact authoritative status/roadmap entrypoint.

Default agents should read:

1. `AGENTS.md`;
2. `docs/README.md`;
3. `docs/PROJECT_REQUIREMENTS.md`;
4. this file;
5. `docs/NEXT_TASK_EN.md`;
6. only the active stage plan and source/config files explicitly named by the task.

Do **not** load closed-stage plans or the historical progress archive unless the active task or manager explicitly requires them.

Optional historical evidence:

- [archive/PROGRESS_EN_THROUGH_STAGE5_2026-09-28.md](archive/PROGRESS_EN_THROUGH_STAGE5_2026-09-28.md) — complete former progress ledger through Stage-5 closure;
- [plan/README.md](plan/README.md) — stage-plan index;
- [PLAN.md](PLAN.md) — cross-stage formulation, architecture, experiment matrix, promotion rules, risks, and orchestration details.

## 3. High-Level Project Roadmap

| Stage | Status | High-level goal | Detailed plan |
|---|---|---|---|
| 0. Reproducible base | COMPLETE | Freeze and validate the historical audio baseline and instrumentation contract. | [plan/STAGE_0.md](plan/STAGE_0.md) |
| 1. Manifest / partial labels | COMPLETE | Canonical sparse-label manifest, split contract, masking, and evaluation separation. | [plan/STAGE_1.md](plan/STAGE_1.md) |
| 2. Video | COMPLETE | Compare the bounded video families and establish the accepted V2 video reference. | [plan/STAGE_2.md](plan/STAGE_2.md) |
| 3. Text / description | COMPLETE | Two-family budget exhausted: T1 standalone baseline complete; T2 blocked at frozen observable-description preflight. | [plan/STAGE_3.md](plan/STAGE_3.md) |
| 4. Fusion baselines | COMPLETE | Honest A+V baselines and strong-audio fusion search; no safe robust winner. | [plan/STAGE_4.md](plan/STAGE_4.md) |
| 5. RAMPS | CLOSED TO OPTIMIZATION | Optimization/search is complete. Matched-seed audio confirmation superseded the prior negative robustness interpretation; R4 is retained as the leading Stage-6 candidate, not yet a final promoted method. | [plan/STAGE_5.md](plan/STAGE_5.md) |
| 6. Ablations / claims audit | COMPLETE | Core A+V evidence matrix complete; claims frozen in STAGE6_CLAIM_LEDGER_EN.md. | [plan/STAGE_6.md](plan/STAGE_6.md) |
| 7. Final evaluation | COMPLETE | Five-seed DEV freeze and one-shot TEST_NONE/SOFT/HARD reporting completed under frozen roles/checkpoints/statistics. | [plan/STAGE_7.md](plan/STAGE_7.md) |

## 4. Frozen Scientific References

### Matched-seed audio and R4 references

Frozen temporal audio is now confirmed on seeds42/43/44 without any method change:

| Seed | Audio D | Audio P | Audio Mean |
|---:|---:|---:|---:|
| 42 | 0.7479183895 | 0.8277353635 | 0.7878268765 |
| 43 | 0.708699 | 0.811771 | 0.760235 |
| 44 | 0.734396 | 0.809382 | 0.771889 |

Audio three-seed DEV aggregate:

- D mean/std: `0.7303377965 / 0.0199221457`;
- P mean/std: `0.8162961212 / 0.0099784282`;
- Mean mean/std: `0.7733169588 / 0.0138512531`;
- Mean range: `0.0275918765`.

Frozen R4 RA-STCH trial `18c6-012` on the same seeds:

- D mean/std: `0.7352563333 / 0.0211467766`;
- P mean/std: `0.8385606667 / 0.0378688091`;
- Mean mean/std: `0.7869088179 / 0.0294384420`;
- Mean range: `0.0576974538`.

Matched-seed R4 minus audio DEV Mean:

- seed42: `+0.0313155773`;
- seed43: `+0.0012100000`;
- seed44: `+0.0082500000`;
- R4 Mean > audio Mean on `3/3` seeds;
- R4 three-seed Mean advantage: `+0.0135918591`.

Interpretation: audio is **more seed-stable** by Mean std/range, but R4 has the higher matched-seed DEV Mean on every tested seed and higher three-seed mean. Therefore the former phrase “strongest robust audio reference” is retired. This does not by itself establish final promotion or significance.

`src/audio` remains frozen.

### Accepted Stage-2 video reference

- model: V2 prototype-aware video;
- DEV depression Score: `0.6201013364`;
- DEV Parkinson Score: `0.7930427585`;
- DEV Mean_Score: `0.7065720475`.

### Frozen semantic pseudo cache

- path: `/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt`;
- SHA256: `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`;
- TRAIN rows: `6325`;
- accepted missing depression: `376/2665`, all positive;
- accepted missing Parkinson: `1801/3660`, positive/negative `212/1589`;
- observed truth always overrides pseudo supervision;
- no missing-label correctness or comorbidity claim is authorized.

## 5. Stage-5 Closure — Revised Scientific Interpretation

Stage-5 optimization/search remains **closed**. No further tuning against seeds43/44 is authorized.

Historical **MANAGER-DECISION-057** closed Stage 5 negative because R4 three-seed Mean `0.7869088179` was compared against the seed42-only audio value `0.7878268765`. The owner-authorized matched-seed audio audit showed that this was not a valid robustness comparator.

With audio seeds43/44 added under the exact frozen method:

- R4 beats matched audio DEV Mean on `3/3` seeds;
- R4 three-seed Mean is `0.7869088179` vs audio `0.7733169588`;
- R4 mean advantage is `+0.0135918591`;
- audio is more stable by Mean sample std (`0.0138512531` vs R4 `0.0294384420`) and Mean range (`0.0275918765` vs R4 `0.0576974538`);
- task-specific R4-audio regressions occur only on P at seed43 (`-0.007408`) and D at seed44 (`-0.006179`), both smaller than the PLAN's initial 1.0 percentage-point unacceptable-drop recommendation.

Manager interpretation: the **negative robustness verdict is superseded**, but Stage-5 optimization stays closed. R4 RA-STCH trial 012 is retained as the **leading matched-seed multimodal candidate for Stage-6 audit**, not a final promoted method. Promotion still requires mechanism/diagnostic support, and final methods require at least five seeds under PROJECT_REQUIREMENTS.

Candidate B and corrected R3-B remain non-substitutes under their frozen Stage-5 evidence. Seeds43/44 remain confirmation evidence, not tuning targets.

## 6. Stage-6 Closure and Frozen Claim Ledger

Core A+V Stage 6 is **COMPLETE**.

Authoritative synthesis:

- [STAGE6_CLAIM_LEDGER_EN.md](STAGE6_CLAIM_LEDGER_EN.md)
- exact completeness result: `CORE STAGE-6 A+V EVIDENCE MATRIX COMPLETE`;
- supported claims: 4;
- unsupported claims: 7;
- diagnostic-only findings: 2;
- TASK-006J procedural exception remains part of the record: 7 total invocations, 1 pre-DEV failed D42 attempt, 6 accepted completed runs;
- no final method promotion or Stage-7 selection was made.

No further core A+V tuning, Stage-5 reopening, or Stage-6 mechanism search is authorized.

## 7. Active Task — TASK-007D

Purpose: produce the authoritative **final research/program closure ledger** from already accepted Stage-3, Stage-6, and Stage-7 evidence.

No compute is authorized.

Final frozen roles:

1. **primary parsimonious paper candidate:** equal-parameter shared fusion;
2. **secondary multimodal reference:** full R4 trial012;
3. **baseline:** temporal audio;
4. **standalone text evidence:** T1; not additive-fusion evidence;
5. **T2:** blocked before implementation/training.

Final Test has already run exactly once under the frozen evaluator/reporting contract. No further Test invocation is authorized.

Exact completion string on success:

`WSM RESEARCH EVIDENCE PROGRAM COMPLETE`.
## 8. Non-Negotiable Current Boundaries

- Do not tune or modify `src/audio`.
- Do not reopen Stage-5 optimization or Stage-6 A+V tuning.
- Frozen Stage-5/6/3 evidence is now selection history only; no final-method tuning or family changes are authorized.
- Unknown labels remain masked, never converted to negative.
- Observed ground truth overrides pseudo labels.
- Test metrics cannot drive epoch, architecture, hyperparameter, threshold, ablation, or follow-up selection.
- Stage 3 Text/Description is COMPLETE; do not reopen its family/model/prompt search.
- Stage 7 and Final Test are COMPLETE; no further Test invocation, tuning, training, threshold fitting, model revision, or experimental follow-up is authorized by the current programme.
- Do not claim verified recovery of genuinely missing disease labels or comorbidity without dual-annotated evidence.

## 9. Recent Authoritative Manager Decisions

Only recent state-changing decisions are repeated here; older decisions are in the archive.

### MANAGER-DECISION-056 — Triple Optuna accepted; R4 trial 012 selected for confirmation

- Candidate B objective winner: Mean `0.8165907903`, but 0/20 relaxed-safe.
- corrected R3-B objective winner: Mean `0.8159243728`, but 0/20 relaxed-safe.
- corrected R4 RA-STCH objective winner `18c6-012`: D/P/Mean `0.759025/0.879259/0.8191424538`;
- R4 had 4/20 relaxed-safe trials; trial 012 was the only objective winner with exact audio D/P non-regression;
- this was DEV-only family selection, not final promotion.

### MANAGER-DECISION-057 — R4 confirmation failure accepted; Stage 5 closed negative; Stage 6 activated

- true seed43 and seed44 confirmations failed the frozen robust gate;
- three-seed Mean `0.7869088179` remained below frozen audio Mean `0.7878268765`;
- Stage 5 closed negative;
- Stage 6 activated;
- no tuning against confirmation seeds is allowed.

### MANAGER ASSIGNMENT — TASK-006A-BUNDLE

- shuffled/mismatched pseudo-target negative control on matched Equal was assigned, then paused before execution for the owner-requested audio matched-seed check;
- after TASK-006-PRE-AUDIO-CONFIRM, it is RESUMED unchanged as the sole active Stage-6 task.

### MANAGER-OVERRIDE-058 — Confirm frozen audio on seeds43/44 before Stage-6 negative controls

- owner requested a matched-seed check because the prior "robust audio reference" interpretation relied on a seed42 audio result while R4 had true seeds42/43/44;
- authorize exactly two frozen-audio production runs: seeds43 and44;
- no audio method or hyperparameter tuning is allowed;
- only seed and run_name may differ from the canonical frozen seed42 config;
- compare audio and frozen R4 on matched seeds42/43/44 using DEV metrics only;
- TASK-006A-BUNDLE is paused until manager review of this evidence.

### MANAGER-DECISION-059 — Accept matched-seed audio confirmation; supersede negative robustness interpretation; resume TASK-006A

- TASK-006-PRE-AUDIO-CONFIRM passed manager review and was merged via PR #49;
- firewall commit `71a5d54375928387392e6dadfda182842515748b` correctly precedes final evidence commit `daec15685fb618161bf68f6b32e5df4b566b1875`;
- configs43/44 are semantically identical to the frozen seed42 config except seed/run_name; no `src` changes occurred;
- exact two production runs occurred, seed43 then seed44, with DEV-only checkpoint selection and no tuning/rerun;
- the Codex handoff statement that audio is “less stable than R4” is rejected: audio has lower Mean std/range;
- however, R4 exceeds matched audio DEV Mean on 3/3 seeds and by `+0.0135918591` in three-seed mean;
- therefore the prior Stage-5 negative robustness interpretation against seed42-only audio is superseded;
- Stage-5 optimization remains closed; R4 trial 012 is retained as the leading matched-seed candidate pending Stage-6 claim/mechanism audit;
- TASK-006A-BUNDLE is resumed as the only next atomic task.

### MANAGER-DECISION-060 — Reject TASK-006A implementation; exact derangement FIX1 required

- branch `codex/task-006a` starts from the correct manager base and changes only the authorized five paths;
- firewall commit `70e5eec9d3710ac7bd0bd05cae6848f32107d190` correctly precedes final commit `c34bb6a20a42c59c7d3aa0c0413ab1c70ba7f7dc`;
- config equivalence to matched Equal passes and no `src` file changed;
- the reported shuffled results show a large matched-over-shuffled DEV Mean gap, but the result is **not accepted** because the derived cache did not follow the frozen permutation mapping literally;
- rejected builder used `source_rows = roll(random_order, 1)` but assigned `derived[field][missing] = source[field][source_rows]`; the frozen contract requires destinations to be `random_order`, i.e. `derived[field][random_order] = source[field][roll(random_order, 1)]`;
- rejected builder also included an unauthorized conditional fixed-point repair; exact circular shift requires no repair;
- calibration reporting used `shuffled - matched` per seed and omitted the required three-seed mean `matched - shuffled` calibration deltas;
- the existing `codex/task-006a` must not be merged until FIX1 passes; its cache SHA `678571febcf6ce85f2bf7aedb4770b255a138a0cdaff5ea3f04a92606afbd814` and first three runs are superseded diagnostic evidence only;
- assign exactly one corrective pass, `TASK-006A-FIX1`, on the **same** `codex/task-006a` branch;
- FIX1 is a reproducibility correction, not tuning. It authorizes exactly three new corrective shuffled Equal runs on seeds42/43/44.

### MANAGER-OVERRIDE-061 — Reuse codex/task-006a for FIX1

- owner explicitly prefers one branch per logical task; no new `codex/task-006a-fix1` branch;
- reuse of existing `codex/task-006a` is explicitly authorized despite the normal no-reuse rule;
- Codex must fetch and merge current `origin/main` into `codex/task-006a` before corrective implementation so MANAGER-DECISION-060/OVERRIDE-061 are in branch history;
- this one merge from `origin/main` into the task branch is explicitly manager-authorized;
- do not reset, rebase, overwrite, or force-push the branch;
- preserve the original rejected commits/runs as superseded evidence, then append a new corrective firewall commit and final evidence commit;
- exactly three new corrective production runs remain authorized.

### MANAGER-DECISION-062 — Reject FIX1 production for wrong cache binding; assign FIX2 on same branch

- FIX1 branch history correctly merged manager main before corrective work;
- corrected builder implementation at firewall commit `096b720c875b427fc6d17d1dd04b65a67614b802` implements the frozen roll-1 mapping literally and removes the unauthorized repair;
- FIX1 produced and audited the exact cache SHA `e4baf2a2eea9dc7703e73b7956971b0d571eedda55fc8e6984750270712cf0f9`;
- however, configs31/32/33 in both `096b720` and final `d5fbbc5b9ac0fb7136996caf4090fd5d309f5fe7` point to the old rejected cache `shuffled_missing_targets_seed6001.pt`, not `shuffled_missing_targets_seed6001_exact_fix1.pt`;
- the reported FIX1 DEV metrics and checkpoint SHA256 values exactly equal the earlier rejected runs, so FIX1 production evidence is superseded and cannot close the claim;
- do not rebuild or overwrite the verified exact cache; verify it read-only by path/SHA/metadata/invariants before production;
- assign exactly one narrow correction: update the three configs to the exact cache, use distinct FIX2 run names, firewall, then run exactly seeds42/43/44 again;
- reuse `codex/task-006a`; no new Codex branch; no tuning.

### MANAGER-DECISION-063 — Accept TASK-006A-FIX2; close negative-control item 8; assign corpus probe

- TASK-006A-FIX2 passed manager review and was merged via PR #54;
- accepted exact negative-control cache SHA256: `e4baf2a2eea9dc7703e73b7956971b0d571eedda55fc8e6984750270712cf0f9`;
- FIX2 firewall `24fd49ba614514b462422b46baf8ae1c5d0b3c98` preceded final evidence `07c149d4661a4a8afd11f53fcc557e57b5239126`;
- exactly three accepted FIX2 runs used the exact cache and seeds42/43/44;
- shuffled three-seed DEV Mean `0.7371180000` vs matched Equal `0.7791726667`;
- matched-minus-shuffled Mean delta `+0.0420546667`; matched exceeds shuffled on `3/3` seeds;
- therefore `SAMPLE-SPECIFIC PSEUDO ALIGNMENT CLAIM SUPPORTED BY THIS NEGATIVE CONTROL`;
- interpretation remains narrow: no missing-label correctness, comorbidity, significance, or final-promotion claim;
- calibration is mixed by task/metric and is diagnostic only;
- Stage-6 item 8 is CLOSED;
- next sole atomic task is TASK-006B, the frozen R4 corpus-identity probe required by Stage-6 item 9.

### MANAGER-DECISION-064 — Accept TASK-006B; close corpus probe; assign isolated RA-STCH audit

- TASK-006B passed manager review and was merged via PR #56;
- firewall commit `3ba0fcce68550eccf61a34d4e20e55a4111212f4` preceded final evidence `ec258e16373f38adabc03213bde8b4d65f14d30b`;
- probe report SHA256: `3361de90e0cb26f14e15f639bb54ef4f7d29d54232b72fb39f154d7f7105a4dc`;
- no main-model retraining, Test-loader iteration, source change, config change, or probe tuning occurred;
- three-seed true corpus AUROC means: audio `0.791292`, video `0.705505`, projected A+V `0.750650`, task-fused `0.759832`, gates `0.551605`, logits `0.575197`;
- task-fused minus projected-A+V AUROC deltas were `+0.024588/-0.000165/+0.003123`, mean `+0.009182`;
- frozen interpretations are `STRONG CORPUS-ID DECODABILITY FLAG NOT TRIGGERED` and `FUSION AMPLIFICATION OF CORPUS DECODABILITY NOT SUPPORTED`;
- corpus identity remains structurally coupled to observed-task identity; no causal shortcut or disease-signal replacement claim is authorized;
- no separate domain-mitigation experiment is triggered by this probe;
- Stage-6 corpus-probe item 9 is CLOSED;
- next sole atomic task is TASK-006C: complete the original fixed-composition RA-STCH ablation on true seeds43/44 and compare three-seed evidence against Equal/Static/Progress.

### MANAGER-DECISION-065 — Accept TASK-006C; close RA-STCH contribution claim negative; assign direct pseudo ablation

- TASK-006C passed manager review and was merged via PR #58;
- firewall commit `45b17ebb6d54acb602ba9698677a37b47292e38d` preceded final evidence `c7d201fe9ff463187827dc3d4a60a7884349c032`;
- exactly two new frozen RA-STCH runs occurred, seed43 then44; retained seed42 was not rerun;
- three-seed fixed-composition RA D/P/Mean = `0.6931503333/0.8519640000/0.7725573333`;
- RA minus Equal Mean by seed = `+0.004725/+0.002163/-0.026734`; aggregate Mean delta `-0.0066153333`;
- RA minus best simpler balancing Mean by seed = `-0.004654/-0.010530/-0.024654`;
- therefore `RA-STCH BALANCING CONTRIBUTION OVER EQUAL NOT SUPPORTED`;
- and `RA-STCH ADVANTAGE OVER SIMPLER BALANCING NOT SUPPORTED`;
- controller diagnostics remained finite/bounded but do not rescue the failed DEV gates;
- this negative mechanism result does not automatically demote the separately optimized trial-012 composition, whose hyperparameters differ and which remains the leading matched-seed candidate pending remaining Stage-6 component audits;
- Stage-6 balancing/RA-STCH claim is CLOSED as negative;
- next sole atomic task is TASK-006D: remove only the direct pseudo-supervision loss term from optimized trial-012 on seeds42/43/44 while retaining pseudo acceptance/reliability/controller information.

### MANAGER-DECISION-066 — Accept TASK-006D; direct pseudo contribution supported; assign reliability ablation

- TASK-006D passed manager review and was merged via PR #60;
- firewall commit `5a5a657c56cdc2eee35c7e85c990022083ab70a1` preceded final evidence `7239d38333ec0019f94644bd10bc5a63aff48e5f`;
- configs40/41/42 differ from full trial-012 only by run_name and pseudo warm-up `final_scale=0.0`;
- exactly three runs occurred, seeds42/43/44;
- no-direct-pseudo three-seed D/P/Mean = `0.7134376667/0.8263193333/0.7698780000`;
- full trial-012 minus no-direct-pseudo aggregate D/P/Mean = `+0.0218186667/+0.0122413333/+0.0170308179`;
- full DEV Mean exceeded the ablation on 2/3 seeds and in aggregate, with no aggregate D/P unacceptable regression under the frozen gate;
- therefore `DIRECT PSEUDO-SUPERVISION CONTRIBUTION SUPPORTED`;
- interpretation is limited to the direct pseudo BCE term; pseudo-label correctness, missing-label recovery, comorbidity recovery, significance, and final promotion remain unauthorized;
- Stage-6 direct pseudo-supervision claim is CLOSED as supported;
- next sole atomic task is TASK-006E: remove graded reliability while preserving pseudo acceptance and targets, then compare three seeds against full trial-012.

### MANAGER-DECISION-067 — Accept TASK-006E; reliability contribution supported; assign semantic-evidence ablation

- TASK-006E passed manager review and was merged via PR #62;
- firewall commit `699115362466518b7a9a02ef6f2aec89c5b70a62` preceded final evidence `cda75f32d1915d9e71cd029a5ba45feee501731b`;
- derived uniform-reliability cache SHA256: `713b5a3d963c8759e4b2c12f148be12818e72c135b6d1420207cf5aa150bcc40`;
- accepted rows/targets/classes/calibrated probabilities were unchanged and accepted reliability was exactly 1.0;
- uniform-reliability three-seed D/P/Mean = `0.7352583333/0.8345853333/0.7849220000`;
- full-minus-ablation aggregate D/P/Mean = `-0.0000020000/+0.0039753333/+0.0019868179`;
- full Mean exceeded the ablation on 2/3 seeds and in aggregate, so `UNCERTAINTY/RELIABILITY CONTRIBUTION SUPPORTED`;
- the effect is small and the claim is limited to the combined implemented graded-reliability signal; BCE weighting and controller reliability are not separately identified;
- Stage-6 uncertainty/reliability item is CLOSED as supported;
- next sole atomic task is TASK-006F: remove the semantic-enabled depression pseudo-acceptance path while preserving the frozen Parkinson pseudo path and all full trial-012 training semantics.

### MANAGER-DECISION-068 — Accept TASK-006F; semantic contribution not supported; assign no-video modality removal

- TASK-006F passed manager review and was merged via PR #64;
- firewall commit `1c8b56977f9731fcefd4bf8df98149faa1915c28` preceded final evidence `bbd23dd9b9c64b4e738f2bca1b21840e9a61465a`;
- derived cache SHA256: `aae9d250c5030eab1e68e07ee8af43fe8ebb7286eca621f9fa5c6dd7af283fe6`;
- exactly 376 depression pseudo rows were removed and all 1801 Parkinson pseudo rows were preserved exactly;
- ablation three-seed D/P/Mean = `0.714447/0.868369/0.791408`;
- full-minus-ablation D/P/Mean = `+0.020809/-0.029808/-0.004499`;
- full D exceeded the ablation on 3/3 seeds, but the ablation had higher aggregate Mean and much higher P, so `SEMANTIC-EVIDENCE CONTRIBUTION VIA DEPRESSION PSEUDO ACCEPTANCE NOT SUPPORTED`;
- no semantic correctness, missing-label recovery, comorbidity, significance, or promotion claim is authorized;
- the final TASK-006F ledger typo “P 376” is manager-corrected to “P accepted 1801”; all runtime/cache evidence already used 1801;
- Stage-6 semantic-evidence item is CLOSED as negative;
- next sole atomic task is TASK-006G: train the same optimized trial-012 composition with video declared unavailable on every sample, using the existing availability-aware model path and preserving parameter count.

### MANAGER-DECISION-069 — Accept TASK-006G; video contribution not supported; assign no-audio removal

- TASK-006G passed manager review and was merged via PR #66;
- firewall commit `e6ac651601a75eb7e839b35230f32b6df41ba7ab` preceded final evidence `bafeee1fe1f5b0fa4457d50b24665778154b03fd`;
- the availability override is backward-compatible and leaves full default behavior unchanged;
- no-video uses fixed architecture/parameter count `295239`, full pseudo cache, and exact availability `[true,false]`;
- video perturbation had no effect on logits/loss/non-video gradients; video weights/features/availability gradients were neutralized as required;
- no-video three-seed D/P/Mean = `0.749645/0.783216/0.766431`;
- full-minus-no-video D/P/Mean = `-0.014389/+0.055344/+0.020478`;
- full Mean exceeded no-video on 2/3 seeds and in aggregate, but full aggregate D was `0.014389` below no-video, violating the frozen `0.010000` non-regression bound;
- therefore `VIDEO MODALITY CONTRIBUTION NOT SUPPORTED`;
- this is a negative claim result, not evidence that video is useless in general or that a video-free model should be promoted;
- the video-removal half of Stage-6 modality item 7 is CLOSED;
- next sole atomic task is TASK-006H: no-audio online-input removal using the same availability mechanism with `[false,true]`, fixed architecture/parameter count, and unchanged frozen pseudo-training path.

### MANAGER-DECISION-070 — Accept TASK-006H; modality removals complete; assign task-aware fusion ablation

- TASK-006H passed manager review and was merged via PR #69;
- firewall commit `2d008e367aa43468759505cc4e7ab6b424a61d02` preceded final evidence `b189173109fde6acfbda92dc4af379b0ba38fea1`;
- exactly three no-audio runs occurred, seeds42/43/44, with no source or pseudo-cache changes;
- availability was exactly `[false,true]`, audio features/gradients were neutralized, and audio perturbation did not affect logits/loss/non-audio gradients;
- no-audio three-seed D/P/Mean = `0.653375/0.792030/0.722703`;
- full-minus-no-audio D/P/Mean = `+0.081881/+0.046531/+0.064206`;
- full DEV Mean exceeded no-audio on 3/3 seeds and all frozen claim criteria passed;
- therefore `ONLINE AUDIO INPUT MODALITY CONTRIBUTION SUPPORTED`;
- interpretation remains limited to the online student audio branch because the frozen pseudo cache retains historical audio-derived teacher information;
- together with TASK-006G, Stage-6 modality-removal item 7 is CLOSED;
- next sole atomic task is TASK-006I: an equal-parameter shared-fusion ablation that removes task-conditioned fusion specialization while preserving sparse MTL, separate disease heads, pseudo/reliability/controller paths, and the full trial-012 training protocol.

### MANAGER-DECISION-071 — Accept TASK-006I; task-aware fusion not supported; assign sparse-MTL audit

- TASK-006I passed manager review and was merged via PR #71;
- firewall commit `cf4f9c481976f4e358242324ce4ff6e6d526e61d` preceded final evidence `5795c0ddadbf2d8f65dfaa112ca3da21e5759151`;
- full/default model behavior was bitwise-compatible, and full/shared-fusion variants both retained exactly `295239` trainable parameters;
- exactly three shared-fusion runs occurred, seeds42/43/44;
- shared-fusion three-seed D/P/Mean = `0.7407873333/0.8467100000/0.7935663333`;
- full-minus-shared D/P/Mean = `-0.0055310000/-0.0081493333/-0.0066575154`;
- full Mean exceeded shared-fusion on `0/3` seeds and the shared aggregate Mean was higher;
- therefore `TASK-AWARE FUSION CONTRIBUTION NOT SUPPORTED`;
- sparse MTL and separate disease heads remained active in TASK-006I, so no sparse-MTL conclusion follows from that negative result;
- Stage-6 task-aware fusion item is CLOSED as negative;
- next sole atomic task is TASK-006J: compare the full joint sparse-MTL model against independently DEV-selected depression-only and Parkinson-only copies on seeds42/43/44.

### MANAGER-DECISION-072 — Accept TASK-006J with procedural exception; assign Stage-6 synthesis

- TASK-006J passed scientific manager review and was merged via PR #73;
- initial firewall `b9e45c71586057fcbcd0bf5add97b373cea89f99` was followed by one failed D42 invocation that reached TRAIN epoch 1 and stopped on an empty depression-active batch before any DEV/Test evaluation or checkpoint selection;
- corrective commit `63c327699f30ebd572de35cfb6a7d144734156ba` changed only the authorized loss file plus PROGRESS, added zero-connected neutral handling for empty active-task batches, re-established clean-origin default compatibility, and was pushed before the accepted completed sequence;
- therefore the original “exactly six invocations” criterion was **formally violated**: total training invocations were 7;
- manager accepts a documented procedural exception because the failed invocation produced no DEV/Test evidence, no selected checkpoint, and no metric-driven feedback; the accepted scientific evidence comes only from the corrected six completed runs;
- corrected completed run order was exactly D42, P42, D43, P43, D44, P44;
- task-isolated D/P/paired-Mean three-seed means = `0.717846/0.881495/0.799671`;
- full-minus-isolated aggregate D/P/Mean = `+0.017410/-0.042935/-0.012762`;
- therefore `SPARSE MTL JOINT-TRAINING CONTRIBUTION NOT SUPPORTED`;
- TASK-006J closes the remaining sparse-MTL item;
- no further Stage-6 A+V experiments are authorized before synthesis;
- next sole atomic task is TASK-006K, a documentation-only Stage-6 evidence synthesis / claim-freeze dossier;
- Stage 6 remains ACTIVE pending manager review of that dossier; Stage 7 and Final Test remain locked;
- deferred Stage 3 Text/Description remains required before paper-ready/final evaluation freeze.

### MANAGER-DECISION-073 — Accept TASK-006K; close Stage 6; activate deferred Stage 3

- TASK-006K passed manager review and was merged via PR #75;
- its only changes were `docs/STAGE6_CLAIM_LEDGER_EN.md` and PROGRESS;
- exact synthesis result: `CORE STAGE-6 A+V EVIDENCE MATRIX COMPLETE`;
- all Stage-6 plan items 1–10 and required diagnostics map to accepted evidence;
- the frozen claim ledger retains 4 supported claims, 7 unsupported claims, and 2 diagnostic-only findings with their exact interpretation boundaries;
- TASK-006J remains explicitly recorded as 7 total invocations / 6 accepted completed runs; the failed first D42 attempt occurred before DEV/Test evaluation and was not metric-driven;
- no final method was promoted or selected;
- core A+V Stage 6 is CLOSED and no further Stage-6 A+V tuning is authorized;
- PROJECT_REQUIREMENTS still requires the deferred Text/Description work before paper-ready freeze;
- Stage 3 Text/Description is therefore ACTIVATED;
- Stage 7 and Final Test remain LOCKED;
- next sole atomic task is TASK-003A: transcript source/alignment/language audit before any T1 encoder implementation or training.

### MANAGER-DECISION-074 — Accept corrected TASK-003A; freeze T1 implementation contract

- corrected TASK-003A passed manager review and was merged via PR #77;
- original firewall `35a50c8`, first field-name runtime fix `f54649f`, manager corrective firewall `b9a2d85`, and final evidence `2e1609fa4c641c8f09153444d4da984f1d78ab53` remain traceable;
- rejected report SHA `2dc2064aca131da0e80c103df1b8dd7a19d9735aa8c183e149392f7aafd1206b` is superseded;
- authoritative corrected report SHA is `4254d81281522015d0633d6534e1616e925a2bf0130f67a37b541a2aacc2ed78`;
- TEST records contain no language-analysis outputs; TEST lexical inspection/language audit/prediction-metric inspection are all false;
- exact accepted conclusions are `SEGMENT TEXT ALIGNMENT NOT ESTABLISHED`, `T1 TEXT UNIT SHOULD BE VIDEO-LEVEL TRANSCRIPT`, and `T1 TRANSCRIPT DATA CONTRACT READY`;
- Stage 3 therefore advances to T1 implementation;
- to avoid an encoder sweep, manager freezes one pretrained multilingual encoder family before DEV performance evidence: `FacebookAI/xlm-roberta-base` revision `e73636d`;
- T1 uses frozen video-level transcript features plus one lightweight two-head chunk-sequence model; no segment-level transcript is invented;
- TASK-003B is implementation/cache-only. Production T1 training is reserved for a later manager-authorized task after pipeline review.

### MANAGER-DECISION-075 — Accept TASK-003B; authorize frozen T1 three-seed evidence

- TASK-003B passed manager review and was merged via PR #79;
- firewall `40ee994`, tokenizer runtime correction `305a3e4`, and final evidence `b10bb12ba6103d54db75879c312c3c74d7f5700b` remain traceable;
- one T1 family only was implemented; no encoder search or production performance evidence occurred;
- frozen encoder identity is `FacebookAI/xlm-roberta-base` revision `e73636d`, resolved commit `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, model SHA `6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb`;
- frozen cache index SHA256 is `4f276b60e8e423a1eab2d69a02782daa274fb183ae028a525f16df3adcf83ea8`;
- cache has 755 entries / 754 available artifacts / one unavailable TRAIN depression transcript;
- T1 TRAIN has exactly 553 unique video-level units (D287/P266); evaluation broadcast remains segment-level only;
- frozen T1 downstream model has `447746` trainable parameters;
- smoke forward/loss/backward and masked-unknown invariance passed; no DEV/Test performance metric was inspected;
- next sole atomic task is TASK-003C: run the already frozen seeds42/43/44 configs exactly once each and record DEV-selected T1 evidence.

### MANAGER-DECISION-076 — Accept TASK-003C; T1 complete; activate T2 preflight

- TASK-003C passed manager review and was merged via PR #81;
- firewall `6af94e4` preceded exactly three frozen production runs, seeds42/43/44, with no retry, tuning, source/config/cache change, or Test-driven decision;
- selected DEV D/P/Mean by seed: seed42 `0.721153/0.899381/0.810267`; seed43 `0.562911/0.825768/0.694339`; seed44 `0.596355/0.809025/0.702690`;
- T1 three-seed D/P/Mean = `0.6268063333/0.8447246667/0.7357653333`;
- T1 D/P/Mean sample std = `0.0834002123/0.0480683689/0.0646553057`;
- T1 Mean is `-0.0375516255` vs matched temporal audio and `-0.0511434846` vs full R4, while T1 P is `+0.0284285455` vs audio and `+0.00616399997` vs R4;
- secondary unique-video DEV metrics were lower than segment-weighted metrics on every seed and were diagnostic only;
- T1 is accepted only as a standalone text baseline; no additive, promotion, or significance claim is made;
- Stage-3 plan explicitly permits one second and final family T2 = T1 + observable description semantic stream;
- to avoid a generator sweep, manager freezes `Qwen/Qwen3-VL-8B-Instruct` revision `1dd1e02d981403da25ed73d43430e4ef598cb94b` for a no-training preflight;
- TASK-003D must establish source granularity, deterministic prompt/generation, environment feasibility, and leakage/hallucination audit before any full T2 cache/model work.

### MANAGER-DECISION-077 — Accept TASK-003D blocked T2 preflight; assign Stage-3 synthesis

- TASK-003D passed manager review and was merged via PR #83;
- exact result: `T2 DESCRIPTION GENERATION CONTRACT BLOCKED`;
- deterministic TRAIN/DEV audit generated 24/24 nonempty and visually grounded/mostly observable outputs;
- frozen gate failed because demographic/identity inference occurred in `4/24` outputs and only `18/24` met the concision requirement;
- diagnosis/health inference, causal/medication inference, and dataset/task/label leakage were all `0/24`;
- TEST generation/manual inspection remained false;
- external report SHA256 is `767d687e2f857938fa882555250d7a4be4159556e49537377967d82c36407930`;
- default video processing OOM was handled transparently through pushed corrective firewalls by freezing uniform deterministic `num_frames=8`; model and prompt remained unchanged;
- no full description cache, T2 model, training, performance metric, prompt variant, or generator switch is authorized;
- the Stage-3 two-family budget is now exhausted: T1 completed, T2 blocked;
- next sole atomic task is TASK-003E, a documentation-only Stage-3 synthesis/closure dossier and pre-Stage-7 candidate inventory.

### MANAGER-DECISION-078 — Accept TASK-003E; close Stage 3; activate Stage 7 with frozen compared-method set

- TASK-003E passed manager review and was merged via PR #85;
- exact Stage-3 synthesis results are `STAGE-3 TEXT/DESCRIPTION EVIDENCE MATRIX COMPLETE` and `STAGE-3 TWO-FAMILY BUDGET EXHAUSTED`;
- Stage 3 is CLOSED: T1 is a completed standalone three-seed text baseline; T2 is blocked before implementation/training; no third text/description family is authorized;
- Stage 7 is ACTIVATED, but Final Test remains LOCKED;
- manager freezes the Stage-7 compared-method set using DEV-only evidence, before seeds45/46 and before any final-Test use:
  1. frozen temporal audio baseline;
  2. full R4 trial012;
  3. equal-parameter shared-fusion trial012 candidate;
- rationale: audio is the mandatory frozen/stable anchor; full R4 is the optimized multimodal reference; shared fusion is the matched equal-parameter single-model comparator that exceeded full Mean on all existing 3/3 seeds while TASK-006I did not support a task-aware-fusion contribution;
- T1 is not selected as a final method because its three-seed Mean `0.7357653333` is below matched audio/R4 and its Mean std `0.0646553057` is high; it remains required standalone text evidence;
- T2 is ineligible because its generation contract is blocked;
- no-semantic-depression is not selected because its D regression versus full is `0.020809`, despite higher P/aggregate Mean, creating a material task trade-off;
- paired task-isolated is not selected because it is a two-model, non-equal-deployment-size diagnostic control;
- frozen final seeds are `42,43,44,45,46`; accepted existing seeds42-44 are reused, not rerun;
- TASK-007A is freeze/preflight only: create seed45/46 config clones, verify existing seed42-44 artifacts, freeze the statistical plan and Test firewall; no production training or Test inspection.

### MANAGER-DECISION-079 — Accept TASK-007A; authorize six missing final-seed runs

- TASK-007A passed manager review and was merged via PR #87;
- diff was exactly six new seed45/46 configs plus `docs/STAGE7_FINAL_FREEZE_EN.md` and PROGRESS;
- every new config is machine-identical to its required reference after removing only `seed` and `run_name`;
- frozen finalists remain exactly temporal audio, full R4 trial012, and equal-parameter shared fusion;
- frozen seeds remain exactly `42,43,44,45,46`; accepted seeds42-44 will not be rerun;
- audio trainable parameter count is `3033416`; full R4/shared are each `295239`;
- existing seed42-44 checkpoint identities are traceable; shared-fusion accepted seed42/43/44 selected epochs are `11/10/11` with checkpoint SHA256 `21504702976a960ffea02a68778ebc3bf7e6cc15ea8c9866f182ff04cbd30785`, `ec746e519e551d69d59954dffff0a73968fb4ab555d43020b7722e635cc477ee`, and `215e3a61c5dcb8ff7dd92ed5ca9336d66405c3f1ba6cc045841f08ec990d10a8`;
- five-seed mean/std/range/95% Student-t CI and paired same-seed delta plans are frozen with multiplier `2.7764451051977987`;
- DEV-only Brier/ECE-15 calibration reporting is frozen;
- Final Test remains locked; no Test metric was inspected in TASK-007A;
- TASK-007B is authorized to run exactly six missing jobs in order audio45, R4-45, shared45, audio46, R4-46, shared46, with no retry, extra seed, config change, or metric-driven stopping;
- after the six jobs, TASK-007B must freeze all fifteen selected checkpoints and compute final five-seed DEV evidence only;
- a separate manager-reviewed TASK-007C will be required before any final TEST_NONE/SOFT/HARD reporting.

### MANAGER-DECISION-080 — Accept TASK-007B; freeze primary candidate; authorize separate Final Test reporting

- TASK-007B passed manager review and was merged via PR #89;
- exact completion is `STAGE-7 FIVE-SEED DEV CHECKPOINT FREEZE COMPLETE`;
- all 15 checkpoint paths/SHA256 identities are frozen and verified for audio/full-R4/shared-fusion across seeds42-46;
- the audio42 historical checkpoint was recovered and the accepted TASK-004H DEV result was reproduced exactly through raw-logit/legacy-argmax equivalent paths; the earlier `0.4367` diagnostic is superseded as an evaluation-path failure;
- five-seed DEV Mean (95% CI): audio `0.765145 [0.746230,0.784059]`; full R4 `0.790057 [0.763530,0.816583]`; shared fusion `0.790914 [0.766258,0.815571]`;
- paired Mean full-R4 minus audio CI is `[0.001169,0.048655]`; shared minus audio `[0.009556,0.041984]`; shared minus full R4 `[-0.009704,0.011419]`;
- full R4 and shared each beat audio Mean on `5/5` frozen DEV seeds; shared vs full is effectively tied under the frozen paired CI;
- manager preselects **equal-parameter shared fusion as the primary parsimonious paper candidate** before Test because it has the slightly higher five-seed DEV Mean, the same `295239` trainable parameter count as full R4, and TASK-006I found `TASK-AWARE FUSION CONTRIBUTION NOT SUPPORTED`;
- this primary-role choice is a DEV-only parsimony decision, not a statistical-superiority claim;
- full R4 remains the secondary multimodal reference; temporal audio remains the frozen baseline;
- TASK-007C is authorized as the one separate Final Test reporting task;
- after Test exposure, no model/config/checkpoint/threshold/candidate-role revision is authorized.

### MANAGER-DECISION-081 — Accept TASK-007C; close Stage 7; assign final documentation closure

- TASK-007C passed manager review and was merged via PR #91;
- exact result: `STAGE-7 FINAL TEST EVALUATION COMPLETE`;
- corrective firewall `bfaae7f812904dd9ae9a7460b394117867db5439` was pushed before any Test iteration;
- after that firewall, evaluator code did not change; the final commit changed evidence documents only;
- exactly one standardized Final Test invocation occurred;
- evaluator SHA256 is `b691c9563bb6ffd63316b664949dcf1bef02d8d3f4e770110c2e70adbc4cee1e`;
- external final report SHA256 is `59f763f04843c8e54d5d3f9893d3abe0530b47a32ea11f5b9585390edeb515bc`;
- protocol membership is TEST_NONE `1364`, TEST_SOFT `1208`, TEST_HARD `1014`;
- shared has the highest five-seed Mean point estimate on all three Test protocols, but every predeclared paired Mean 95% CI includes zero; no statistical-superiority claim is authorized;
- shared retains the pre-Test primary role; full R4 remains secondary; temporal audio remains baseline;
- no post-Test model/config/checkpoint/threshold/role revision occurred;
- Stage 7 is COMPLETE and no further experiment/Test invocation is authorized;
- next sole task TASK-007D is documentation-only final research/program closure.
## 10. Historical Evidence

The full former 5,000+ line ledger is intentionally no longer part of default context.

Use it only when a historical implementation detail, exact old command, old PR/commit, or prior corrective review is material:

[archive/PROGRESS_EN_THROUGH_STAGE5_2026-09-28.md](archive/PROGRESS_EN_THROUGH_STAGE5_2026-09-28.md)


### TASK-006-PRE-AUDIO-CONFIRM - pre-run firewall complete

Status: partial; the frozen-audio seed43/44 firewall passed and was committed before production.

Created configs:

- `configs/wsm_mm_pd_dep_v1/audio/01_frozen_baseline_seed43.yaml`
- `configs/wsm_mm_pd_dep_v1/audio/02_frozen_baseline_seed44.yaml`

Firewall evidence:

- Both configs validate with `chimera-ml validate-config`.
- Config 01 differs from config00 only by `seed: 43` and `run_name: frozen_audio_wavlm_l9_pool4_seed43`.
- Config 02 differs from config00 only by `seed: 44` and `run_name: frozen_audio_wavlm_l9_pool4_seed44`.
- The registered audio DataModule, model, and loss instantiated successfully for both configs.
- TRAIN-only forward/loss/backward smoke passed for both configs with finite loss and nonzero finite trainable gradients: config01 batch shape `(8, 1245, 768)`, loss `0.3348849714`; config02 batch shape `(8, 1250, 768)`, loss `0.3805317283`; each reported 113 nonzero-gradient tensors.
- No optimizer step was executed. No DEV or Test loader was iterated.
- Frozen audio checkpoint SHA256 remains `0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2`.
- `git diff --check` passed and `git diff origin/main -- src` is empty.
- No source file or existing config was changed.

No production training has started. The firewall commit is pushed before the exactly two authorized invocations, in order: seed43 once, then seed44 once. TASK-006A-BUNDLE remains paused/unstarted; Stage 5 remains historically closed negative.

### TASK-006-PRE-AUDIO-CONFIRM - final frozen-audio seed confirmation evidence

Status: complete. Exactly two new production invocations occurred, in order: seed43 once, then seed44 once. No seed42 rerun, sweep, tuning, restart, or shuffled-pseudo work occurred. Stage 5 remains historically closed negative; TASK-006A-BUNDLE remains paused/unstarted.

Run and MLflow identities:

- Seed43 run: `frozen_audio_wavlm_l9_pool4_seed43_2026-09-28_13-12_audio_mamba_segment_model_08192c1b`; MLflow run ID `4119e9a654ec481ca42606f7f207b969`; status `FINISHED`; artifact URI `/media/maxim/Programs/Projects/WSM/mlruns/5/4119e9a654ec481ca42606f7f207b969/artifacts`; resolved config `configs/wsm_mm_pd_dep_v1/audio/01_frozen_baseline_seed43.yaml`; firewall commit `71a5d54`.
- Seed44 run: `frozen_audio_wavlm_l9_pool4_seed44_2026-09-28_13-21_audio_mamba_segment_model_8720f109`; MLflow run ID `ab9a3a73a0e94a2c889a69b25038d141`; status `FINISHED`; artifact URI `/media/maxim/Programs/Projects/WSM/mlruns/5/ab9a3a73a0e94a2c889a69b25038d141/artifacts`; resolved config `configs/wsm_mm_pd_dep_v1/audio/02_frozen_baseline_seed44.yaml`; firewall commit `71a5d54`.
- Both runs stopped by the frozen DEV-only early-stopping monitor after 11 and 10 epochs respectively; selected checkpoints were frozen by maximum `dev/mean_score` before reading same-epoch Test monitoring.

Audio DEV table:

| Seed | Selected epoch | DEV D Score | DEV P Score | DEV Mean | Selected checkpoint SHA256 |
|---:|---:|---:|---:|---:|---|
| 42 | 4 | 0.7479183895 | 0.8277353635 | 0.7878268765 | `0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2` |
| 43 | 5 | 0.708699 | 0.811771 | 0.760235 | `1a65f8dd605a6d942bed828f126e79a1f02930dd578176ba31f49c4dc10899dd` |
| 44 | 4 | 0.734396 | 0.809382 | 0.771889 | `7832d20a5431666365e2214423633f2be87bb27f37406d5a5dda8adda21e9852` |

Complete new-seed DEV metrics:

- Seed43 epoch 5: D UAR/MF1/Score = 0.709057/0.708341/0.708699; P UAR/MF1/Score = 0.811042/0.812500/0.811771; Mean = 0.760235.
- Seed44 epoch 4: D UAR/MF1/Score = 0.735201/0.733592/0.734396; P UAR/MF1/Score = 0.797447/0.821317/0.809382; Mean = 0.771889.

Same-epoch Test monitoring only (not used for selection or interpretation):

- Seed43 epoch 5: TEST_NONE Mean = 0.791690; TEST_SOFT Mean = 0.802602; TEST_HARD Mean = 0.818189.
- Seed44 epoch 4: TEST_NONE Mean = 0.811919; TEST_SOFT Mean = 0.820354; TEST_HARD Mean = 0.831768.

Audio three-seed DEV aggregate:

- D Score mean/std = 0.7303377965 / 0.0199221457; min/max/range = 0.708699 / 0.7479183895 / 0.0392193895.
- P Score mean/std = 0.8162961212 / 0.0099784282; min/max/range = 0.809382 / 0.8277353635 / 0.0183533635.
- Mean mean/std = 0.7733169588 / 0.0138512531; min/max/range = 0.760235 / 0.7878268765 / 0.0275918765.

Matched R4 minus audio DEV deltas:

| Seed | D delta | P delta | Mean delta | R4 Mean > audio Mean |
|---:|---:|---:|---:|---|
| 42 | +0.0111066105 | +0.0515236365 | +0.0313155773 | yes |
| 43 | +0.0098280000 | -0.0074080000 | +0.0012100000 | yes |
| 44 | -0.0061790000 | +0.0226780000 | +0.0082500000 | yes |

R4-vs-audio three-seed variability comparison:

| Metric | R4 mean/std | Audio mean/std | R4 min/max/range | Audio min/max/range |
|---|---:|---:|---|---|
| D | 0.7352563333 / 0.0211467766 | 0.7303377965 / 0.0199221457 | 0.718527 / 0.759025 / 0.0404980 | 0.708699 / 0.7479183895 / 0.0392194 |
| P | 0.8385606667 / 0.0378688091 | 0.8162961212 / 0.0099784282 | 0.804363 / 0.879259 / 0.0748960 | 0.809382 / 0.8277353635 / 0.0183534 |
| Mean | 0.7869088179 / 0.0294384420 | 0.7733169588 / 0.0138512531 | 0.761445 / 0.8191424538 / 0.0576975 | 0.760235 / 0.7878268765 / 0.0275919 |

Interpretation boundary: R4 exceeds matched audio DEV Mean on 3/3 seeds, but this is a robustness comparator audit only. No significance test, promotion, demotion, Stage-6 ablation choice, or final-method claim is made. Test remains monitoring-only.

Final integrity: production invocation count exactly 2; `src/audio` and all `src/*` unchanged; no Test-driven decision; no post-hoc tuning/rerun; no Stage 6 shuffled-pseudo work, Stage 7, Text/Description, or Final Test work started. Stage 6 is active at the plan level but this task does not start it.


### TASK-006A-BUNDLE   pre-training negative-control firewall

Status: firewall complete; production has not started. The exact three shuffled-control configs are frozen before the first production invocation.

Changed paths before firewall commit:

- `scripts/common/build_ramps_shuffled_pseudo_negative_control.py`;
- `configs/wsm_mm_pd_dep_v1/ablations/31_shuffled_pseudo_equal_seed42.yaml`;
- `configs/wsm_mm_pd_dep_v1/ablations/32_shuffled_pseudo_equal_seed43.yaml`;
- `configs/wsm_mm_pd_dep_v1/ablations/33_shuffled_pseudo_equal_seed44.yaml`;
- `docs/PROGRESS_EN.md`.

Cache build command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/common/build_ramps_shuffled_pseudo_negative_control.py --source /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt --output /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001.pt --shuffle-seed 6001
```

Builder result: `RAMPS_SHUFFLED_NEGATIVE_CONTROL_PASS`. Source SHA256 is `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`; derived cache SHA256 is `678571febcf6ce85f2bf7aedb4770b255a138a0cdaff5ea3f04a92606afbd814`. Rows are `6325`; missing D/P are `2665/3660`; accepted D/P are `376/1801`; accepted class counts are D positive/negative `376/0` and P positive/negative `212/1589`. Acceptance-assignment Hamming differences are D/P `646/1774`. Within-task permutation fixed points are D/P `0/0`; permutation SHA256 values are D `1a3853bb4ad61c627d320be9f1c90fa7f6493a3ea922613f08be458c0e963f6b` and P `82df2450281b6cb1bbc5ecdfcc9c733b707dd53553c5ecdf0979032b836d6af1`. Source segment IDs, observed masks, observed targets, and the complete missing-row tuple multisets are unchanged; source cache was not modified.

Config validation result: all three configs are valid under `chimera-ml validate-config`. Each is a matched Equal copy differing only in pseudo-cache path and run name; seeds remain `42/43/44`, and the model, loss, optimizer, warm-up, callbacks, logging, and training parameters are unchanged.

TRAIN-only registry/data/model/loss smoke passed with the derived cache and no optimizer step. It selected one TRAIN batch containing accepted shuffled pseudo rows and ran forward/loss/backward at pseudo_scale=1.0. Result: `FROZEN_DATAMODULE_MODEL_LOSS_FIREWALL_PASS`; TRAIN rows `6325`; accepted pseudo rows in smoke batch `17`; trainable parameters `403079`; finite loss `0.7944685221`; pseudo-target and reliability gradients are `None`; model gradients are finite. Plugin registration imported without a project-module warning. No DEV/Test loader was iterated and no optimizer step was executed.

Required frozen hashes checked: audio checkpoint `0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2`; source pseudo cache `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`. `git diff --check` passed and `git diff -- src/audio` is empty.

The firewall commit must precede exactly these production commands, in order: config31 seed42, config32 seed43, config33 seed44. No sweep, tuning, Test-driven selection, or other Stage-6 work is authorized.


### TASK-006A-BUNDLE   shuffled pseudo negative-control production evidence

Status: complete. Exactly three production runs completed, in the authorized order: shuffled Equal seed42, seed43, seed44. No sweep, extra seed, optimizer-only run, post-hoc tuning, or source/config change occurred.

Selected only by maximum DEV `dev/mean_score`:

| Seed | Run | Selected epoch | DEV D Score | DEV P Score | DEV Mean | Selected checkpoint SHA256 |
|---:|---|---:|---:|---:|---:|---|
| 42 | `stage6_shuffled_pseudo_equal_seed42_2026-09-28_13-49_wsm_av_r3_disease_query_model_dea1d1da` | 1 | 0.707618 | 0.746877 | 0.727247 | `b24a8be454db99ba735fadf3b040b07a154bfbb78caa52d3ce0664df717c2302` |
| 43 | `stage6_shuffled_pseudo_equal_seed43_2026-09-28_13-52_wsm_av_r3_disease_query_model_a77a85b2` | 3 | 0.689827 | 0.776322 | 0.733074 | `fb0b863509ff10fa19adfc5e07d89e8d8350aa49ea8f58be219a65734da3b8fa` |
| 44 | `stage6_shuffled_pseudo_equal_seed44_2026-09-28_13-55_wsm_av_r3_disease_query_model_7c7c3c9f` | 3 | 0.674850 | 0.807394 | 0.741122 | `95c523d35dc747cdcd1f102661866b9cb4e846d09fb61bf5d5efa12bf9ca2392` |

The three shuffled DEV Mean values have mean `0.7338143333`, sample standard deviation `0.0069670637`, and range `0.727247 0.741122`. Matched Equal DEV Mean values are `0.777920/0.779671/0.779927`, mean `0.7791726667`, sample standard deviation `0.0010923664`, and range `0.777920 0.779927`. Shuffled minus matched paired deltas are seed42 `-0.0506730`, seed43 `-0.0465970`, and seed44 `-0.0388050`; three-seed mean delta is `-0.0453583333`. Matched Equal exceeds shuffled on `3/3` seeds and in the three-seed mean.

Same-epoch Test protocol outputs were recorded only as post-freeze monitoring, never for selection or interpretation. Shuffled TEST_NONE/SOFT/HARD Mean values were seed42 `0.820707/0.837643/0.836956`, seed43 `0.745789/0.745173/0.726297`, and seed44 `0.780918/0.783803/0.780038`.

DEV calibration audit used identical DEV rows and observed masks, with binary Brier score and 15-bin confidence ECE. Observed counts were D/P `621/312` for every seed and both methods.

| Seed | Method | D Brier | D ECE-15 | P Brier | P ECE-15 |
|---:|---|---:|---:|---:|---:|
| 42 | matched Equal | 0.250991 | 0.216796 | 0.090156 | 0.041386 |
| 42 | shuffled | 0.183690 | 0.078068 | 0.133782 | 0.054640 |
| 43 | matched Equal | 0.250396 | 0.221254 | 0.093445 | 0.048379 |
| 43 | shuffled | 0.258899 | 0.218367 | 0.128615 | 0.111257 |
| 44 | matched Equal | 0.254806 | 0.228324 | 0.088165 | 0.038778 |
| 44 | shuffled | 0.262067 | 0.219220 | 0.120834 | 0.082013 |

Calibration deltas (shuffled minus matched) were D Brier `-0.067301/+0.008503/+0.007262` and D ECE `-0.138728/-0.002886/-0.009104` for seeds42/43/44; P Brier `+0.043627/+0.035171/+0.032668` and P ECE `+0.013254/+0.062877/+0.043235`. This is a DEV calibration audit only.

Interpretation: `SAMPLE-SPECIFIC PSEUDO ALIGNMENT CLAIM SUPPORTED BY THIS NEGATIVE CONTROL`, because matched Equal DEV Mean exceeds shuffled on all 3/3 seeds and the matched three-seed mean exceeds shuffled by `0.0453583333`. This supports the sample-specific alignment claim for this frozen negative control; it does not establish missing-label correctness, comorbidity recovery, final-method promotion, or statistical significance.

Production commands were exactly:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/ablations/31_shuffled_pseudo_equal_seed42.yaml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/ablations/32_shuffled_pseudo_equal_seed43.yaml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/ablations/33_shuffled_pseudo_equal_seed44.yaml
```

No Test protocol was used for selection, comparison, calibration, or interpretation. No Stage-5 optimization was reopened; no Stage-7, Text/Description, or Final Test work started. Stage 6 remains active for the remaining manager-assigned claims audit.


### TASK-006A-FIX1   corrective pre-training firewall

Status: firewall complete; corrective production has not started. The existing `codex/task-006a` branch was reused under explicit manager authorization. Current `origin/main` `3c413acd7b0f9c7c909f944d611650224bab0c38` was merged first in merge commit `9e6b6aa`. The original TASK-006A implementation and three runs remain in history as NOT MERGED / superseded diagnostic evidence and are excluded from the FIX1 claim decision.

The rejected builder/config working state was corrected before this firewall commit. The exact mapping is: for each task, `order = missing[torch.randperm(...)]`, `source_order = torch.roll(order, shifts=1, dims=0)`, then each permuted field is assigned exactly as `derived[field][order, task] = source[field][source_order, task].clone()`. D uses generator seed `6001`; P uses `6002`. No fixed-point repair, swap, sorting, rejection sampling, remapping, or post-hoc permutation change exists.

Fresh cache command:

```bash
PATH=/media/maxim/Programs/Projects/WSM/.venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/common/build_ramps_shuffled_pseudo_negative_control.py --source /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt --output /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001_exact_fix1.pt --shuffle-seed 6001
```

Result: `RAMPS_SHUFFLED_NEGATIVE_CONTROL_PASS`. Source SHA before generation and after generation: `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`. Fresh FIX1 cache SHA256: `e4baf2a2eea9dc7703e73b7956971b0d571eedda55fc8e6984750270712cf0f9`. Rows `6325`; missing D/P `2665/3660`; accepted D/P `376/1801`; class counts D `376/0`, P `212/1589`; acceptance Hamming D/P `646/1788`. Destination/source sets each equal the complete missing-row set; source equals exact roll-1 destination order; fixed points D/P `0/0`. Permutation SHA256 over ordered int64 `[destination_order, source_order]` pairs: D `1dcfde36463f21d2d2d525d70f6642c6b599cc23534f3cb47ffc41953ec4d4b8`; P `847f8e2cf156cd0d4ecb28f5269d5cc79fd5943c6605f44c57c78528ecc6d767`. Segment IDs, observed masks/targets, tuple multisets, neutral rejected fields, accepted target/probability equality, and observed-truth precedence all passed. The rejected external cache was untouched.

All three FIX1 configs passed `chimera-ml validate-config`. Programmatic equivalence passed for each matched Equal reference: only `data.params.pseudo_cache_path` and `run_name` differ; seeds, architecture, Equal loss, optimizer, warm-up, callbacks, logging, and all other semantics are unchanged. The FIX1 cache loads through the existing semantic DataModule.

TRAIN-only firewall smoke was run for each config at `pseudo_scale=1.0`, with no DEV/Test loader iteration and no optimizer step:

- seed42: `FIX1_CONFIG_FIREWALL_SMOKE_PASS`, trainable params `403079`, finite loss `0.7784184813`;
- seed43: `FIX1_CONFIG_FIREWALL_SMOKE_PASS`, trainable params `403079`, finite loss `0.7689405670`;
- seed44: `FIX1_CONFIG_FIREWALL_SMOKE_PASS`, trainable params `403079`, finite loss `0.8094737520`.

For each smoke, observed supervision and accepted missing-head pseudo supervision produced non-zero finite model gradients; pseudo target/reliability gradients were `None`; main heads, audio/video projections, task queries, and gate gradients were finite and non-zero. `git diff --check` passed and `git diff origin/main -- src` was empty.

No corrective production run may start before this firewall is committed and pushed. The only authorized subsequent commands are the three FIX1 production commands in order: seeds42, 43, 44.


### TASK-006A-FIX1   corrective production and claim evidence

Status: complete. Exactly three NEW corrective production runs completed in order: FIX1 seed42, seed43, seed44. The prior rejected runs remain superseded diagnostic evidence and are not reused, averaged, selected among, or counted.

Run identities:

- seed42: `stage6_shuffled_pseudo_equal_seed42_exact_fix1_2026-09-28_14-48_wsm_av_r3_disease_query_model_71480159`; MLflow ID `75fbd0da0dc9439389ca8026a8902405`; status `FINISHED`; artifact URI `/media/maxim/Programs/Projects/WSM/mlruns/5/75fbd0da0dc9439389ca8026a8902405/artifacts`;
- seed43: `stage6_shuffled_pseudo_equal_seed43_exact_fix1_2026-09-28_14-51_wsm_av_r3_disease_query_model_79faa7ba`; MLflow ID `9a72fa9c8ec5469b821ea0b74b44a379`; status `FINISHED`; artifact URI `/media/maxim/Programs/Projects/WSM/mlruns/5/9a72fa9c8ec5469b821ea0b74b44a379/artifacts`;
- seed44: `stage6_shuffled_pseudo_equal_seed44_exact_fix1_2026-09-28_14-54_wsm_av_r3_disease_query_model_b4590a31`; MLflow ID `6e760048f620431da3c2dc4f430b2a17`; status `FINISHED`; artifact URI `/media/maxim/Programs/Projects/WSM/mlruns/5/6e760048f620431da3c2dc4f430b2a17/artifacts`.

DEV-only selected checkpoints:

| Seed | Epoch | D UAR/MF1/Score | P UAR/MF1/Score | Mean | Checkpoint SHA256 |
|---:|---:|---|---|---:|---|
| 42 | 1 | 0.707796/0.707439/0.707618 | 0.737681/0.756072/0.746877 | 0.727247 | `b24a8be454db99ba735fadf3b040b07a154bfbb78caa52d3ce0664df717c2302` |
| 43 | 3 | 0.690056/0.689597/0.689827 | 0.766322/0.786323/0.776322 | 0.733074 | `fb0b863509ff10fa19adfc5e07d89e8d8350aa49ea8f58be219a65734da3b8fa` |
| 44 | 3 | 0.675023/0.674677/0.674850 | 0.801794/0.812994/0.807394 | 0.741122 | `95c523d35dc747cdcd1f102661866b9cb4e846d09fb61bf5d5efa12bf9ca2392` |

FIX1 shuffled DEV D/P/Mean: seed42 `0.707618/0.746877/0.727247`; seed43 `0.689827/0.776322/0.733074`; seed44 `0.674850/0.807394/0.741122`. Three-seed means are D `0.690765`, P `0.7768643333`, Mean `0.7338143333`; sample standard deviations are D `0.0164041257`, P `0.0302621449`, Mean `0.0069670637`.

Matched Equal minus FIX1 shuffled deltas are:

| Seed | D delta | P delta | Mean delta |
|---:|---:|---:|---:|
| 42 | +0.004880 | +0.096465 | +0.050673 |
| 43 | +0.013472 | +0.079721 | +0.046597 |
| 44 | +0.033127 | +0.044484 | +0.038805 |

Aggregate matched-minus-FIX1 deltas are D `+0.0171600000`, P `+0.0735566667`, and Mean `+0.0453583333`. Matched DEV Mean exceeds FIX1 on `3/3` seeds and in the three-seed mean.

Corrected DEV calibration audit used identical DEV rows/masks; observed counts were D/P `621/312` for every seed and method. Values and deltas are matched minus FIX1:

| Seed | D matched Brier | D FIX1 Brier | D delta | D matched ECE-15 | D FIX1 ECE-15 | D delta | P matched Brier | P FIX1 Brier | P delta | P matched ECE-15 | P FIX1 ECE-15 | P delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.250991 | 0.183690 | +0.067301 | 0.216796 | 0.078068 | +0.138728 | 0.090156 | 0.133782 | -0.043627 | 0.041386 | 0.054640 | -0.013254 |
| 43 | 0.250396 | 0.258899 | -0.008503 | 0.221254 | 0.218367 | +0.002886 | 0.093445 | 0.128615 | -0.035171 | 0.048379 | 0.111257 | -0.062877 |
| 44 | 0.254806 | 0.262067 | -0.007262 | 0.228324 | 0.219220 | +0.009103 | 0.088165 | 0.120834 | -0.032668 | 0.038778 | 0.082013 | -0.043235 |

Three-seed mean matched-minus-FIX1 calibration deltas: D Brier `+0.0171788832`; D ECE-15 `+0.0502391444`; P Brier `-0.0371552979`; P ECE-15 `-0.0397886039`. This audit is diagnostic only; no recalibration or rerun occurred.

Same-epoch Test monitoring, read only after checkpoint freeze and never used for selection or interpretation: seed42 TEST_NONE/SOFT/HARD Mean `0.820707/0.837643/0.836956`; seed43 `0.745789/0.745173/0.726297`; seed44 `0.780918/0.783803/0.780038`.

Frozen claim result: `SAMPLE-SPECIFIC PSEUDO ALIGNMENT CLAIM SUPPORTED BY THIS NEGATIVE CONTROL`. This is narrow sample-specific pseudo-alignment evidence only. No claim is made about missing-label correctness, comorbidity recovery, significance, or final promotion. No tuning, sweep, retry for metric improvement, post-hoc config change, source change, or additional run occurred.


### TASK-006A-FIX2 — cache-binding firewall

Status: firewall complete; FIX2 production has not started. FIX1 production is rejected/superseded because configs31/32/33 were bound to the old rejected cache path `/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001.pt`. The accepted exact cache is `/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001_exact_fix1.pt`.

The existing `codex/task-006a` branch was reused after merging current manager `origin/main` `4722fca3464ca60f8840b0f67efc2ff927d24c41`. The corrected FIX1 builder and exact cache were not modified or regenerated.

Read-only exact-cache verification passed:

- cache SHA256: `e4baf2a2eea9dc7703e73b7956971b0d571eedda55fc8e6984750270712cf0f9`;
- source SHA256 remains `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`;
- mapping: `destination_order = order`, `source_order = roll(order, 1)`, and `derived[field][order, task] = source[field][source_order, task]`;
- permutation SHA256: D `1dcfde36463f21d2d2d525d70f6642c6b599cc23534f3cb47ffc41953ec4d4b8`, P `847f8e2cf156cd0d4ecb28f5269d5cc79fd5943c6605f44c57c78528ecc6d767`;
- fixed points D/P `0/0`; acceptance Hamming D/P `646/1788`;
- missing D/P `2665/3660`; accepted D/P `376/1801`; class counts D `376/0`, P `212/1589`;
- semantic DataModule loaded the exact cache and resolved its exact path/SHA.

Corrected configs:

- `31_shuffled_pseudo_equal_seed42.yaml` -> run name `stage6_shuffled_pseudo_equal_seed42_exact_fix2`;
- `32_shuffled_pseudo_equal_seed43.yaml` -> run name `stage6_shuffled_pseudo_equal_seed43_exact_fix2`;
- `33_shuffled_pseudo_equal_seed44.yaml` -> run name `stage6_shuffled_pseudo_equal_seed44_exact_fix2`.

All three passed config validation and programmatic equivalence against matched Equal configs13/19/20: only the exact cache path and run name differ.

FIX2 firewall smoke passed for all three configs with exact DataModule resolution, R3 trainable parameters `403079`, finite loss, nonzero observed/pseudo model gradients, finite main-head/projection/query/gate gradients, and no pseudo-target/reliability gradients:

- seed42 loss `0.7388503551`;
- seed43 loss `0.7636069059`;
- seed44 loss `0.7564906478`.

No optimizer step and no DEV/Test loader iteration occurred. `git diff --check` passed and `git diff origin/main -- src` was empty.

The firewall commit must precede exactly three FIX2 production commands in order: seeds42, 43, 44. No sweep, tuning, retry, or post-hoc config change is authorized.


### TASK-006A-FIX2 — corrected cache binding production evidence

Status: complete. FIX1 production is rejected because configs31/32/33 used the old wrong cache path `/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001.pt`. The accepted exact cache path is `/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001_exact_fix1.pt`, SHA256 `e4baf2a2eea9dc7703e73b7956971b0d571eedda55fc8e6984750270712cf0f9`. The exact cache was verified read-only; it was not regenerated or overwritten.

The new FIX2 firewall commit is `24fd49b`. Exactly three NEW FIX2 production runs completed in order: seed42, seed43, seed44. No prior shuffled run was reused, averaged, selected among, or counted.

Run identities:

- seed42: `stage6_shuffled_pseudo_equal_seed42_exact_fix2_2026-09-28_15-17_wsm_av_r3_disease_query_model_fdd3d5a9`; MLflow ID `1783b4c6424842b6a6cb88aa38c4c164`; status `FINISHED`; artifact URI `/media/maxim/Programs/Projects/WSM/mlruns/5/1783b4c6424842b6a6cb88aa38c4c164/artifacts`;
- seed43: `stage6_shuffled_pseudo_equal_seed43_exact_fix2_2026-09-28_15-21_wsm_av_r3_disease_query_model_21343cc1`; MLflow ID `23d9b071710347d7983d79b7ea43351f`; status `FINISHED`; artifact URI `/media/maxim/Programs/Projects/WSM/mlruns/5/23d9b071710347d7983d79b7ea43351f/artifacts`;
- seed44: `stage6_shuffled_pseudo_equal_seed44_exact_fix2_2026-09-28_15-25_wsm_av_r3_disease_query_model_6c38e6e8`; MLflow ID `8941f5f4bb0d485cb07bbd00e56b1980`; status `FINISHED`; artifact URI `/media/maxim/Programs/Projects/WSM/mlruns/5/8941f5f4bb0d485cb07bbd00e56b1980/artifacts`.

DEV-only selected results:

| Seed | Epoch | D UAR/MF1/Score | P UAR/MF1/Score | Mean | Checkpoint SHA256 |
|---:|---:|---|---|---:|---|
| 42 | 5 | 0.675490/0.674841/0.675166 | 0.785645/0.812657/0.799151 | 0.737158 | `baa2336955ea192acc1304a26c5f3aad3affadc6df3e13aa278228a5a91832a7` |
| 43 | 3 | 0.690056/0.689597/0.689827 | 0.766322/0.786323/0.776322 | 0.733074 | `fb0b863509ff10fa19adfc5e07d89e8d8350aa49ea8f58be219a65734da3b8fa` |
| 44 | 3 | 0.675023/0.674677/0.674850 | 0.801794/0.812994/0.807394 | 0.741122 | `95c523d35dc747cdcd1f102661866b9cb4e846d09fb61bf5d5efa12bf9ca2392` |

FIX2 shuffled D/P/Mean values are seed42 `0.675166/0.799151/0.737158`, seed43 `0.689827/0.776322/0.733074`, and seed44 `0.674850/0.807394/0.741122`. Three-seed means are D `0.6799476667`, P `0.7942890000`, Mean `0.7371180000`; sample standard deviations are D `0.0085572124`, P `0.0160964772`, Mean `0.0040241491`.

Matched Equal minus FIX2 deltas:

| Seed | D delta | P delta | Mean delta |
|---:|---:|---:|---:|
| 42 | +0.037332 | +0.044191 | +0.040762 |
| 43 | +0.013472 | +0.079721 | +0.046597 |
| 44 | +0.033127 | +0.044484 | +0.038805 |

Aggregate matched-minus-FIX2 deltas are D `+0.0279770000`, P `+0.0561320000`, and Mean `+0.0420546667`. Matched DEV Mean exceeds FIX2 on `3/3` seeds and in the three-seed mean.

DEV calibration audit used identical rows/masks, with observed counts D/P `621/312` for every seed. Values are matched, FIX2, then matched-minus-FIX2:

| Seed | D Brier | D ECE-15 | P Brier | P ECE-15 |
|---:|---|---|---|---|
| 42 | 0.250991 / 0.249330 / +0.001661 | 0.216796 / 0.189452 / +0.027344 | 0.090156 / 0.122209 / -0.032053 | 0.041386 / 0.095453 / -0.054067 |
| 43 | 0.250396 / 0.258899 / -0.008503 | 0.221254 / 0.218367 / +0.002886 | 0.093445 / 0.128615 / -0.035171 | 0.048379 / 0.111257 / -0.062877 |
| 44 | 0.254806 / 0.262067 / -0.007262 | 0.228324 / 0.219220 / +0.009103 | 0.088165 / 0.120834 / -0.032668 | 0.038778 / 0.082013 / -0.043235 |

Three-seed mean matched-minus-FIX2 calibration deltas: D Brier `-0.0047011574`; D ECE-15 `+0.0131110949`; P Brier `-0.0332974344`; P ECE-15 `-0.0533929869`. No recalibration or rerun occurred.

Same-epoch Test monitoring was read only after checkpoint freeze and did not affect selection or interpretation: seed42 TEST_NONE/SOFT/HARD Mean `0.777046/0.771482/0.778540`; seed43 `0.745789/0.745173/0.726297`; seed44 `0.780918/0.783803/0.780038`.

Frozen claim result: `SAMPLE-SPECIFIC PSEUDO ALIGNMENT CLAIM SUPPORTED BY THIS NEGATIVE CONTROL`. This is narrow sample-specific pseudo-alignment evidence only. No missing-label correctness, comorbidity recovery, significance, or final-promotion claim is made. No tuning, sweep, retry, post-hoc config change, or source change occurred.


### TASK-006B — corpus-probe firewall

Status: firewall complete; full probe not yet executed. Branch: `codex/task-006b`, based on manager `origin/main` `6487f1e`.

Implemented only [`scripts/common/run_r4_corpus_probe.py`](../scripts/common/run_r4_corpus_probe.py). The firewall command was:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/common/run_r4_corpus_probe.py --firewall

Firewall passed for all three frozen configurations/checkpoints: checkpoint files exist and SHA256 matches seed42 `104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a`, seed43 `6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e`, and seed44 `b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17`. Correct model state loaded at epochs 11/10/11 and models were in eval mode. Tiny two-row canonical TRAIN+DEV extraction passed for every seed; all six representations were finite with exact dimensions audio/video/projected A+V/task fused/gates/logits = `160/160/320/320/4/2`. Corpus labels were read from `sample_meta[*]["corpus"]` only. No Test loader was iterated.

Local AUROC synthetic checks passed for perfect `1.0`, reversed `0.0`, and tied constant `0.5`; the deterministic zero-initialized float64 LBFGS probe sanity check passed. Frozen probe settings are encoded in the script: TRAIN-only population standardization, class-balanced weighted BCE, weight-only L2 `1e-4`, and LBFGS `lr=1.0`, `max_iter=250`, `tolerance_grad=1e-10`, `tolerance_change=1e-12`, `history_size=50`, `strong_wolfe`. `git diff --check` passed and `git diff origin/main -- src` was empty. No main-model training occurred.

Mandatory firewall commit and push must precede the single full-probe command.


### TASK-006B — frozen R4 corpus probe

Status: complete; Stage-6 corpus-probe item 9 evidence produced. Full command executed exactly once after firewall push:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/common/run_r4_corpus_probe.py --output /media/maxim/Programs/Features/WSM/stage6_corpus_probe/r4_corpus_probe_v1.json

External report: `/media/maxim/Programs/Features/WSM/stage6_corpus_probe/r4_corpus_probe_v1.json`; SHA256 `3361de90e0cb26f14e15f639bb54ef4f7d29d54232b72fb39f154d7f7105a4dc`. Firewall commit: `3ba0fcce68550eccf61a34d4e20e55a4111212f4`; final evidence commit follows. Frozen checkpoint paths/SHA: seed42 `logs/wsm_mm_pd_dep_v1/optuna_r4_ra_stch_base_seed42_2026-09-28_01-54_wsm_av_r3_disease_query_model_r4-ra-stch-optuna-v1-18c6-012_0ce9c7a4/checkpoints/epoch=11_dev_mean_score=0.8191.pt` / `104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a`; seed43 `logs/wsm_mm_pd_dep_v1/r4_ra_stch_optuna_trial012_confirm_seed43_2026-09-28_12-01_wsm_av_r3_disease_query_model_12038580/checkpoints/epoch=10_dev_mean_score=0.7614.pt` / `6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e`; seed44 `logs/wsm_mm_pd_dep_v1/r4_ra_stch_optuna_trial012_confirm_seed44_2026-09-28_12-07_wsm_av_r3_disease_query_model_41bc602b/checkpoints/epoch=11_dev_mean_score=0.7801.pt` / `b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17`.

Canonical counts were TRAIN `6325` (depression `2665`, parkinson `3660`) and DEV `933` (depression `312`, parkinson `621`) for each seed. All six representations completed with dimensions `audio_projected=160`, `video_projected=160`, `projected_av=320`, `task_fused=320`, `gates=4`, `logits=2`. Probes used TRAIN-only standardization, CPU float64 zero initialization, class-balanced weighted BCE, weight-only L2 `1e-4`, and exact LBFGS settings from NEXT_TASK. No Test loader was iterated or inspected; no main-model training or re-selection occurred; no source/config changes occurred.

Per-seed DEV AUROC (true / shuffled; true-minus-shuffled):

- seed42: audio `0.774320 / 0.559834; +0.214486`, video `0.708318 / 0.460408; +0.247910`, projected_av `0.746470 / 0.510075; +0.236395`, task_fused `0.771058 / 0.609119; +0.161939`, gates `0.674398 / 0.670047; +0.004351`, logits `0.619214 / 0.637986; -0.018771`.
- seed43: audio `0.785747 / 0.608938; +0.176808`, video `0.715998 / 0.511288; +0.204710`, projected_av `0.740028 / 0.577640; +0.162388`, task_fused `0.739863 / 0.476501; +0.263362`, gates `0.711853 / 0.707002; +0.004852`, logits `0.616097 / 0.629000; -0.012903`.
- seed44: audio `0.813808 / 0.613521; +0.200287`, video `0.692199 / 0.483515; +0.208684`, projected_av `0.765453 / 0.530611; +0.234841`, task_fused `0.768575 / 0.541605; +0.226971`, gates `0.268565 / 0.278382; -0.009817`, logits `0.490281 / 0.646946; -0.156664`.

Three-seed true AUROC mean/sample std: audio `0.791292/0.020300`, video `0.705505/0.011934`, projected_av `0.750650/0.013012`, task_fused `0.759832/0.017222`, gates `0.551605/0.244405`, logits `0.575197/0.073718`. Shuffled true-control means: audio `0.594098`, video `0.485070`, projected_av `0.539442`, task_fused `0.542408`, gates `0.551810`, logits `0.637977`. Task-fused minus projected_av AUROC deltas by seed: `+0.024588`, `-0.000165`, `+0.003123`; mean `+0.009182`, below the frozen amplification gate.

Gate audit audio-weight mean depression-minus-parkinson: seed42 depression query `-0.123735`, Parkinson query `-0.125313`; seed43 `-0.090391`, `-0.091659`; seed44 `-0.037737`, `-0.046210`. Full mean/q25/median/q75/sample-std distributions are in the external report.

Frozen interpretation strings: `STRONG CORPUS-ID DECODABILITY FLAG NOT TRIGGERED`; `FUSION AMPLIFICATION OF CORPUS DECODABILITY NOT SUPPORTED`. Corpus identity is structurally coupled to which disease label is observed, so this remains a risk diagnostic; no causal shortcut, disease-signal replacement, R4 promotion/demotion, mitigation, or significance claim is made. Stage-5 optimization remains closed; Stage 6 remains active; Stage 7, Text/Description, and Final Test remain locked.


### TASK-006C — RA-STCH firewall

Status: firewall complete; production runs not started. Branch: `codex/task-006c`, based on manager `origin/main` `b228191`. Only `docs/PROGRESS_EN.md` is authorized to change.

Validated exactly:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config --config-path configs/wsm_mm_pd_dep_v1/fusion/16_r4_ra_stch_seed42.yaml
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config --config-path configs/wsm_mm_pd_dep_v1/fusion/17_r4_ra_stch_seed43.yaml
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config --config-path configs/wsm_mm_pd_dep_v1/fusion/18_r4_ra_stch_seed44.yaml

All three configs were valid. Programmatic equivalence passed: configs17/18 differ from config16 only by top-level seed and run_name; seeds are exactly `42/43/44`. Retained seed42 checkpoint exists with SHA256 `53397e3bbcbfc95abd30bdf63fec018a28f0effc2d92d66231061fb2d051254a`; semantic cache exists with SHA256 `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`.

Firewall smoke passed for configs17/18 using TRAIN-only canonical mixed-corpus rows, with no optimizer step and no DEV/Test iteration:

- trainable parameters: `403079` for both;
- semantic accepted pseudo counts: D/P `376/1801`;
- finite loss: seed43 `0.5106716156`, seed44 `0.5835766196`;
- finite nonzero model gradients;
- detached pseudo target/reliability tensors;
- finite RA diagnostics; seed43 grad-norm EMA D/P `0.094643/0.079261`, grad-cosine EMA `0.0`, reliability EMA D/P `0.650805/0.876119`; seed44 `0.141683/0.091648`, `0.0`, `0.650805/0.876119`;
- controller weights remained within frozen `[0.2,0.8]` bounds.

`git diff --check`, `git diff origin/main -- src`, and `git diff origin/main -- configs` passed. No source/config/script changes occurred. The mandatory firewall commit must be pushed before exactly two production runs in order: seed43, then seed44.


### TASK-006C — isolated RA-STCH balancing audit

Status: complete. Branch: `codex/task-006c`. Firewall commit `45b17ebb6d54acb602ba9698677a37b47292e38d` was pushed before production. Only the two authorized production runs occurred, in order, with no seed42 rerun, sweep, retry, config change, or optimized trial-012 run:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/fusion/17_r4_ra_stch_seed43.yaml
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/fusion/18_r4_ra_stch_seed44.yaml

Retained seed42 was not rerun: selected epoch 13, DEV D/P/Mean `0.700992/0.864298/0.782645`, checkpoint SHA256 `53397e3bbcbfc95abd30bdf63fec018a28f0effc2d92d66231061fb2d051254a`. New production identities: seed43 run `r4_ra_stch_seed43_c1_2026-09-28_16-07_wsm_av_r3_disease_query_model_39bc2679`, MLflow ID `25b36566c88046cd98614595fa682523`, FINISHED, selected epoch 4, DEV D/P/Mean `0.698839/0.864829/0.781834`, checkpoint SHA256 `aa75298c6b557eda3d7e8594a77b0c55661abd9c67cd4ed6b4f1687b8eba4262`; seed44 run `r4_ra_stch_seed44_c1_2026-09-28_16-12_wsm_av_r3_disease_query_model_49a0dea2`, MLflow ID `3d3ade0154b64f73817c6d5dc2685411`, FINISHED, selected epoch 4, DEV D/P/Mean `0.679620/0.826765/0.753193`, checkpoint SHA256 `7d9510cfef6f4a871e5447c7142aa30e4ef23798576d735f45efc7ff4774f916`. Selection used DEV/mean_score only.

Three-seed RA DEV evidence (seed42/43/44): D mean/std/range `0.6931503333/0.0117669580/0.021372`; P `0.8519640000/0.0218245890/0.038064`; Mean `0.7725573333/0.0167749060/0.029452`. Same-seed RA minus Equal D/P/Mean: seed42 `-0.011506/+0.020956/+0.004725`; seed43 `-0.004460/+0.008786/+0.002163`; seed44 `-0.028357/-0.025113/-0.026734`. Three-seed RA minus Equal means: D/P/Mean `-0.0147743333/+0.0015430000/-0.0066153333`.

Same-seed RA minus Static-STCH D/P/Mean: seed42 `-0.024085/+0.014813/-0.004636`; seed43 `-0.000106/+0.011645/+0.005770`; seed44 `-0.022452/-0.026857/-0.024654`. Same-seed RA minus corrected Progress D/P/Mean: seed42 `-0.000253/-0.009056/-0.004654`; seed43 `-0.022758/+0.001698/-0.010530`; seed44 `-0.022351/-0.020443/-0.021397`. Best-simple (`max(Static,Progress)`) RA Mean deltas: seed42 `-0.004654`; seed43 `-0.010530`; seed44 `-0.024654`; mean `-0.0132793333`. RA minus contextual corrected R3-B means (D/P/Mean `0.696178/0.857996/0.777087`) is `-0.0030276667/-0.0060320000/-0.0045296667`.

Frozen claims: `RA-STCH BALANCING CONTRIBUTION OVER EQUAL NOT SUPPORTED`; `RA-STCH ADVANTAGE OVER SIMPLER BALANCING NOT SUPPORTED`. Rule A fails because RA three-seed Mean is below Equal despite exceeding Equal on 2/3 seeds; Rule B fails all same-seed best-simple Mean comparisons and both simpler three-seed Mean comparisons.

Selected-epoch controller diagnostics (alpha D/P; progress D/P; grad-norm EMA D/P; grad-cosine EMA; reliability EMA D/P; pseudo scale): seed42 `0.269192/0.730808; +0.020070/-0.151804; 0.248995/0.121096; +0.009814; 0.671974/0.773563; 1.0`; seed43 `0.353600/0.646400; +0.001794/-0.366308; 0.451421/0.248391; -0.030544; 0.672728/0.770895; 0.2`; seed44 `0.362685/0.637315; -0.051906/-0.264870; 0.280293/0.130971; +0.014310; 0.671132/0.781050; 0.2`. All diagnostics are finite; alpha is within `[0.2,0.8]`, seed-specific rather than identical, grad-cosine range is `-0.030544` to `+0.014310`, reliability D range `0.671132–0.672728`, P range `0.770895–0.781050`.

Post-freeze DEV-only calibration/gate audit used observed counts D/P `621/312` for every seed and no Test rows. RA three-seed means: D Brier/ECE-15 `0.2505394063/0.2257061193`, P `0.0952518079/0.0757453021`; task audio-gate mean/sample-std D `0.3630868495/0.1935827980`, P `0.3701813122/0.1922030300`. RA minus frozen Equal calibration means: D Brier/ECE `-0.0015246377/-0.0058562747`, P `+0.0046632639/+0.0142875391`. RA minus corrected Progress: D `+0.0025287953/+0.0075696203`, P `+0.0100845399/+0.0059937391`.

Same-epoch Test monitoring was read only after each selected checkpoint was frozen and did not affect selection, calibration, comparison, or claims: seed42 TEST_NONE/SOFT/HARD `0.777133/0.767804/0.760071`; seed43 `0.746138/0.739737/0.729451`; seed44 `0.797072/0.798145/0.805330`. No Test row was used in post-hoc diagnostics. Frozen pseudo coverage remains D/P `376/1801`, with class counts D `376/0` and P `212/1589`. Stage-5 optimization remains closed; Stage 6 remains active; Stage 7, Text/Description, and Final Test remain locked.


### TASK-006D — no direct pseudo-supervision firewall

Status: firewall complete; production runs not started. Branch: `codex/task-006d`, based on manager `origin/main` `b927bf8`. Added only configs40/41/42; no source, existing-config, or script changes.

Validated all three configs with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config --config-path ...`: all valid. Programmatic comparison against fusion refs37/38/39 passed: only `experiment_info.params.run_name` and warmup `final_scale` differ; required seeds are 42/43/44, `loss.params.pseudo_scale=0.0`, observed-only epochs `3`, ramp epochs `5`, and cache/reliability settings unchanged. `scale_for_epoch(1..30)==0.0`.

Firewall command completed with TRAIN-only forward/loss/backward checks for all three configs: frozen semantic cache SHA `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`; accepted counts D/P `376/1801`, classes D `376/0`, P `212/1589`; accepted rows present; trainable parameters `403079`; pseudo targets/reliability detached; controller diagnostics finite and reliability EMA updated from accepted reliability. Deterministic pseudo-target perturbation left total loss and model gradients unchanged within strict `1e-12` tolerance because pseudo scale is zero. No optimizer step and no DEV/Test iteration occurred. `git diff origin/main -- src` and `git diff origin/main -- configs` were empty; `git diff --check` passed.

Mandatory firewall commit/push precedes exactly three production runs in order seed42, seed43, seed44.


### TASK-006D — three-seed no direct pseudo-supervision ablation

Status: complete; Stage-6 direct pseudo-supervision ablation evidence produced. Firewall commit `5a5a657` was pushed before production. Exactly three production runs occurred, in order, with no retries, sweep, extra seed, optimized-trial rerun, or post-firewall config change:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/ablations/40_no_direct_pseudo_seed42.yaml
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/ablations/41_no_direct_pseudo_seed43.yaml
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/ablations/42_no_direct_pseudo_seed44.yaml

Runs were FINISHED in MLflow: seed42 `06355f9a57154ccf8201595ff3982098`, seed43 `8b2eeb1879a840bbbeaf72d7e081999e`, seed44 `289fd655d74e4e75a89a50af7df32b49`. DEV/mean_score-only selected checkpoints:

| Seed | Epoch | DEV D/P/Mean | Checkpoint SHA256 |
|---:|---:|---|---|
| 42 | 7 | `0.750015/0.849786/0.799900` | `d0991ddf92b8fd556d1c57dfa3c17b3b083e5b16a1e21d04592cece804b57669` |
| 43 | 19 | `0.683769/0.770704/0.727236` | `f5958bd638b021836e9ead892815c17822579fb796d7e074ad61c6a86fc96892` |
| 44 | 12 | `0.706529/0.858468/0.782498` | `d67232b2ba3395ced7a1ac95e95aacfbdaa354ce68ee5e507f37954f54fd968c` |

No-direct three-seed DEV means/sample SDs: D `0.7134376667/0.0336590313`, P `0.8263193333/0.0483595209`, Mean `0.7698780000/0.0379402494`. Frozen full trial-012 means are D/P/Mean `0.7352563333/0.8385606667/0.7869088179`; full-minus-ablation mean deltas are `+0.0218186667/+0.0122413333/+0.0170308179`. Full minus ablation per-seed Mean deltas are seed42 `+0.0192424538`, seed43 `+0.0342090000`, seed44 `-0.0023590000`. The predeclared gate passes (`2/3` per-seed DEV Mean wins, higher three-seed Mean, and task means no more than `0.010` lower), so the exact frozen claim is: `DIRECT PSEUDO-SUPERVISION CONTRIBUTION SUPPORTED`. This concerns only the direct pseudo BCE term.

Selected-epoch controller diagnostics (alpha D/P; progress D/P; grad-norm EMA D/P; grad-cosine EMA; reliability EMA D/P; pseudo scale): seed42 `0.312667/0.687333; +0.190917/-0.493318; 0.279859/0.266622; 0.000000; 0.674348/0.757363; 0.000000`; seed43 `0.212604/0.787396; -0.137111/-0.608890; 0.437248/0.243887; 0.000000; 0.673802/0.779818; 0.000000`; seed44 `0.256048/0.743952; +0.094125/-0.245362; 0.386038/0.230013; 0.000000; 0.663937/0.782318; 0.000000`. All finite; alpha remains within `[0.2,0.8]`. Accepted pseudo coverage remained D/P `376/1801`, classes D `376/0` and P `212/1589`; reliability/controller paths remained active.

Post-freeze DEV-only calibration/gate audit iterated only `dm.val_dataset`, with observed counts D/P `621/312` per seed and no Test row. Per-seed `(D Brier, D ECE-15, D audio-gate mean/std; P Brier, P ECE-15, P audio-gate mean/std)`:

- seed42: `0.180180/0.110249/0.433490/0.195184; 0.084544/0.045940/0.559094/0.182210`
- seed43: `0.264118/0.231195/0.376595/0.144412; 0.160619/0.153661/0.437323/0.140837`
- seed44: `0.218597/0.182075/0.430556/0.156280; 0.087115/0.079878/0.490820/0.121299`

Three-seed no-direct calibration/gate means: D Brier/ECE `0.2209647345/0.1745065810`, P `0.1107593512/0.0931596507`, audio gate D mean/std `0.4135470291/0.1652921041`, P `0.4957457086/0.1481153195`. Frozen full means were D Brier/ECE `0.1789253730/0.0700041175`, P `0.0899480473/0.0898388979`, gate D `0.4017960926/0.1487621441`, P `0.4908300142/0.1301527048`; full-minus-ablation deltas respectively `-0.0420393616/-0.1045024635`, `-0.0208113039/-0.0033207529`, `-0.0117509365/-0.0165299600`, `-0.0049156944/-0.0179626147`. No recalibration or threshold tuning occurred.

Same-epoch Test monitoring was read only after each DEV checkpoint was frozen: seed42 TEST_NONE/SOFT/HARD `0.803498/0.812067/0.803166`, seed43 `0.805364/0.801955/0.811135`, seed44 `0.796826/0.802473/0.793822`. Test did not affect selection, audit, comparison, or claim; no Test row was used in post-hoc diagnostics. No pseudo-label correctness, missing-label recovery, comorbidity recovery, significance, domain mitigation, promotion, or demotion claim is made. Stage-5 optimization remains closed; Stage 6 remains active; Stage 7, Text/Description, and Final Test remain locked.

Final scope checks: `git diff --check` passed; `git diff origin/main -- src` and `git diff origin/main -- configs` are empty; only configs40/41/42 and this ledger differ from origin/main.


### TASK-006E — uniform accepted-reliability firewall

Status: firewall complete; production runs not started. Branch: `codex/task-006e`, based on manager `origin/main` `919a5a7`.

Added [`scripts/common/build_ramps_uniform_reliability_ablation.py`](../scripts/common/build_ramps_uniform_reliability_ablation.py) and configs43/44/45. The builder command was:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/common/build_ramps_uniform_reliability_ablation.py --source /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt --output /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_reliability_ablation/uniform_accepted_reliability_v1.pt

Source SHA before/after: `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`. Derived SHA: `713b5a3d963c8759e4b2c12f148be12818e72c135b6d1420207cf5aa150bcc40`. The builder refuses overwrite, preserves canonical version and all non-reliability fields, and records only explicit `reliability_ablation` metadata. Accepted counts D/P `376/1801`; classes D `376/0`, P `212/1589`; accepted reliability is exactly `1.0`; rejected/observed reliability exactly `0.0`. Source accepted reliability min/mean/max: D `0.6104820873/0.6726164732/0.7853315456`, P `0.5000977972/0.7718680412/0.9037118705`. Changed accepted reliability entries D/P: `376/1801`.

All configs validated successfully. Programmatic equivalence against refs37/38/39 passed: only run_name and pseudo_cache_path differ; seeds remain 42/43/44. Frozen full checkpoint files and all three required SHA256 values verified. TRAIN-only firewall smoke passed for every config: derived cache loaded through the existing DataModule; trainable parameters `295239`; accepted pseudo rows present; pseudo targets/reliability detached; direct pseudo supervision at scale `1.0` produced finite nonzero model gradients; RA reliability EMA updated exactly to `[1.0, 1.0]`; controller diagnostics were finite. No optimizer step and no DEV/Test iteration occurred. `git diff --check` passed and `git diff origin/main -- src` was empty.

Mandatory firewall commit/push precedes exactly three production runs in order seed42, seed43, seed44.


### TASK-006E — uniform accepted-reliability production ablation

Status: complete; Stage-6 uncertainty/reliability evidence produced. Firewall commit `6991153` was pushed before production. Exactly three production runs occurred in order, with no sweep, retry, extra seed, or post-firewall cache/config change:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/ablations/43_uniform_reliability_seed42.yaml
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/ablations/44_uniform_reliability_seed43.yaml
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/ablations/45_uniform_reliability_seed44.yaml

All MLflow runs were FINISHED: seed42 `017b81c94ec24726a61e448723634dd5`, seed43 `c8247e56925f434e8c41b39eac764e82`, seed44 `ebe5d6fcd91749d5893613dc3d8b9d7d`. DEV/mean_score-only selected checkpoints:

| Seed | Epoch | DEV D/P/Mean | Checkpoint SHA256 |
|---:|---:|---|---|
| 42 | 11 | `0.755716/0.871962/0.813839` | `e3eeab3eb974c65b4410b35a183e42cbe0d484cc0e21f231543ef0f4ffe51065` |
| 43 | 10 | `0.720095/0.799734/0.759915` | `76098413ad16314f0575a1b574a8d8bf851fa6bc67039f154acc0272eba94554` |
| 44 | 11 | `0.729964/0.832060/0.781012` | `3de5862828ddabe36c507990780fe690269079520d99d8a9111f67c11ffb78b7` |

Uniform-reliability three-seed DEV mean/sample SD/range: D `0.7352583333/0.0183912040/0.035621`, P `0.8345853333/0.0361801600/0.072228`, Mean `0.7849220000/0.0271738021/0.053924`. Frozen full trial-012 mean is D/P/Mean `0.7352563333/0.8385606667/0.7869088179`; aggregate full-minus-ablation deltas are D `-0.0000020000`, P `+0.0039753333`, Mean `+0.0019868179`. Per-seed full-minus-ablation D/P/Mean deltas: seed42 `+0.003309/+0.007297/+0.0053034538`, seed43 `-0.001568/+0.004629/+0.0015300000`, seed44 `-0.001747/+0.000000/-0.0008730000`.

The exact frozen gate passes: full DEV Mean wins `2/3` seeds, full three-seed Mean is higher, and full D/P means are not more than `0.010000` below the ablation. Record exactly: `UNCERTAINTY/RELIABILITY CONTRIBUTION SUPPORTED`. This concerns the implemented graded reliability signal as a whole, jointly covering pseudo-BCE reliability weighting and RA reliability input; it does not separate those effects.

Selected controller diagnostics (alpha D/P; progress D/P; grad-norm EMA D/P; grad-cosine EMA; reliability EMA D/P; pseudo scale): seed42 `0.391610/0.608390; +0.143760/+0.082452; 0.379403/0.259262; +0.045782; 1.000000/1.000000; 1.000000`; seed43 `0.249567/0.750433; +0.050214/-0.701800; 0.356957/0.284160; -0.005376; 1.000000/1.000000; 1.000000`; seed44 `0.291637/0.708363; +0.119063/-0.194524; 0.344999/0.259215; +0.009421; 1.000000/1.000000; 1.000000`. All diagnostics were finite and reliability EMA reached exactly `1.0/1.0` for every selected run.

Post-freeze DEV-only calibration/gate audit used observed counts D/P `621/312` per seed and no Test rows. Per-seed `(D Brier, D ECE-15, D audio-gate mean/std; P Brier, P ECE-15, P audio-gate mean/std)`:

- seed42: `0.175110/0.075461/0.419239/0.196150; 0.071968/0.069305/0.549952/0.170355`
- seed43: `0.188070/0.093503/0.349542/0.111722; 0.111041/0.088769/0.433419/0.103559`
- seed44: `0.176722/0.062885/0.435787/0.134986; 0.087910/0.115290/0.488573/0.111126`

Uniform-reliability three-seed means: D Brier/ECE `0.1799669747/0.0772831558`, P `0.0903061726/0.0911213222`, audio gate D mean/std `0.4015226364/0.1476193691`, P `0.4906478624/0.1283467487`. Full-minus-ablation deltas: D Brier/ECE `-0.0010416017/-0.0072790382`, P `-0.0003581253/-0.0012824243`, gate D mean/std `+0.0002734562/+0.0011427750`, P `+0.0001821518/+0.0018059562`. No recalibration or threshold search occurred.

Same-epoch Test monitoring was read only after checkpoint freeze: seed42 TEST_NONE/SOFT/HARD `0.801394/0.811302/0.808134`; seed43 `0.820499/0.828142/0.841820`; seed44 `0.822560/0.833204/0.850724`. Test did not affect selection, interpretation, or follow-up, and no Test rows entered the post-hoc audit. No pseudo-label correctness, semantic acceptance correctness, missing-label recovery, comorbidity recovery, significance, or final-promotion claim is made. Stage-5 optimization remains closed; Stage 6 remains active; Stage 7, Text/Description, and Final Test remain locked.

Final scope checks: source cache remained SHA `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`; derived cache remains external at SHA `713b5a3d963c8759e4b2c12f148be12818e72c135b6d1420207cf5aa150bcc40`; `git diff --check` passed; `git diff origin/main -- src` was empty; only the builder, configs43/44/45, and this ledger are tracked changes.


### TASK-006F — no semantic-enabled depression pseudo path firewall

Status: firewall complete; production runs not started. Branch: `codex/task-006f`, based on manager `origin/main` `6fd035c`.

Authorized source change: [`src/fusion/data/wsm_ramps_semantic_datamodule.py`](../src/fusion/data/wsm_ramps_semantic_datamodule.py) now accepts optional expected accepted/positive/negative count contracts, defaulting exactly to `(376,1801)`, `(376,212)`, `(0,1589)`. Supplied contracts require exactly two non-negative integers; expected missing counts remain fixed at `(2665,3660)` and all other cache invariants are unchanged.

Full backward compatibility passed for configs37/38/39 with omitted parameters: full counts remained D/P `376/1801`, classes D `376/0`, P `212/1589`, and a canonical TRAIN batch was tensor-identical against an isolated clean `origin/main` DataModule implementation.

Builder: [`scripts/common/build_ramps_no_semantic_depression_ablation.py`](../scripts/common/build_ramps_no_semantic_depression_ablation.py). Source SHA before/after: `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`. Derived cache: `/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_semantic_ablation/no_semantic_depression_v1.pt`, SHA `aae9d250c5030eab1e68e07ee8af43fe8ebb7286eca621f9fa5c6dd7af283fe6`. Exactly 376 D accepted rows were removed; all 1801 P accepted rows, targets, reliability, and classes were preserved exactly. D fields are neutralized to accept false, NaN targets, zero reliability, and class `-1`.

Configs46/47/48 validated and differ from refs37/38/39 only by run_name, cache path, and the three explicit expected-count contracts. Frozen full checkpoint paths/SHA values were verified. TRAIN-only firewall passed for all three ablation configs: trainable parameters `295239`; D accepted `0`; P accepted `1801`, classes `212/1589`; P pseudo gradients finite/nonzero at pseudo scale `1.0`; D pseudo contribution absent; pseudo tensors detached; P reliability EMA finite (`0.738575935` in the deterministic smoke batch); D reliability EMA uninitialized with controller fallback `0.5`; controller weights/diagnostics finite. No optimizer step and no DEV/Test loader iteration occurred. `git diff --check` passed; all forbidden source scopes (`src/audio`, `src/video`, `src/fusion/models`, `src/fusion/loss`, `src/common/callbacks`) were empty.

Mandatory firewall commit/push precedes exactly three production runs in order seed42, seed43, seed44.



### TASK-006F final semantic-evidence ablation evidence

Status: complete. Branch codex/task-006f. Historical basis: D accepted 0 and P accepted 1801 rows. Firewall commit 1c8b56977f9731fcefd4bf8df98149faa1915c28 was pushed before production. Exactly three runs were executed in order, with no retry, sweep, tuning, or extra seed.

Derived cache: /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_semantic_ablation/no_semantic_depression_v1.pt. Derived SHA aae9d250c5030eab1e68e07ee8af43fe8ebb7286eca621f9fa5c6dd7af283fe6. Source SHA 17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945. Exactly 376 D rows were removed and all 1801 P rows were preserved exactly. Configs46/47/48 validated against refs37/38/39 with only run name, cache path, and expected-count contracts changed.

Finished MLflow runs: seed42 49be8f3bc9d747bba9de19b82aab363b, seed43 e9bd2ba394c643dc862e99922c198b76, seed44 4d33b0088fd44cbbacd4fcf42a2b3f2e; all FINISHED. DEV-only selected results: seed42 epoch15 D/P/Mean 0.733193/0.893824/0.813509, checkpoint 0a361e8ce634f74ef1f544716f2c1bee2fef6910ee2610666f4742e7d956745d; seed43 epoch10 0.692111/0.804363/0.748237, checkpoint 666ed0785adf21e67cffe122f53641ad5ea5926e94ef63c8eeeccf08e08be20a; seed44 epoch28 0.718037/0.906919/0.812478, checkpoint a536b6c07c0fecb49768cc11410b0df052254a1abf28535686895e71aa34fe6c.

Ablation three-seed means/std/ranges: D 0.714447/0.020775/0.041082, P 0.868369/0.055816/0.102556, Mean 0.791408/0.037391/0.065272. Full-minus-ablation deltas: D +0.020809, P -0.029808, Mean -0.004499. Per-seed D/P/Mean deltas: seed42 +0.025832/-0.014565/+0.005633, seed43 +0.026416/0.000000/+0.013208, seed44 +0.010180/-0.074859/-0.032339. Frozen result: SEMANTIC-EVIDENCE CONTRIBUTION VIA DEPRESSION PSEUDO ACCEPTANCE NOT SUPPORTED.

DEV-only audit observed D/P counts 621/312. Ablation means D Brier/ECE 0.189869/0.105671 and gate mean/std 0.369660/0.144342; P 0.078377/0.060888 and gate 0.448038/0.134718. Full-minus-ablation deltas: D Brier/ECE -0.010944/-0.035667 and gate +0.032136/+0.004420; P +0.011571/+0.028951 and gate +0.042792/-0.004565. Controller diagnostics were finite; selected alpha D/P: seed42 0.418806/0.581194, seed43 0.265542/0.734458, seed44 0.365922/0.634078. Reliability EMA D/P: 0.500000/0.769233, 0.500000/0.781396, 0.500000/0.779168. D accepted count was zero with fallback 0.5; P accepted count 1801 with classes 212/1589; pseudo scale was 1.0.

Same-epoch Test monitoring was read only after freeze: seed42 0.796484/0.803648/0.815369, seed43 0.815810/0.820375/0.837140, seed44 0.775684/0.778415/0.789896 for NONE/SOFT/HARD. Test affected no decision and no Test rows entered the audit. No correctness, missing-label, comorbidity, significance, causality, or final-promotion claim is made. git diff --check passed; forbidden source scopes and src/audio were unchanged; only authorized paths differ from origin/main.


### TASK-006G firewall

Branch codex/task-006g. Added only an optional modality_available_override to the semantic DataModule/collate path. Null/default behavior passed NaN-safe tensor comparison against the canonical full semantic collate; configs49/50/51 are equivalent to refs37/38/39 except run_name and [true,false] override. Config validation passed for all three.

TRAIN-only firewall passed before production: full pseudo cache SHA 17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945, accepted counts D/P 376/1801, class counts D 376/0 and P 212/1589; no-video availability was [true,false] for every row; trainable parameters remained 295239. Model weights were exactly audio 1/video 0, features_video exactly zero, video_aux_valid all false, audio_aux_valid all true. A strongly perturbed raw video tensor left logits, loss, and all non-video gradients unchanged within 1e-7; video-only gradients were zero or None. Accepted pseudo supervision produced finite nonzero active-path gradients; pseudo targets/reliability were detached; controller diagnostics were finite. No optimizer step and no DEV/Test iteration occurred. Forbidden source scopes and src/audio remained unchanged.

Firewall verification was completed with no production run started.


### TASK-006G final no-video modality-removal evidence

Status: complete; branch codex/task-006g. The semantic DataModule accepts optional modality_available_override; default null remains backward-compatible and [true,false] forces audio available/video unavailable without changing raw tensors, masks, pseudo fields, labels, IDs, or split membership. Firewall commit: e6ac651. Exactly three production runs were executed after the firewall, in order, with no sweep, retry, tuning, extra seed, or post-firewall change.

Run identities and DEV-only selected checkpoints:
- seed42 MLflow 56a6fceb846e440e9f5ee4425c64418c, run directory stage6_no_video_trial012_seed42_2026-09-28_20-44_wsm_av_r3_disease_query_model_b26772b5, selected epoch30, checkpoint SHA b5105863d9fd9da293abc5c5fa2e8b466faa2eb62e1ca73700e0696b3d3f4dfd. DEV D UAR/MF1/Score 0.742997/0.742422/0.742710; P 0.778330/0.800766/0.789548; Mean 0.766129.
- seed43 MLflow f324499e3b7c416582c847e7decf1306, run directory stage6_no_video_trial012_seed43_2026-09-28_20-55_wsm_av_r3_disease_query_model_c36f4ab6, selected epoch14, checkpoint SHA 00f68d7e6a47fa979f3529dcf951c569a6299734cf8760fff4950cf8f181f0bd. DEV D UAR/MF1/Score 0.745238/0.745280/0.745259; P 0.773568/0.796060/0.784814; Mean 0.765037.
- seed44 MLflow 95321ff8ce334d5e8846ae06f009a616, run directory stage6_no_video_trial012_seed44_2026-09-28_21-03_wsm_av_r3_disease_query_model_cc4c3b54, selected epoch12, checkpoint SHA 45be3f38b6a2dcaada2045b1a05430a5b7aaed32c1fe295816759cd0e85d3775. DEV D UAR/MF1/Score 0.761018/0.760915/0.760966; P 0.764044/0.786530/0.775287; Mean 0.768127.

Same-epoch Test monitoring was read only after each checkpoint freeze. NONE/SOFT/HARD Mean: seed42 0.818529/0.831500/0.845517; seed43 0.776637/0.787068/0.806849; seed44 0.796877/0.817045/0.833882. Test did not affect selection, claim, or follow-up and no Test rows entered the calibration audit.

No-video three-seed DEV mean/std/range: D 0.749645/0.009887/0.018256; P 0.783216/0.007264/0.014261; Mean 0.766431/0.001567/0.003090. Frozen full-minus-no-video deltas: D -0.014389, P +0.055344, Mean +0.020478. Per-seed full-minus-no-video D/P/Mean: seed42 +0.016315/+0.089711/+0.053013; seed43 -0.026732/+0.019549/-0.003592; seed44 -0.032749/+0.056773/+0.012012. The frozen rule fails criterion 3 because full three-seed D is 0.014389 below no-video, exceeding 0.010000. Exact claim: VIDEO MODALITY CONTRIBUTION NOT SUPPORTED.

Selected controller diagnostics alpha D/P; progress D/P; grad-norm EMA D/P; grad-cosine EMA; reliability EMA D/P; pseudo scale:
- seed42 0.240506/0.759494; +0.124813/-0.597987; 0.548988/0.529721; +0.018974; 0.668266/0.791510; 1.0.
- seed43 0.258275/0.741725; +0.169270/-0.793759; 0.428030/0.485103; +0.018560; 0.672106/0.773204; 1.0.
- seed44 0.227573/0.772427; +0.196643/-0.915312; 0.432169/0.476490; +0.035741; 0.663937/0.782318; 1.0.
Task modality weights were exactly audio 1/video 0 at selected evaluation. TRAIN pseudo cache identity remained SHA 17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945 with accepted counts D/P 376/1801 and classes D 376/0, P 212/1589.

Post-freeze DEV calibration audit observed counts were D/P 621/312. Per-seed D Brier/ECE-15: 0.170881/0.063989, 0.173205/0.061858, 0.169655/0.064567. Per-seed P: 0.119443/0.091652, 0.126624/0.101381, 0.130272/0.115606. Audio gate mean/std was exactly 1.000000/0.000000 for all seeds. Ablation calibration means D Brier/ECE 0.171247/0.063471 and P 0.125446/0.102880. Full-minus-ablation calibration deltas: D +0.007678/+0.006533; P -0.035498/-0.013041. No recalibration or threshold search occurred.

Contextual frozen matched temporal-audio DEV values were D/P/Mean seed42 0.747918/0.827735/0.787827, seed43 0.708699/0.811771/0.760235, seed44 0.734396/0.809382/0.771889; no-video minus audio Mean was -0.021698/+0.004802/+0.003762 by seed and -0.006886 in the three-seed mean. This is contextual only and not an architecture-equivalent comparison.

Final checks: config validation passed for configs49/50/51; default full behavior and config equivalence passed; git diff --check passed; src/audio, src/video, src/fusion/models, src/fusion/loss, and src/common/callbacks remained unchanged; no optimizer step or DEV/Test iteration occurred in the firewall; only the authorized DataModule source, configs49/50/51, and this ledger differ from origin/main. No causal, significance, disease-content, missing-label, comorbidity, or final-promotion claim is made. Stage 5 remains closed; Stage 6 remains active; Stage 7, Text/Description, and Final Test remain locked.


### TASK-006H firewall

Branch codex/task-006h. Configs52/53/54 validated and are equivalent to refs37/38/39 except run_name and modality_available_override [false,true]; seeds remain 42/43/44. Frozen full trial-012 checkpoint SHA256 values were verified: seed42 104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a, seed43 6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e, seed44 b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17.

TRAIN-only firewall passed before production. Pseudo cache SHA remained 17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945; accepted counts D/P 376/1801; class counts D 376/0 and P 212/1589. All TRAIN rows exposed availability [false,true]. Trainable parameters remained 295239. Audio modality weight was exactly 0 and video exactly 1; features_audio was exactly zero; audio_aux_valid was all false and video_aux_valid all true. Strong perturbation of audio_cls and temporal audio left logits, total loss, and non-audio-specific gradients unchanged within 1e-7. Audio projection and audio auxiliary-head gradients were zero; active video gradients were finite and nonzero. Observed and accepted pseudo supervision remained active at pseudo_scale 1.0; pseudo targets/reliability were detached; controller diagnostics were finite. No optimizer step and no DEV/Test iteration occurred. git diff origin/main -- src was empty. Firewall evidence passed before any production run.


### TASK-006H final no-audio online-input ablation evidence

Status: complete; branch codex/task-006h. The already-merged modality availability implementation was reused unchanged. The experiment removes only the online student audio input branch with availability [false,true]. The frozen pseudo cache remains unchanged and was built from historical audio/video teacher evidence including calibrated audio probabilities; this is not a pure removal of all audio-derived information.

Exactly three production runs were executed in order with no sweep, retry, tuning, extra seed, or post-firewall change. Firewall commit: 2d008e3. MLflow runs:
- seed42: run 5360fbad0d31412f844e09c3a8634ba5; directory stage6_no_audio_trial012_seed42_2026-09-28_21-37_wsm_av_r3_disease_query_model_3ed3db4c; selected epoch15; checkpoint SHA 65dd8cdb87b714f182c48236fd2fcd0f2edd888d631a4ef96923bb7962445c8e.
- seed43: run 680298d3e42b4f6eab47eb9e7b64d6ea; directory stage6_no_audio_trial012_seed43_2026-09-28_21-44_wsm_av_r3_disease_query_model_16bb529f; selected epoch17; checkpoint SHA 33fd0aa9feff542cabae342f1e7707cdd258de7bf66cf922165e7ca8a0bf8b84.
- seed44: run 807d52750853430cb47142b94e1ccebe; directory stage6_no_audio_trial012_seed44_2026-09-28_21-53_wsm_av_r3_disease_query_model_fce7eb00; selected epoch15; checkpoint SHA d7225afca6be82fa8cd173c4bc296b3cd6d77fc5df61b2bfc6166b87a0a42e64.

Selected DEV metrics, D UAR/MF1/Score; P UAR/MF1/Score; Mean:
- seed42: 0.663072/0.663023/0.663048; 0.777847/0.787413/0.782630; 0.722839.
- seed43: 0.621709/0.621578/0.621643; 0.792133/0.800256/0.796194; 0.708919.
- seed44: 0.675630/0.675237/0.675434; 0.792202/0.802331/0.797266; 0.736350.

Same-epoch Test monitoring was read only after checkpoint freeze. NONE/SOFT/HARD Mean: seed42 0.719794/0.724139/0.711500; seed43 0.722973/0.722234/0.715096; seed44 0.712371/0.717666/0.716429. Test affected no selection, claim, or next-step decision and no Test rows entered the calibration audit.

No-audio three-seed DEV mean/std/range: D 0.653375/0.028170/0.053791; P 0.792030/0.008158/0.014636; Mean 0.722703/0.013716/0.027431. Full-minus-no-audio deltas: D +0.081881, P +0.046531, Mean +0.064206. Per-seed full-minus-no-audio D/P/Mean: seed42 +0.095977/+0.096629/+0.096303; seed43 +0.096884/+0.008169/+0.052526; seed44 +0.052783/+0.034794/+0.043789. All frozen claim criteria pass. Exact claim: ONLINE AUDIO INPUT MODALITY CONTRIBUTION SUPPORTED. This concerns only the online student audio branch under the unchanged pseudo-training pipeline.

Selected controller diagnostics alpha D/P; progress D/P; grad-norm EMA D/P; grad-cosine EMA; reliability EMA D/P; pseudo scale:
- seed42 0.295119/0.704881; -0.116082/-0.587310; 0.451352/0.347683; +0.021724; 0.671878/0.769233; 1.0.
- seed43 0.314330/0.685670; -0.264692/-0.556062; 0.543409/0.294848; +0.019771; 0.665876/0.771728; 1.0.
- seed44 0.278227/0.721773; -0.066403/-0.389670; 0.402969/0.263181; -0.000277; 0.670356/0.763186; 1.0.
Task modality weights were exactly audio 0/video 1 at selected evaluation. Pseudo cache SHA remained 17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945 with accepted counts D/P 376/1801 and classes D 376/0, P 212/1589.

DEV-only post-freeze calibration observed counts were D/P 621/312. Per-seed D Brier/ECE-15: 0.217216/0.133699, 0.244519/0.189715, 0.205690/0.096362. Per-seed P: 0.142127/0.098603, 0.142571/0.084615, 0.143890/0.121325. Audio gate mean/std was exactly 0.000000/0.000000 and video gate mean/std exactly 1.000000/0.000000 for all seeds. Three-seed ablation calibration means D Brier/ECE 0.222475/0.139925 and P 0.142863/0.101514. Full-minus-ablation calibration deltas: D -0.043550/-0.069921; P -0.052915/-0.011675. No recalibration or threshold search occurred.

Contextual Stage-2 V2 video reference: D/P/Mean 0.6201013364/0.7930427585/0.7065720475. No-audio minus Stage-2 video per seed: seed42 +0.042947/-0.010413/+0.016267; seed43 +0.001542/+0.003151/+0.002347; seed44 +0.055333/+0.004223/+0.029778. This is contextual and not architecture-equivalent.

Final scope checks: config validation, exact equivalence, frozen full checkpoint SHA verification, and firewall all passed; git diff --check passed; git diff origin/main -- src was empty; only configs52/53/54 and this ledger differ. No source, existing config, or pseudo-cache changes occurred. No significance, total audio-information removal, missing-label, comorbidity, causality, or final-promotion claim is made. Stage 6 modality-removal item 7 is complete; Stage 5 remains closed; Stage 7, Text/Description, and Final Test remain locked.


### TASK-006I — equal-parameter task-aware fusion firewall

Status: firewall complete; production runs not started. Branch: codex/task-006i, based on manager origin/main 0aa4e5a. Authorized changes are limited to the R3 model switch, configs55/56/57, and this ledger.

Implemented optional boolean model.params.task_aware_fusion, default true; the clean origin/main class and modified default path produced bitwise-identical preds and all relevant auxiliary tensors on the same deterministic TRAIN batch. The false path uses the frozen shared query mean, averaged candidate norms, one shared gate/weight vector, averaged fusion norms, duplicated task features/candidates/weights, and retains separate disease and auxiliary heads. Full and shared models each have exactly 295239 trainable parameters.

Config validation passed for configs55/56/57. Programmatic equivalence against refs37/38/39 passed: only run_name and task_aware_fusion=false differ; seeds remain 42/43/44. Frozen full checkpoint SHA256 values verified: seed42 104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a, seed43 6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e, seed44 b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17.

TRAIN-only firewall passed: semantic pseudo cache SHA 17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945, accepted D/P 376/1801, accepted positive D/P 376/212, accepted negative D/P 0/1589; shared audio/video candidates, modality weights, and fused features were exactly equal across task rows; separate-head perturbation was isolated in deterministic eval mode. Both query rows received equal finite nonzero gradients; candidate norms, fusion norms, projections, shared gate, main heads, and all active auxiliary heads had finite gradients. Pseudo targets/reliability were detached; observed and accepted pseudo supervision were active at pseudo_scale=1.0; RA grad/reliability/controller diagnostics were finite. No optimizer step and no DEV/Test loader iteration occurred.

Firewall checks also passed git diff --check, and forbidden source scopes remained unchanged. Mandatory firewall commit must be pushed before exactly three production commands, seed42 then seed43 then seed44; no sweep, tuning, retry, or post-firewall change is authorized.


### TASK-006I — final shared-fusion ablation evidence

Status: complete; Stage-6 task-aware-fusion item closed negative. Firewall commit cf4f9c4 was pushed before production. Exactly three production commands were executed once and in order: configs55 seed42, 56 seed43, 57 seed44. No sweep, tuning, retry, extra seed, or post-firewall change occurred. MLflow/run identities were stage6_shared_fusion_trial012_seed42_2026-09-28_22-44_wsm_av_r3_disease_query_model_5db549e8, stage6_shared_fusion_trial012_seed43_2026-09-28_22-51_wsm_av_r3_disease_query_model_454bf80b, and stage6_shared_fusion_trial012_seed44_2026-09-28_22-57_wsm_av_r3_disease_query_model_a398d7d8.

DEV-only maximum mean_score selected checkpoints:
- seed42 epoch11, DEV D UAR/MF1/Score 0.765546/0.765102/0.765324; P 0.875983/0.891228/0.883606; Mean 0.824465; SHA256 21504702976a960ffea02a68778ebc3bf7e6cc15ea8c9866f182ff04cbd30785.
- seed43 epoch10, DEV D UAR/MF1/Score 0.725070/0.723922/0.724496; P 0.809179/0.827126/0.818152; Mean 0.771324; SHA256 ec746e519e551d69d59954dffff0a73968fb4ab555d43020b7722e635cc477ee.
- seed44 epoch11, DEV D UAR/MF1/Score 0.731746/0.731149/0.731448; P 0.826087/0.850658/0.838372; Mean 0.784910; SHA256 215e3a61c5dcb8ff7dd92ed5ca9336d66405c3f1ba6cc045841f08ec990d10a8.

Shared-fusion three-seed DEV mean/sample-std/range: D 0.7407873333/0.0216999061/0.040476; P 0.8467100000/0.0335141494/0.065454; Mean 0.7935663333/0.0276077987/0.053141. Frozen full-minus-shared deltas are D -0.0055310000, P -0.0081493333, Mean -0.0066575154. Full exceeds shared on 0/3 seeds and shared aggregate Mean is higher. Exact frozen claim: TASK-AWARE FUSION CONTRIBUTION NOT SUPPORTED. This tests task-conditioned fusion specialization only and does not claim sparse-MTL contribution.

Selected controller diagnostics alpha D/P; progress D/P; grad-norm EMA D/P; grad-cosine EMA; reliability EMA D/P; pseudo scale:
- seed42 0.374351/0.625649; 0.190553/0.092954; 0.382847/0.272144; +0.057377; 0.666639/0.766244; 1.0.
- seed43 0.247172/0.752828; 0.029190/-0.668857; 0.348745/0.325148; -0.017432; 0.667222/0.781396; 1.0.
- seed44 0.301713/0.698287; 0.107800/-0.112563; 0.347162/0.255129; +0.015960; 0.683096/0.780802; 1.0.
All controller values were finite. DEV shared task-row maximum differences for modality weights and fused features were zero.

TRAIN-only D/P shared-fusion gradient diagnostics over audio projection, video projection, task queries, both candidate norms, shared gate, and both fusion norms: seed42 L2 4.110034/1.888330 cosine +0.054242; seed43 3.139425/2.456655 cosine +0.221981; seed44 3.704894/4.017676 cosine +0.089523. All were finite. Sparse MTL, separate disease heads, semantic pseudo cache, direct pseudo supervision, reliability, and RA controller remained present.

DEV-only post-freeze calibration used observed counts D/P 621/312 for every seed. Shared per-seed Brier/ECE-15: seed42 D 0.171636/0.064297 and P 0.070815/0.074090; seed43 D 0.187495/0.105519 and P 0.105897/0.087201; seed44 D 0.174902/0.076025 and P 0.085497/0.108315. Shared three-seed means: D 0.178011/0.081947 and P 0.087403/0.089869. Full three-seed means: D 0.178925/0.070004 and P 0.089948/0.089839. Full-minus-shared calibration deltas: D Brier +0.000914/ECE -0.011943; P Brier +0.002545/ECE -0.000030. Shared DEV audio gate mean/std: seed42 0.464828/0.206305, seed43 0.384742/0.124929, seed44 0.452735/0.132959; three-seed mean 0.434102/0.154731. No recalibration or threshold search occurred.

Same-epoch Test monitoring was read only after each checkpoint was frozen: NONE/SOFT/HARD means seed42 0.801124/0.811002/0.815546, seed43 0.819130/0.826236/0.839551, seed44 0.808215/0.818077/0.831848. Test did not affect selection, calibration, claim, or next-step decision; no Test row entered post-hoc diagnostics.

Final scope checks passed: git diff --check; forbidden src/audio, src/video, src/fusion/data, src/fusion/loss, and src/common/callbacks diffs are empty; only the authorized model file, configs55/56/57, and this ledger differ from origin/main. No Test-driven decision, sparse-MTL claim, significance claim, disease-content claim, missing-label/comorbidity claim, promotion, or demotion claim is made. Stage-5 optimization remains closed; Stage 6 remains active; Stage 7, Text/Description, and Final Test remain locked.


### TASK-006J — sparse-MTL task-isolated firewall

Status: firewall complete; production runs not started. Branch: codex/task-006j, based on manager origin/main 4c5d4f6. Authorized changes are limited to the R4 loss switch, configs58–63, and this ledger.

Implemented backward-compatible loss parameter training_task with values both, depression, and parkinson. Clean origin/main comparison on a deterministic TRAIN batch passed: default both loss value, every model gradient, diagnostics, and controller state were tensor-identical. Single-task routing computes only the active objective; inactive objective terms and reliability updates are excluded, pseudo warm-up remains active, and RA weights cannot affect the single-task total.

All six configs validated. Programmatic equivalence against refs37/38/39 passed with only the authorized run_name, training_task, and three active-task monitor changes; seeds are exactly 42/42/43/43/44/44. Frozen full checkpoint SHA256 values verified: seed42 104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a, seed43 6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e, seed44 b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17. All six DataModule/model/loss configurations instantiated; each model has exactly 295239 trainable parameters and required callback-bearing config metadata.

TRAIN-only isolation firewall passed for depression and Parkinson: inactive logits and auxiliary-output perturbations left total loss and all active gradients unchanged; inactive task-query, candidate-norm, fusion-norm, main-head, and auxiliary-head parameters had zero/None gradients; active task branches, projections, and shared gate had finite gradients; active accepted pseudo supervision at pseudo_scale=1.0 produced finite nonzero gradients; active reliability diagnostics were finite and inactive reliability remained uninitialized; pseudo targets/reliability were detached; controller-weight perturbation left single-task totals unchanged. Pseudo cache SHA remained 17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945 with accepted D/P 376/1801, positive D/P 376/212, negative D/P 0/1589. No optimizer step and no DEV/Test loader iteration occurred.

git diff --check and forbidden-source checks passed. Mandatory firewall commit must be pushed before exactly six production commands in order: D42, P42, D43, P43, D44, P44. No sweep, tuning, retry, extra seed, or post-firewall config change is authorized.


TASK-006J corrective firewall note: the first D-only seed42 invocation reached TRAIN epoch 1 and stopped on an empty depression-active batch before completing evaluation. Inspection found 29 canonical TRAIN and 10 DEV batches without depression-active rows. The loss was corrected in the authorized R4 loss file so single-task empty batches return a zero-connected neutral loss without computing inactive terms or gradients. Corrective TRAIN-only checks passed for both active-task directions, including empty-batch neutral loss and deterministic clean-origin default compatibility. The failed invocation is not counted as a completed production run; the required production sequence restarts with corrected firewall commit before the six completed runs.


### TASK-006J — final sparse-MTL joint-training audit evidence

Status: complete for the corrected six-run sequence; exact frozen claim: SPARSE MTL JOINT-TRAINING CONTRIBUTION NOT SUPPORTED. Branch codex/task-006j. Firewall commits b9e45c7 and corrective 63c3276 were pushed before the six completed runs. One initial D-only seed42 invocation at 23:30 reached TRAIN epoch 1 and stopped on empty active-task handling before evaluation; it is recorded as failed diagnostic setup, not counted among the six completed production runs. The corrected six commands then completed exactly in order: D42, P42, D43, P43, D44, P44. No sweep, tuning, metric retry, extra seed, or post-correction config change occurred.

Completed MLflow identities:
- D42 ed70407bb31c45a689ffb2b1a5f100ed; run stage6_d_only_trial012_seed42_2026-09-28_23-35_wsm_av_r3_disease_query_model_8851ab7b.
- P42 ff9a31a20bb748a59b160144788405fa; run stage6_p_only_trial012_seed42_2026-09-28_23-39_wsm_av_r3_disease_query_model_fe1b59e5.
- D43 c621fd3f7db0428987b5d7f609b59118; run stage6_d_only_trial012_seed43_2026-09-28_23-49_wsm_av_r3_disease_query_model_21fdaad0.
- P43 3054a688776d47eba89d598cfd2c0372; run stage6_p_only_trial012_seed43_2026-09-28_23-54_wsm_av_r3_disease_query_model_820a113f.
- D44 457af2bf694c4c3bb539a1319a18e100; run stage6_d_only_trial012_seed44_2026-09-29_00-05_wsm_av_r3_disease_query_model_4c3989d5.
- P44 c10343cad7ce4ae3a8ebd9344a170581; run stage6_p_only_trial012_seed44_2026-09-29_00-10_wsm_av_r3_disease_query_model_eda84d7e.

DEV active-task selection by the required active monitor:
- seed42 D-only epoch5, active D UAR/MF1/Score 0.722456/0.722335/0.722396; inactive P Score 0.541443; checkpoint SHA256 55f27fef743ade6684b56159ee60d410178c56ab73e741001613d535b920fbc9.
- seed42 P-only epoch21, active P UAR/MF1/Score 0.894893/0.901431/0.898162; inactive D Score 0.724889; checkpoint SHA256 efca8c5743322c4a54a29e60af4578d3a9af7b4c9518cee32b48764db9061b4d.
- seed43 D-only epoch6, active D UAR/MF1/Score 0.723343/0.721865/0.722604; inactive P Score 0.590842; checkpoint SHA256 083ba177a21e7d9e91ebcf753b0efecc8f2430ef9f74a839a3e4cd892cad89d9.
- seed43 P-only epoch28, active P UAR/MF1/Score 0.887992/0.905687/0.896839; inactive D Score 0.633823; checkpoint SHA256 7e050e6a454602888adf4054b82a06b07b6234e942a57c481b8a6908be3dd167.
- seed44 D-only epoch9, active D UAR/MF1/Score 0.709057/0.708018/0.708538; inactive P Score 0.583242; checkpoint SHA256 d13de374ef118e7393179504796d1b547f9b83d8836f39080558c5697a961ed7.
- seed44 P-only epoch12, active P UAR/MF1/Score 0.840235/0.858735/0.849485; inactive D Score 0.726031; checkpoint SHA256 e1bd25a250b7aec8b8a7bfefa390a49e9e7a1c321cda1b958a7ab8e90cfbd1e4.

Task-isolated active scores: single_D 0.722396/0.722604/0.708538; single_P 0.898162/0.896839/0.849485; paired Mean 0.810279/0.809722/0.779012 for seeds42/43/44. Three-seed single_D mean/std/range 0.717846/0.008062/0.014066; single_P 0.881495/0.027730/0.048677; paired Mean 0.799671/0.017894/0.031268. Full-minus-single same-seed D/P/paired-Mean deltas: seed42 +0.036629/-0.018903/+0.008863; seed43 -0.004077/-0.092476/-0.048276; seed44 +0.019679/-0.017425/+0.001128. Aggregate deltas: D +0.017410, P -0.042935, paired Mean -0.012762. Frozen gate fails on aggregate Mean and aggregate P, so the exact claim is SPARSE MTL JOINT-TRAINING CONTRIBUTION NOT SUPPORTED.

DEV-only active calibration used observed counts D=621 and P=312 per seed. Per-seed active Brier/ECE-15: D42 0.195620/0.117496; P42 0.065901/0.073082; D43 0.215300/0.156690; P43 0.070935/0.092257; D44 0.216406/0.156601; P44 0.080944/0.093646. Three-seed active means: D Brier/ECE 0.209109/0.143595; P 0.072593/0.086329. Matching full-reference means are D 0.178925/0.070004 and P 0.089948/0.089839; full-minus-single calibration deltas: D Brier -0.030183/ECE -0.073591; P Brier +0.017355/ECE +0.003510.

Active DEV audio-gate mean/std: D42 0.466700/0.146920; P42 0.438374/0.196786; D43 0.441651/0.154020; P43 0.297180/0.191392; D44 0.414473/0.133582; P44 0.502733/0.186750. Three-seed gate means/std: D 0.440941/0.026121 across seeds; P 0.412762/0.105142. Deterministic TRAIN active shared-gradient L2: D42 5.158658, P42 2.908360, D43 7.441620, P43 1.993566, D44 6.581375, P44 2.909846. Loss active grad-norm diagnostics were finite: D42 0.204165, P42 0.142843, D43 0.272388, P43 0.144416, D44 0.279896, P44 0.165217. RA weights were instrumentation only and are not interpreted as task balancing because each run returned one task objective.

Pseudo coverage remained unchanged: accepted D/P 376/1801; accepted classes D positive/negative 376/0 and P 212/1589. Each task-isolated model retained 295239 trainable parameters; the two-model pair is explicitly not an equal-total-deployment-size comparator.

Same-epoch Test monitoring was read only after active-task checkpoint freeze: D42 NONE/SOFT/HARD 0.465976/0.593093/0.594806; P42 0.576740/0.707599/0.713087; D43 0.535903/0.634482/0.640188; P43 0.495735/0.651528/0.631796; D44 0.395258/0.578021/0.580933; P44 0.519774/0.707595/0.714565. Test did not affect selection, claim, or follow-up, and no Test rows entered post-hoc diagnostics.

Final scope checks passed: git diff --check; model, DataModule, callbacks, audio, and video diffs against origin/main are empty; only the authorized R4 loss file, configs58–63, and this ledger differ. No sparse-MTL significance, pseudo-label correctness, missing-label/comorbidity recovery, promotion, or demotion claim is made. Stage 6 remains active pending manager review; Stage 5 remains closed; Stage 7, Text/Description, and Final Test remain locked.

### TASK-006K — Stage-6 claim ledger and completeness matrix

Status: complete as a documentation-only synthesis; Stage 6 remains active pending manager review. Branch: codex/task-006k. Added [docs/STAGE6_CLAIM_LEDGER_EN.md](STAGE6_CLAIM_LEDGER_EN.md), mapping all Stage-6 plan items 1–10 and the required DEV, calibration, pseudo-coverage/class-balance, gradient/controller, negative-transfer, and gate diagnostics to accepted evidence.

Exact completeness result: `CORE STAGE-6 A+V EVIDENCE MATRIX COMPLETE`.

Supported claims preserved exactly:
- `DIRECT PSEUDO-SUPERVISION CONTRIBUTION SUPPORTED`
- `UNCERTAINTY/RELIABILITY CONTRIBUTION SUPPORTED`
- `ONLINE AUDIO INPUT MODALITY CONTRIBUTION SUPPORTED`
- `SAMPLE-SPECIFIC PSEUDO ALIGNMENT CLAIM SUPPORTED BY THIS NEGATIVE CONTROL`

Unsupported claims preserved exactly:
- `SPARSE MTL JOINT-TRAINING CONTRIBUTION NOT SUPPORTED`
- `TASK-AWARE FUSION CONTRIBUTION NOT SUPPORTED`
- `SEMANTIC-EVIDENCE CONTRIBUTION VIA DEPRESSION PSEUDO ACCEPTANCE NOT SUPPORTED`
- `RA-STCH BALANCING CONTRIBUTION OVER EQUAL NOT SUPPORTED`
- `RA-STCH ADVANTAGE OVER SIMPLER BALANCING NOT SUPPORTED`
- `VIDEO MODALITY CONTRIBUTION NOT SUPPORTED`
- `FUSION AMPLIFICATION OF CORPUS DECODABILITY NOT SUPPORTED`

Diagnostic-only findings preserved: `STRONG CORPUS-ID DECODABILITY FLAG NOT TRIGGERED`; no causal corpus-shortcut claim. TASK-006J is recorded accurately as 7 total invocations, with the first D42 failure before DEV/Test/checkpoint selection, corrective firewall, and six accepted completed runs in order D42/P42/D43/P43/D44/P44; no metric-driven retry or tuning.

Candidate implications remain descriptive: full R4, matched audio, shared-fusion, no-semantic, paired task-isolated, no-video, and no-audio reference facts are recorded without promotion, demotion, or final-method selection. No run, source, config, script, cache, or Test-based decision occurred in TASK-006K. Stage 3 Text/Description remains deferred pending manager activation; Stage 7 and Final Test remain locked. Manager review is required before any next task.

### TASK-003A pre-execution firewall

Branch: `codex/task-003a`. Implemented only [scripts/text/audit_transcript_sources.py](../scripts/text/audit_transcript_sources.py). The audit is structural and metadata-only: it derives only canonical video-level transcript paths, parses canonical JSON segment IDs, never stores transcript text, refuses an existing external output, and sets Test lexical inspection to false.

Firewall checks passed before the full dataset audit: `--help`; synthetic plain-text, SRT-like, WebVTT, timestamped-line, JSON-like, invalid-UTF8, and empty-file parser cases; existing-output overwrite refusal; raw-text output-schema guard; `py_compile`; `git diff --check`; and `git diff origin/main -- src configs pyproject.toml` empty. No source, config, dependency, training, model, cache, or Test inspection occurred.

The firewall commit must precede the single full audit invocation. The external report path is `/media/maxim/Programs/Features/WSM/stage3_text_audit/transcript_source_audit_v1.json`; the full audit will fail rather than overwrite it.

### TASK-003A — transcript source, alignment, and language audit

Status: complete audit-only. Branch: `codex/task-003a`. Added [docs/STAGE3_TRANSCRIPT_AUDIT_EN.md](STAGE3_TRANSCRIPT_AUDIT_EN.md). External report: `/media/maxim/Programs/Features/WSM/stage3_text_audit/transcript_source_audit_v1.json`. Report SHA256: `2dc2064aca131da0e80c103df1b8dd7a19d9735aa8c183e149392f7aafd1206b`.

The pushed firewall commit was `35a50c8`; it passed help, synthetic parser cases, overwrite refusal, raw-text schema guard, syntax, diff check, and empty `src`/config/dependency diff. The first full invocation stopped before output creation on a script field-name defect; the narrow correction was pushed as `f54649f`, and the corrected successful audit completed without changing data or conclusions.

Canonical coverage: 8,622 segments and 755 unique corpus/split/video groups. Non-empty transcript coverage was depression TRAIN 287/288 videos and 3,587/3,660 segments; depression DEV 56/56 and 621/621; depression TEST 55/55 and 827/827; Parkinson TRAIN 266/266 and 2,665/2,665; Parkinson DEV 44/44 and 312/312; Parkinson TEST 46/46 and 537/537. The report found 754 plain-text files and 1 missing file, with no decode errors.

No source CSV had explicit segment timing columns; transcript timestamp parsing found zero spans/lines. Exact conclusion: `SEGMENT TEXT ALIGNMENT NOT ESTABLISHED`; required granularity conclusion: `T1 TEXT UNIT SHOULD BE VIDEO-LEVEL TRANSCRIPT`. TRAIN/DEV Unicode audit found only Latin letters and digits. The 24-video deterministic manual sample had coarse `English_or_other_Latin` labels, no mixed-language flags, and no committed snippets. Duplicate-hash audit found no cross-split duplicates and three within-TRAIN duplicate hashes, explicitly listed in the audit document.

Exact readiness result: `T1 TRANSCRIPT DATA CONTRACT READY`. No encoder, model, dependency, config, source package, training, Test prediction/metric inspection, or Test lexical inspection occurred. T1 implementation remains unauthorized until manager review.

### TASK-003A corrective firewall — TEST language isolation


Manager review rejected the original report because the script computed `coarse_language` and `mixed_language` for TEST records. No Test predictions or performance metrics were inspected, but the implementation exceeded the structural-only Test contract.

On the same branch `codex/task-003a`, the script was corrected before report replacement. TEST records now receive only structural fields plus `language_audit_performed: false`; the executable main-path guard rejects any TEST language-analysis fields and requires `test_language_audit_performed = false`, `test_lexical_inspection_performed = false`, and `test_predictions_or_metrics_inspected = false`. Readiness now uses explicit path, TRAIN/DEV decode, TRAIN/DEV language evidence, duplicate-audit, and alignment-conclusion predicates; missing transcripts remain availability gaps.

Corrective firewall commit `b9a2d85` was pushed before replacing the rejected report. Checks passed: `--help`; relevant synthetic parser tests; synthetic TEST-record/schema assertion; overwrite refusal; raw-text schema guard; `git diff --check`; and empty `git diff origin/main -- src configs pyproject.toml`. The old report SHA `2dc2064aca131da0e80c103df1b8dd7a19d9735aa8c183e149392f7aafd1206b` is superseded. The corrected report SHA is `4254d81281522015d0633d6534e1616e925a2bf0130f67a37b541a2aacc2ed78`. The earlier field-name runtime defect remains recorded above.


### TASK-003B corrective firewall

The initial cache invocation stopped before creating the external cache root because this Transformers XLM-R tokenizer exposes neither public `build_inputs_with_special_tokens` nor `prepare_for_model`. This was a pure runtime API defect; no cache artifact or scientific conclusion was produced. The extractor was corrected to use the frozen tokenizer's two normal single-sequence special-token IDs around each pre-chunked content-token list, preserving the <=512 input and non-special pooling contract.

Corrective firewall checks passed: corrected long-transcript multi-chunk extraction produced finite float32 [2,768] features; encoder remained eval/no-grad with all parameters frozen; `git diff --check` passed; forbidden source scopes remained empty. The corrected replacement cache build must follow this pushed corrective firewall.


### TASK-003B final T1 cache evidence

Status: complete implementation/cache-only; no production training. Final branch remains `codex/task-003b`. Corrected cache root: `/media/maxim/Programs/Features/WSM/text_t1_xlmr_v1`. Cache index SHA256: `4f276b60e8e423a1eab2d69a02782daa274fb183ae028a525f16df3adcf83ea8`.

Frozen encoder: `FacebookAI/xlm-roberta-base`; requested revision `e73636d`; resolved commit `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`; model.safetensors SHA256 `6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb`. Extraction used full video-level transcripts, no truncation, non-overlapping chunks of at most 510 content tokens, normal two special tokens, final hidden-layer mean pooling over valid non-special tokens, float32 ordered [C,768] features, eval/no-grad, and no labels/prompts/token IDs/raw text in artifacts or index.

Cache facts: 755 canonical entries; 754 available plain-text transcripts; one unavailable TRAIN depression transcript; all 754 artifacts finite float32 [C,768] with C>=1; transcript SHA256 values match the accepted TASK-003A report; artifact hashes and index hash verified; no raw text or token IDs persisted. Canonical TRAIN units are exactly depression 287, Parkinson 266, total 553, with no duplicate corpus/video identities. Evaluation protocol streams are canonical segment rows: DEV 933, TEST_NONE 1364, TEST_SOFT 1208, TEST_HARD 1014; each broadcasts its parent video feature sequence and never multiplies TRAIN units.

Registrations and config construction passed for `wsm_text_t1_datamodule` and `wsm_text_t1_chunk_transformer`; all three seed-clone configs validated and instantiated with seeds 42/43/44 and identical semantics otherwise. Fixed T1 model trainable parameter count is 447746. TRAIN forward/loss/backward smoke passed with finite loss 0.7279856205, finite nonzero gradients, and masked unknown-target invariance. Shape-only loads passed for DEV and all three Test streams; no performance metrics were computed or inspected.

Two pure runtime corrections were handled transparently after pushed firewalls: tokenizer special-token API correction commit `305a3e4`, then the DataModule per-segment protocol/NaN-mask correction in the final evidence commit. The first cache attempt produced no artifacts before the tokenizer correction; the incomplete empty cache root was removed only for the authorized replacement build. Stage 3 remains active; Stage 7 and Final Test remain locked.


### TASK-003C pre-production firewall

Branch: codex/task-003c. Frozen cache identity verified: index SHA256 4f276b60e8e423a1eab2d69a02782daa274fb183ae028a525f16df3adcf83ea8; 755 entries; all 754 referenced artifacts exist; encoder/revision/resolved commit/model SHA match the accepted TASK-003B identity. All three frozen configs validate, instantiate, and differ only by seed/run_name. The fixed model has 447746 trainable parameters.

TRAIN membership is exactly D287/P266/total553 unique video units. Evaluation membership is DEV933/TEST_NONE1364/TEST_SOFT1208/TEST_HARD1014. Shape-only TRAIN and DEV batches passed. No DEV/Test performance metric was computed or inspected. git diff --check and forbidden src, config, script, and dependency diffs are clean. No production run has started before this firewall commit.


### TASK-003C  final T1 three-seed text baseline evidence

Status: complete. Exact result: `T1 THREE-SEED TEXT BASELINE COMPLETE`. Branch: `codex/task-003c`. Only the three frozen production commands were executed, exactly once and in order, after firewall commit `6af94e4` was pushed: seed42, seed43, seed44. No sweep, tuning, retry, extra seed, cache/config/source/dependency change, or Test-driven decision occurred.

Production commands:
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/text/00_t1_xlmr_seed42.yaml`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/text/01_t1_xlmr_seed43.yaml`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train --config-path configs/wsm_mm_pd_dep_v1/text/02_t1_xlmr_seed44.yaml`

Frozen DEV-only checkpoint selection by maximum `dev/mean_score`:
- seed42: epoch16, D/P/Mean `0.721153/0.899381/0.810267`; checkpoint `logs/wsm_mm_pd_dep_v1/stage3_t1_xlmr_video_text_seed42_2026-09-29_12-15_wsm_text_t1_chunk_transformer_cdf61d2c/checkpoints/epoch=16_dev_mean_score=0.8103.pt`; SHA256 `151b6a4ede993f25d168f669711a0a35545958b9ed0a32772e58a11b3db13536`; MLflow run `4f9638ac06c648a59e626eac066679f0`.
- seed43: epoch7, D/P/Mean `0.562911/0.825768/0.694339`; checkpoint `logs/wsm_mm_pd_dep_v1/stage3_t1_xlmr_video_text_seed43_2026-09-29_12-17_wsm_text_t1_chunk_transformer_b850ebcc/checkpoints/epoch=7_dev_mean_score=0.6943.pt`; SHA256 `25f89cdc487c9094c4f6511d760c67c133c3891b302cf75ccf96294f44469f2e`; MLflow run `f52995473b374d8eaf7e589d0262165f`.
- seed44: epoch2, D/P/Mean `0.596355/0.809025/0.702690`; checkpoint `logs/wsm_mm_pd_dep_v1/stage3_t1_xlmr_video_text_seed44_2026-09-29_12-18_wsm_text_t1_chunk_transformer_2523c8b6/checkpoints/epoch=2_dev_mean_score=0.7027.pt`; SHA256 `fbd337609fe9c38eb09c3614ffd0bc2ee56801b271cc420ec692610eaeb1ab1b`; MLflow run `7c2b84a965984daba286707fcf55169e`.

Three-seed primary DEV D/P/Mean mean, sample std, range: mean `0.626806/0.844725/0.735765`; std `0.083400/0.048068/0.064655`; range `0.158242/0.090356/0.115928`. Per-seed selected DEV D UAR/MF1/Score and P UAR/MF1/Score: seed42 `0.721802/0.720505/0.721153`, `0.904072/0.894689/0.899381`; seed43 `0.571102/0.554720/0.562911`, `0.823119/0.828417/0.825768`; seed44 `0.597712/0.594997/0.596355`, `0.815390/0.802661/0.809025`.

DEV-only secondary audit deduplicated canonical `(corpus, video_id)`: 100 unique videos per seed. Segment-minus-unique D/P/Mean deltas: seed42 `+0.023759/+0.106582/+0.065170`; seed43 `+0.009376/+0.077403/+0.043390`; seed44 `+0.054893/+0.086415/+0.070654`. Unique-video D/P/Mean: seed42 `0.697394/0.792799/0.745097`; seed43 `0.553534/0.748365/0.650950`; seed44 `0.541461/0.722611/0.632036`. Unique metrics were not used for selection.

DEV-only calibration used observed D=621 and P=312 per seed, with no recalibration or threshold search. Per-seed Brier/ECE-15: seed42 D `0.179343/0.198466`, P `0.108042/0.208747`; seed43 D `0.247784/0.234108`, P `0.136693/0.169739`; seed44 D `0.244794/0.177437`, P `0.211774/0.253974`. Three-seed means: D `0.223973/0.203336`, P `0.152170/0.210820`.

Same-epoch TEST_NONE/SOFT/HARD monitoring was inspected only after each DEV checkpoint freeze and did not affect selection or claims. D/P/Mean: seed42 NONE `0.831266/0.722989/0.777127`, SOFT `0.815629/0.712308/0.763968`, HARD `0.812956/0.699957/0.756456`; seed43 NONE `0.729547/0.679297/0.704422`, SOFT `0.723828/0.669494/0.696661`, HARD `0.720002/0.661341/0.690671`; seed44 NONE `0.702429/0.743890/0.723159`, SOFT `0.709882/0.750566/0.730224`, HARD `0.707727/0.699957/0.703842`. No Test rows entered post-hoc diagnostics.

Context only, not an additive or promotion claim: matched-audio three-seed D/P/Mean `0.7303377965/0.8162961212/0.7733169588`; full R4 `0.7352563333/0.8385606667/0.7869088179`; accepted V2 video single-run `0.6201013364/0.7930427585/0.7065720475`. T1 is a standalone text baseline only; no three-seed significance claim.

Final scope checks passed: only `docs/PROGRESS_EN.md` differs from `origin/main`; `src`, configs, scripts, cache/dependency files remain unchanged; `src/audio` stayed unchanged; `git diff --check` passed; main/master was not modified. Stage 3 remains active; Stage 6 is complete; Stage 7 and Final Test remain locked. Manager review is required before any next task.


### TASK-003D  T2 observable-description generator/source/prompt preflight

Status: complete preflight with frozen contract blocked. Exact result: `T2 DESCRIPTION GENERATION CONTRACT BLOCKED`. Branch: `codex/task-003d`.

Allowed implementation: [scripts/description/preflight_qwen3vl_observable.py](../scripts/description/preflight_qwen3vl_observable.py). Firewall commit `f823708` was pushed before VLM loading. The initial frozen invocation reached generation and failed before writing a report with CUDA OOM under default video processing. Transparent corrective firewall commits `72de483` and `aad83a9` were pushed: deterministic uniform `num_frames=8`, with `fps=null`, was frozen and recorded; model and prompt were unchanged. The final replacement invocation completed 24 TRAIN/DEV generations plus two fresh repeat calls for the first sample. No TEST content, training, performance metric, or T2 cache was generated.

Frozen model: `Qwen/Qwen3-VL-8B-Instruct`; requested/resolved revision `1dd1e02d981403da25ed73d43430e4ef598cb94b`; processor/model `Qwen3VLProcessor`/`Qwen3VLForConditionalGeneration`; safetensors index SHA256 `520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070`. Shard SHA256 values and package/device/dtype/memory evidence are in [docs/STAGE3_T2_DESCRIPTION_PREFLIGHT_EN.md](STAGE3_T2_DESCRIPTION_PREFLIGHT_EN.md). Prompt SHA256 `118b93ba090af2b4ce755d9da1476e073cab0624760e9aa4c347faad8750ee4d`; generation was deterministic (`do_sample=false`, 120 new tokens, one sequence, no beams/tuning). Peak allocated/reserved GPU memory was `14653126144/15919480832` bytes.

Structural source audit covered 8,622 canonical `.mp4` rows from `build_manifest` identity mapping: all 8,622 existed and decoded/opened successfully; depression TRAIN/DEV/TEST `3660/621/827`, Parkinson TRAIN/DEV/TEST `2665/312/537`; zero failures. Deterministic sample was exactly 24 rows, six each from depression TRAIN/DEV and Parkinson TRAIN/DEV, with IDs recorded in the audit document. TEST generation/manual inspection remained false.

Manual audit aggregate over 24 outputs: nonempty 24/24; visually grounded/mostly observable 24/24; major unsupported/hallucinated 0; diagnosis/disease/health-state inference 0; demographic/identity inference 4; causal/medication inference 0; dataset/task/label leakage 0; concise enough for semantic encoding 18/24. The four demographic failures were explicit gender terms; six concision failures were overly long, repetitive, or truncated. First-sample fresh-call determinism passed exactly: hash A/B both `bcad9ba7c4a55880b82e0106afcb93ebc3476a902683b9ea60725468b5aa63da`.

External report: `/media/maxim/Programs/Features/WSM/stage3_t2_preflight/qwen3vl8b_observable_preflight_v1.json`; SHA256 `767d687e2f857938fa882555250d7a4be4159556e49537377967d82c36407930`. Failed frozen gate conditions: 5 (zero demographic/identity inference) and 9 (at least 22/24 concise; observed 18/24). The human-readable evidence is [docs/STAGE3_T2_DESCRIPTION_PREFLIGHT_EN.md](STAGE3_T2_DESCRIPTION_PREFLIGHT_EN.md); it contains no verbatim generated descriptions. No model/prompt change, dependency change, source/config change, training, cache generation, TEST inspection, or performance claim was made.


### TASK-003E  Stage-3 text/description synthesis and closure dossier

Status: documentation-only synthesis complete; Stage 3 remains ACTIVE pending manager closure. Branch: `codex/task-003e`. Ledger: [docs/STAGE3_CLAIM_LEDGER_EN.md](STAGE3_CLAIM_LEDGER_EN.md).

Exact completeness result: `STAGE-3 TEXT/DESCRIPTION EVIDENCE MATRIX COMPLETE`.
Exact family-budget result: `STAGE-3 TWO-FAMILY BUDGET EXHAUSTED`.

TASK-003A preserved the accepted transcript conclusions `SEGMENT TEXT ALIGNMENT NOT ESTABLISHED`, `T1 TEXT UNIT SHOULD BE VIDEO-LEVEL TRANSCRIPT`, and `T1 TRANSCRIPT DATA CONTRACT READY`; authoritative report SHA256 `4254d81281522015d0633d6534e1616e925a2bf0130f67a37b541a2aacc2ed78`. TASK-003B preserved the frozen XLM-R identity/cache and D287/P266/total553 TRAIN units with 447746 downstream trainable parameters. TASK-003C preserved `T1 THREE-SEED TEXT BASELINE COMPLETE` with aggregate DEV D/P/Mean `0.6268063333/0.8447246667/0.7357653333`, sample std `0.0834002123/0.0480683689/0.0646553057`, and standalone-text-only interpretation.

TASK-003D preserved `T2 DESCRIPTION GENERATION CONTRACT BLOCKED`: Qwen3-VL revision `1dd1e02d981403da25ed73d43430e4ef598cb94b`, prompt SHA256 `118b93ba090af2b4ce755d9da1476e073cab0624760e9aa4c347faad8750ee4d`, tested deterministic uniform `num_frames=8`, report SHA256 `767d687e2f857938fa882555250d7a4be4159556e49537377967d82c36407930`. Manual audit was nonempty 24/24, visually grounded 24/24, unsupported 0/24, diagnosis/health inference 0/24, demographic/identity inference 4/24, causal/medication 0/24, leakage 0/24, concise 18/24. The T2 safety/concision gate failed; no T2 cache, implementation, training, or performance evidence exists.

The descriptive pre-Stage-7 inventory records matched temporal audio, full R4 trial012, equal-parameter shared fusion, no-semantic depression, paired task-isolated, T1 text-only, and blocked/ineligible T2. It does not rank, score, tier, promote, demote, or select a Stage-7 set. Paired task-isolated uses two models and is not an equal-total-deployment-size comparator. T1 is contextual standalone evidence, not additive A+V evidence. No significance claim is made.

No compute, inference, source/config/script/dependency/cache change, Test content/metric analysis, or Stage-7 activation occurred in TASK-003E. Stage 3 remains pending manager closure; Stage 7 and Final Test remain locked. Main/master remain untouched.


### TASK-007A — Stage-7 final candidate/config/seed/statistical freeze

Status: preflight complete; production training and Final Test remain unauthorized. Branch: `codex/task-007a`. Dossier: [docs/STAGE7_FINAL_FREEZE_EN.md](STAGE7_FINAL_FREEZE_EN.md).

The manager-frozen compared set is exactly frozen temporal audio, full optimized R4 trial012, and equal-parameter shared-fusion trial012. Frozen seeds are `42,43,44,45,46`; accepted seeds42–44 were audited from committed evidence and were not rerun. Exactly six seed45/46 clones were created: audio configs03/04, full R4 configs40/41, and shared-fusion configs42/43 under `fusion/`. YAML equivalence passed for every new/reference pair after removing only `seed` and `experiment_info.params.run_name`; all six pass `chimera-ml validate-config`.

All six DataModules, models, losses, and callbacks instantiated without training. Audio trainable parameters were `3033416`; full R4 and shared fusion each retained exactly `295239`. Shape-only TRAIN and DEV batches passed for the audio and fusion families. No optimizer step, metric calculation, inference, or Test loader iteration occurred. Every new config retains max-mode `dev/mean_score` checkpoint and early-stopping selection.

Frozen accepted checkpoint identities and shared-fusion run identities are recorded in the dossier. The semantic pseudo cache remained unchanged at SHA256 `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`, TRAIN rows `6325`, accepted D/P `376/1801`, and classes D `376/0`, P `212/1589`; canonical audio features remain `/media/maxim/Databases/WSM_NEW/features`, dimension 768.

The dossier freezes five-seed mean/sample-std/min/max/range and Student-t 95% CI (`2.7764451051977987`, df=4), same-seed paired deltas for all three method pairs, and DEV-only Brier/ECE-15 summaries. Final Test is explicitly unauthorized: no new or historical Test values are inspected; all fifteen selected checkpoints must be frozen before a separate final-Test task. Next production order is exactly audio45, R4-45, shared45, audio46, R4-46, shared46, with no retry, extra seed, or post-firewall config change.

Scope checks passed: `git diff --check`; forbidden source/script/dependency/cache scopes are unchanged; no production command was run. Main/master remain untouched.


### TASK-007B — final DEV-only production firewall

Status: firewall complete; no production run has started. Branch: `codex/task-007b`, based on manager `origin/main` `918017894165a95d55f24b599a077b542baae2ed`. The frozen Stage-7 dossier is unchanged. All six seed45/46 configs are semantically identical to their frozen references except seed/run_name, all validate, and all retain max-mode `dev/mean_score` checkpoint/early-stopping selection.

No-training instantiation passed for the audio, full R4, and shared-fusion families with registered DataModule context: audio trainable parameters `3033416`; full R4 `295239`; shared fusion `295239`; required callbacks instantiated. The semantic pseudo cache SHA256 remains `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`; frozen feature/cache roots and audio feature contract remain unchanged. No Test loader was iterated and no metric was computed.

Firewall scope checks passed: `git diff --check`; `src`, scripts, configs relative to origin/main, and the frozen dossier are clean before production. This firewall evidence is committed and pushed before any training. The only authorized next actions are exactly six commands in order: audio45, R4-45, shared45, audio46, R4-46, shared46. No retry, sweep, extra seed, config change, or metric-driven stopping is authorized.


### TASK-007B — final DEV-only production evidence

Status: incomplete. Exact result: `STAGE-7 FIVE-SEED DEV CHECKPOINT FREEZE INCOMPLETE`. Branch: `codex/task-007b`. Firewall commit `e5043f6` was pushed before production.

Exactly six production invocations completed once and in order: audio45, full R4-45, shared fusion-45, audio46, full R4-46, shared fusion-46. New DEV-selected checkpoint paths, SHAs, MLflow identities, epochs, DEV D/P/Mean scores, and calibration values are recorded in [docs/STAGE7_FINAL_DEV_EVIDENCE_EN.md](STAGE7_FINAL_DEV_EVIDENCE_EN.md). Selection used only `dev/mean_score`; Test inspection is explicitly `false`.

The exact blocking evidence is the accepted audio seed42 checkpoint SHA `0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2`: no full local checkpoint path or historical run identity is available in this workspace, so its SHA cannot be reverified and its DEV re-evaluation/calibration cannot be completed. No replacement run is authorized. The five-seed DEV score summaries and descriptive paired Mean deltas are recorded, but the all-15 checkpoint freeze and complete five-seed calibration gate remain incomplete.

No Test loader was called, no Test metric key was queried, and no raw production output or Test log was opened. A historical PROGRESS search incidentally displayed pre-existing Test lines; those values were not used or transcribed into evidence. No retry, tuning, extra seed, config/source/cache/dependency change occurred. Main/master remain untouched; Stage 7 remains active and Final Test remains locked.

### TASK-007B — corrective audio42 recovery

Status remains incomplete. Exact result: `STAGE-7 FIVE-SEED DEV CHECKPOINT FREEZE INCOMPLETE`. Branch: `codex/task-007b`. Original firewall commit: `e5043f62d6e836c2074fa16b2e5e7b03b813905a`; prior incomplete evidence commit: `54c9547331e90658332ad905bf6c1c9a9cb62457`.

The accepted historical audio42 checkpoint was recovered read-only at `logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt`, run identity `multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77`, and SHA256 `0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2`. The existing `FrozenAudioTemporalAdapter` strict-loaded epoch 4, with frozen parameters and eval mode; no source or `src/audio` change occurred.

The corrected canonical DEV-only reproduction did not agree with accepted TASK-004H evidence within `0.0005`: reproduced D/P/Mean `0.5047152194/0.4494219653/0.4367154383` versus accepted `0.7480392157/0.8277353635/0.7878268765`. Attempted observed-only audio42 calibration was D/P counts `621/312`, Brier `0.4688533750/0.3203929081`, ECE-15 `0.4694130072/0.3149863942`; these are not finalized because the score reproduction failed. The all-15 checkpoint freeze and complete five-seed calibration summaries remain blocked.

All six seed45/46 production runs were NOT rerun; no training occurred in this correction; no Test loader was constructed/iterated and no Test metric query occurred. The prior broad historical PROGRESS search incidentally surfaced pre-existing Test lines, but `FINAL TEST VALUES NOT USED FOR TASK-007B SELECTION OR RECOVERY`. No config/source/cache/finalist/stat-plan change occurred. Stage 7 remains active and Final Test remains locked.


### TASK-007B — corrective evaluation-path diagnostic

Status complete. Exact result: STAGE-7 FIVE-SEED DEV CHECKPOINT FREEZE COMPLETE. Branch: codex/task-007b. Prior recovery commit: b892ff0eef4e932e5214dfb818aecc2a502042b8.

Diagnostic A reconstructed the exact TASK-004H AV path using WSMAVFusionDataModule.val_dataset only across 933 DEV rows. Raw base logits matched the legacy two-class difference with max absolute difference 0.0; argmax mismatch count was 0; raw-logit helper metrics and independent legacy-argmax metrics matched within 1e-12, reproducing D/P/Mean 0.7480392157/0.8277353635/0.7878268765. Diagnostic B native-vs-adapter logit difference was 0.0 for both tasks and reproduced the same metrics. Diagnostic C matched all 12 deterministic AV/native rows: task IDs, labels, observed targets, feature paths/SHA256, sanitized tensor shapes/contents, and masks. Diagnostic D seed43/44 controls reproduced accepted D/P/Mean 0.7086990/0.8117710/0.7602350 and 0.7343964/0.8093819/0.7718891 within 0.0005.

Correct audio42 DEV calibration from raw logits: counts D/P 621/312; Brier 0.2117752880/0.1241006106; ECE-15 0.1664041658/0.1066767589. All 15 checkpoint identities are frozen (14 previously verified plus recovered audio42). Five-seed calibration summaries were finalized with the frozen Student-t multiplier 2.7764451051977987; full values are in the Stage-7 evidence dossier. Runtime: Python 3.12.3, torch 2.10.0+cu128, NumPy 2.5.1, CPU float32, CUDA 12.8/cuDNN 91002.

The prior 0.4367 attempt is superseded as an evaluation-path failure; its calibration values are discarded. Exact narrower mechanism was not claimed because the prior temporary code was unavailable. No training, production rerun, source/config/cache/environment change, or Test loader/Test metric query occurred in this correction. The prior procedural exception remains documented: FINAL TEST VALUES NOT USED FOR TASK-007B SELECTION OR RECOVERY. Stage 7 remains active pending manager review; Final Test remains locked.


### TASK-007C — DEV firewall blocked before Final Test

Status incomplete. Exact result: STAGE-7 FINAL TEST EVALUATION INCOMPLETE. Branch: codex/task-007c.

The evaluator scripts/common/evaluate_stage7_final_test.py was implemented with the frozen 15-entry ledger, checkpoint/config/pseudo-cache SHA verification, raw-logit metrics, historical audio42 margin/argmax checks, sparse-mask semantics, synthetic arithmetic checks, DEV-only preflight, overwrite refusal, and atomic output. Static checks, --help, synthetic checks, 15-entry count, and overwrite refusal passed.

The mandatory all-15 DEV preflight was attempted using the exact semantic RAMPS DataModule. It stopped before any Test loader or Test dataset iteration at shared-fusion seed43: checkpoint SHA ec746e519e551d69d59954dffff0a73968fb4ab555d43020b7722e635cc477ee, config configs/wsm_mm_pd_dep_v1/ablations/56_shared_fusion_seed43.yaml, frozen DEV D/P/Mean 0.724496/0.818152/0.771324, reproduced 0.7229314130/0.8181521375/0.7705417752; D/Mean absolute mismatches 0.0015645870/0.0007822248, exceeding 0.0005. The configured shared mode and a narrow task-aware diagnostic both failed the frozen ledger, so no source/config/checkpoint/cache correction is authorized.

Final Test invocation count: 0. External report was not created. No training, Test loader iteration, Test metrics, recalibration, threshold fitting, role revision, or retry occurred. Pre-Test roles remain shared primary, full R4 secondary, audio baseline. Stage 7 remains active; Final Test remains locked pending manager review.


### TASK-007C corrective config-native DEV firewall

Status: corrective DEV firewall passed; Final Test remains pending the pushed firewall. Branch: `codex/task-007c`. The initial incomplete result remains preserved above; its Test invocation count was 0 and no external report existed.

Installed Chimera semantics were inspected read-only: `device = cuda` falls back to CPU only when CUDA is unavailable, and `mixed_precision: true` enables `torch.amp.autocast(device_type=device.type, enabled=use_amp)` with the default CUDA autocast dtype. In this environment the dtype is `torch.float16`; model parameters remain float32 and evaluator outputs are accumulated/cast to CPU float32 for metrics.

DEV-only A/B/C diagnostic for shared seed43 used canonical DEV rows only and the verified checkpoint SHA256 `ec746e519e551d69d59954dffff0a73968fb4ab555d43020b7722e635cc477ee`. A CPU float32/no-autocast reproduced `0.7229314130/0.8181521375/0.7705417752`; B CUDA float32/no-autocast reproduced the same scores with zero sign disagreements and max raw-logit delta `1.9669533e-06`; C CUDA config-native float16 autocast reproduced `0.7244960383/0.8181521375/0.7713240879`, matching frozen `0.724496/0.818152/0.771324` within `0.0005`. C-vs-A had one sign disagreement and max raw-logit delta `0.0028064251`.

Shared controls passed with config-native semantics: seed42 `0.7653242327/0.8836056059/0.8244649193` versus frozen `0.765324/0.883606/0.824465`; seed44 `0.7314476158/0.8383722433/0.7849099296` versus frozen `0.731448/0.838372/0.784910`; every component stayed within `0.0005`. The evaluator now reads each frozen fusion config's `train.device` and `train.mixed_precision`, mirrors Chimera's CUDA fallback/autocast gate, and casts fusion outputs to CPU float32 before metrics. No model/config/cache/source change was made.

Corrective firewall checks passed: evaluator `--help`; metric/raw-threshold/margin-argmax/sparse-mask/Student-t synthetic checks; 15 unique frozen entries; overwrite refusal; all 15 checkpoint/config SHA checks; pseudo-cache SHA check; and full evaluator DEV preflight. All 15 DEV D/P/Mean values matched the frozen ledger within `0.0005`; audio42 compatibility equivalence passed; preflight reported `test_iteration: false`; `git diff --check` passed; forbidden `src`, config, cache, dependency, and audio scopes are unchanged. The corrective firewall must be committed and pushed before the single authorized Final Test invocation.

No Test loader/dataset was constructed or iterated during the corrective diagnostics or full preflight; Test invocation count remains exactly `0`; external report remains absent. Stage 7 remains active pending the pushed firewall and one authorized Final Test run.


### TASK-007C — corrected firewall and standardized Final Test evidence

Status: complete. Exact result: `STAGE-7 FINAL TEST EVALUATION COMPLETE`. Branch: `codex/task-007c`. Corrective firewall commit `bfaae7f812904dd9ae9a7460b394117867db5439` was pushed before Test; evaluator SHA256 `b691c9563bb6ffd63316b664949dcf1bef02d8d3f4e770110c2e70adbc4cee1e`; report `/media/maxim/Programs/Features/WSM/stage7_final_test/final_test_v1.json` SHA256 `59f763f04843c8e54d5d3f9893d3abe0530b47a32ea11f5b9585390edeb515bc`.

The config-native corrective firewall passed all 15 DEV reproductions within `0.0005`, including shared seed43 C-mode `0.7244960383/0.8181521375/0.7713240879`, shared42/44 controls, all frozen checkpoint/config/pseudo-cache SHA checks, audio42 equivalence, synthetic checks, overwrite refusal, and `test_iteration: false`. Exactly one Final Test invocation then completed; protocol counts are TEST_NONE `1364`, TEST_SOFT `1208`, TEST_HARD `1014`.

Five-seed Test D/P/Mean means with frozen two-sided Student-t 95% CIs are:
- TEST_NONE: audio `0.752954/0.848230/0.800592` CI Mean `[0.781012,0.820172]`; R4 `0.790977/0.816028/0.803502` CI `[0.780018,0.826987]`; shared `0.799233/0.812774/0.806003` CI `[0.794417,0.817590]`.
- TEST_SOFT: audio `0.775925/0.850212/0.813068` CI Mean `[0.793383,0.832754]`; R4 `0.802296/0.820389/0.811343` CI `[0.785432,0.837253]`; shared `0.812757/0.817941/0.815349` CI `[0.801248,0.829450]`.
- TEST_HARD: audio `0.788289/0.860246/0.824268` CI Mean `[0.808625,0.839910]`; R4 `0.806044/0.837490/0.821767` CI `[0.790908,0.852626]`; shared `0.815916/0.837049/0.826483` CI `[0.810207,0.842758]`.

Paired Mean mean deltas (95% CI), in order R4−audio, shared−audio, shared−R4: TEST_NONE `+0.002910[-0.021829,+0.027649]`, `+0.005411[-0.014285,+0.025108]`, `+0.002501[-0.013563,+0.018565]`; TEST_SOFT `-0.001726[-0.029386,+0.025934]`, `+0.002281[-0.014017,+0.018578]`, `+0.004006[-0.014503,+0.022516]`; TEST_HARD `-0.002501[-0.029933,+0.024931]`, `+0.002215[-0.013123,+0.017553]`, `+0.004716[-0.015705,+0.025137]`. All paired Mean CIs include zero; no statistical superiority claim.

Descriptive Test−frozen-DEV five-seed Mean deltas (audio/R4/shared): TEST_NONE `+0.035448/+0.013446/+0.015089`; TEST_SOFT `+0.047924/+0.021286/+0.024435`; TEST_HARD `+0.059123/+0.031710/+0.035568`. Full per-seed D/P UAR/MF1/Score, summary min/max/range/std/CI, paired D/P/Mean deltas, and all traceability rows are in [STAGE7_FINAL_TEST_EVIDENCE_EN.md](STAGE7_FINAL_TEST_EVIDENCE_EN.md).

Frozen roles remain unchanged: shared primary, full R4 secondary multimodal reference, temporal audio baseline. No post-Test role/config/checkpoint/threshold revision, training, recalibration, tuning, retry, or further experiment occurred. Main/master untouched; Stage 7 is pending manager closure.


## TASK-008A paper draft v1

Branch: `codex/task-008a-paper-draft`. This presentation-only task created the internal-review manuscript package under `paper/`, including sectioned LaTeX, frozen-evidence result and ablation tables, bibliography, and the native vector SVG method overview. No training, inference, Test invocation, metric recomputation, checkpoint loading, or scientific-claim/role revision occurred. The research programme remains scientifically frozen; this is article drafting only, pending manager review.


## TASK-008A paper draft v2 corrective

Branch: codex/task-008a-paper-draft; prior reviewed commit: 89fbe9af57c5b2ab226fcbdf51480769025d8b0a (corrective evidence commit recorded in Git handoff). This manuscript-only correction expands the internal-review paper into a compact two-column, conference-like draft without claiming official EMNLP formatting.

The revised method describes the actual equal-parameter shared-fusion forward path (projected audio/video, mean disease query, shared candidates, availability-masked gate, fused feature, independent D/P heads) and contrasts it with task-specific full R4. It gives the full frozen observed/pseudo/auxiliary/agreement loss decomposition, detached pseudo/reliability contract, warm-up (3 observed-only + 5 ramp), selected loss constants, and RA-STCH scalarization with its exact unsupported-contribution boundaries.

The Results/Discussion now includes canonical protocol membership, final five-seed DEV/Test/paired summaries, the internal DEPART-like V1/V2 DEV/NONE/SOFT/HARD table, a separate non-leaderboard published-DEPART UAR context table, DEV calibration diagnostics, and a quantitative Stage-6 ablation companion. Frozen roles are unchanged: shared fusion is the primary parsimonious candidate, full R4 the secondary multimodal reference, temporal audio the baseline; T1 remains standalone and T2 remains blocked.

The method figure was redrawn as [wsm_method_overview.svg](../paper/figures/wsm_method_overview.svg) and exported to the vector [wsm_method_overview.pdf](../paper/figures/wsm_method_overview.pdf), with primary inference and training-only supervision panels. The figure style is an original compact presentation informed only by the general readability of the supplied TACME manuscript. The bibliography corrects the full DEPART author/venue/DOI entry and identifies TACME as a supplied manuscript after title-page inspection.

Build verification: pdflatex, bibtex, and two final pdflatex passes completed; 11 PDF pages including references (approximately nine pages of body/tables); no undefined citations/references, no overfull boxes, and no missing/raster figure. Rendered figure, primary-result, and ablation-table pages were visually inspected; final tables precede the bibliography. No training, inference, checkpoint loading, metric recomputation, Test invocation, cache/source/config/script change, or frozen scientific role/claim revision occurred. src/audio remains unchanged.


### TASK-008A bibliography-only final fix

Prior manuscript commit: a6b935e2fe90c42de786a285830350825111fc6f. Editorial metadata only: lin2024stch now matches the official ICML/PMLR record, and durand2019learning now matches the official CVPR 2019 partial-label record. The manuscript build still passes; no scientific text, equations, metrics, tables, figures, roles, Test evidence, or conclusions changed.



### TASK-008B paper submission QA / appendix pass

Branch: codex/task-008b-paper-submission-qa; final evidence commit SHA is recorded in the Git handoff. Paper-only submission QA added appendix_reproducibility.tex: frozen architecture, optimization, pseudo-cache, checkpoint, and evaluator identities; accepted five-seed DEV and one-shot Final-Test tables; and the exact Stage-6 claim traceability matrix. The DEPART context caption now explicitly limits the comparison to UAR, a different formulation, non-equivalence of cleaned DEPART Test and WSM TEST_HARD unless independently established, and no leaderboard use.

QA audited the manuscript against the accepted final and stage ledgers: no Final-Test superiority, missing-label, comorbidity, clinical-validity, causal-corpus claims, or scientific role change was introduced. Roles remain frozen: shared fusion primary by the pre-Test DEV-only parsimony decision, full R4 secondary, temporal audio baseline; T1 standalone and T2 blocked. TACME remains a supplied manuscript with no formal venue/year asserted; n.d. avoids a BibTeX empty-year warning.

Build verification used the required pdflatex, bibtex, pdflatex, pdflatex sequence: success, 14 pages, no undefined citations/references, no overfull boxes, and no BibTeX metadata warning. Rendered title/abstract, method figure, main-result and ablation tables, appendix tables, and bibliography transition were visually inspected; the method figure remains vector and legible, the claim matrix is unscaled readable 8-point text, the appendix follows the conclusion, and no floats appear after references. No training, inference, Test invocation, checkpoint loading, metric recomputation, or new statistics occurred; src/audio, source, configs, scripts, caches, and evidence ledgers remain unchanged.


### TASK-008C final paper release bundle

Branch: codex/task-008c-paper-release; final evidence commit SHA is recorded in the Git handoff. Release packaging adds paper/release/WSM_paper_draft.pdf, paper/README.md, paper/SUBMISSION_CHECKLIST.md, paper/release/SOURCE_MANIFEST.md, and paper/release/RELEASE_NOTES.md. The clean accepted manuscript build is 14 pages; PDF SHA256 is fca4fcc90e4107592d8b1ce0f46bd8825d67476b58074fa5575f0cb2ee74eefd.

The required pdflatex, bibtex, pdflatex, pdflatex sequence passed with no undefined citations/references, missing figures, or overfull boxes. Rendered title/abstract, method figure, main results, DEPART context, Stage-6 ablations, reproducibility appendix, and references were visually inspected; the method figure remains vector-only. The release is venue-neutral: official class/style, authorship, limits, and required venue statements remain explicitly pending.

No scientific content, roles, claims, thresholds, or results changed. Shared remains the DEV-only primary parsimonious candidate, full R4 the secondary multimodal reference, and temporal audio the baseline; T1 remains standalone and T2 blocked. No experiment, inference, Test invocation, checkpoint loading, metric recomputation, or new statistics occurred; src/audio, source, configs, scripts, caches, and evidence ledgers remain unchanged.


### TASK-011B-BUNDLE-C1 — corrected Shared ownership firewall

Status: pre-production firewall complete; production not started. Branch: `codex/task-011b-shared-progress-full-suite`; exact base: `1027209de07f65098adcb69fc41e9657cc77c1cf`. The Shared gradient-MTL ownership map now treats the full Shared path as shared: `23/14/14/0`; task-aware R4 remains `14/18/18/1`. The gradient-MTL callback validates the model boolean `task_aware_fusion` and rejects missing/non-boolean or mismatched ownership.

Regression command `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_task010a_gradient_mtl.py` passed `25 passed`, including Shared PCGrad/CAGrad/GradNorm/DB-MTL finite smokes and callback compatibility/rejection tests. The orchestration script is [scripts/common/run_postclosure_shared_progress_full_suite.py](../scripts/common/run_postclosure_shared_progress_full_suite.py). `--prepare`, `--dry-run`, and `--validate` passed; the frozen manifest contains exactly `52` rows in the required A-G order. Validation proved config validity, canonical/DEV-only TRAIN/DEV membership equivalence, parameter count `552775`, cache contracts, both ownership modes, and TRAIN-only forward/loss/backward smokes. It reported `test_loader_accessed=False`.

No production invocation has occurred (`0/52`), no Test loader/metric/mixed summary/Final-Test artifact was inspected, and no metric-bearing run has been retried. Firewall commit must be pushed before the exact `--run` production command. Main/master and `src/audio` remain untouched.


### TASK-011B-BUNDLE-R2 — post-production DEV-only collection recovery

Status: complete; production evidence remains frozen at `52/52`, `0` failed, with no production rerun. Branch: `codex/task-011b-shared-progress-full-suite`; firewall commit: `c05703667bf7337fe0af8e5e4d15690ffb1a003b`.

`POST-PRODUCTION REPORTING-ONLY COLLECTION RECOVERY` resolved MLflow logger suffixes using anchored prefixes, FINISHED status, frozen config snapshots/SHA, seed/model/loss/optimizer identity, selected checkpoint identity/SHA, selected epoch, and allowlisted DEV-only metrics. Mapping audit passed `52/52` manifest rows to `52` scientific results, referencing `53` MLflow IDs with exactly one duplicate-alias row. Row 34 aliases `44455703f7de44f6b032b4cde21ddd5d` and `b43e09ce179a49f199993628ad3c063d` matched exactly; canonical ID is the lexicographically smaller ID, bookkeeping only. Row-34 selected epoch `5`, checkpoint SHA `4215e95623afe66cd42706f42f0f72e2662a679561cff844e89a7fde836e40f1`, DEV Mean `0.8073689041`.

Safe `--collect` passed with `test_metrics_read=False` and generated [POSTCLOSURE_SHARED_PROGRESS_FULL_SUITE_EN.md](POSTCLOSURE_SHARED_PROGRESS_FULL_SUITE_EN.md). Five-seed parent DEV Mean mean/std/range/95% CI: `0.801826/0.029208/0.078680/[0.765559,0.838092]`. All Group B/C/D/E/F/G DEV-only means and paired same-seed deltas are recorded in the dossier. Progress/gradient-MTL substitution results remain descriptive because common hyperparameters were tuned under Progress. No Test loader, Test metric, mixed raw summary, Final-Test artifact, production config, manifest, checkpoint, or model/training semantics changed.


### TASK-011B-BUNDLE-R3 — corrected DEV synthesis semantics

Status: complete; reporting-only correction on `codex/task-011b-shared-progress-full-suite`. No production run, sweep, retry, inference, or Test access occurred. The accepted production ledger remains exactly `52/52` COMPLETE with `0` failed; safe collection remains `52` scientific results, `53` referenced MLflow IDs, and exactly `1` equivalent alias row. `dev_only_results.json` retains `test_access: false` and no Test metric keys.

The corrected dossier is [POSTCLOSURE_SHARED_PROGRESS_FULL_SUITE_EN.md](POSTCLOSURE_SHARED_PROGRESS_FULL_SUITE_EN.md), SHA256 `5ec40adf96954b8fe5c99876bc3c8c8575b6a94b76fbc7a31b25166f8c3cd624`. D-only and P-only controls now report only their selected task UAR/F1/Score, task Score mean/sample SD, and paired task-Score deltas versus the joint parent; joint Mean is explicitly absent. All Group B/C/D/F/G two-task rows now include D/P UAR/F1/Score means, Mean mean/sample SD, paired same-seed DEV Mean deltas, and wins/3.

The dossier includes the five-seed tuned-parent versus frozen Shared trial012 DEV comparison: tuned-minus-frozen Mean deltas `+0.0092344328/-0.0163041676/+0.0307886787/+0.0176691360/+0.0131680720`, mean delta/sample SD/95% exploratory paired t CI `+0.0109112304/0.0172435866/[-0.0104995103,+0.0323219710]`, wins `4/5`; task-score mean deltas D `-0.0014224191` with `2/5` wins and P `+0.0232447067` with `4/5` wins. These are descriptive DEV-only post-closure comparisons, not Final-Test superiority claims. Progress/gradient-MTL rows remain frozen-recipe substitution audits, not globally tuned rankings.

No source/config/manifest/ledger/checkpoint/model/loss semantics changed; `src/audio` remains unchanged; main/master remain untouched.


### TASK-012B-C2 — aggregate coverage and synthetic evidence correction

Status: complete; overall audit remains INCONCLUSIVE — pending manager review. C2 changes only audit completeness/coverage, not the original scientific numbers or verdict. The offline output now exposes all seven series (D/P UAR, MF1, Score, and joint Mean), each with mean, sample SD, min/max/range, and Student-t CI, for DEV plus TEST_NONE/SOFT/HARD. DEV Mean is 0.801825630372/0.029207897030/[0.765559236167,0.838092024578]; Test Mean summaries remain 0.768738/0.764781/0.755323 with direct deltas -0.037265101865/-0.050567885186/-0.071159977989, wins 0/5 each.

Synthetic coverage now demonstrates unequal observed counts D=4/P=3 with expected D confusion (0,2,1,1), Score 0.225, P confusion (1,0,0,2), Score 1.0, joint Mean 0.6125; and unequal fake batches of sizes 2/3 with global confusion (1,3,0,1), UAR 0.625, MF1 0.4, Score/Mean 0.5125 versus unweighted batch 0.5. The suite remains 8 passed; strict validator remains OFFLINE_CHECKS_PASS.

New untracked output: logs/task012b_shared_progress_test_correctness_audit/audit_summary_c2.json, 14,902 bytes, SHA256 94543e6c0fc8be4e9baf1c55382deed3668139c002c7ee312bc8a51fd475e29. Original DEV/Test/marker hashes are unchanged. No production inference, dataset/model construction, Test rerun, training, tuning, or selection occurred; src/audio remains unchanged.

### TASK-012B-C1 — Corrective Shared + Progress Test correctness audit

Status: complete as an offline read-only audit; **Codex audit verdict — pending manager review: INCONCLUSIVE**. Branch: `codex/task-012b-shared-progress-test-correctness-audit`, based on `origin/main` at `11231ca0c528236441bf4f4dae09e8fd616e1928`.

Changed files remain the four scoped TASK-012B files: [corrective dossier](TASK012B_SHARED_PROGRESS_TEST_CORRECTNESS_AUDIT_EN.md), [strict offline validator](../scripts/common/audit_task012a_saved_evidence.py), [synthetic audit tests](../tests/test_task012b_shared_progress_test_correctness_audit.py), and this ledger. The validator now enforces exact artifact identity/length, DEV/Test cardinality and identity joins, observed counts, finite/range checks, tight arithmetic, frozen config/checkpoint SHA checks, explicit heading/key historical parsing, all seven aggregates, direct paired Mean deltas, task contributions, and rounding residuals. The unchanged production metric function was compared with an independent oracle on fabricated tensors; malformed fixtures are rejected.

Verification: `.venv/bin/python -m pytest -q tests/test_task012b_shared_progress_test_correctness_audit.py` — `8 passed` at `2026-10-09T11:07:06Z–11:07:16Z`; strict offline validator — exit `0`, `OFFLINE_CHECKS_PASS` at `2026-10-09T11:07:26Z`. Authentic artifact hashes remain DEV `67a7c70754070dd0594262ab706aac8d3d853d9b2a84f349adf97e08e8ac2e64`, Test `ddcae2835c04a346c1a27d3e9fa9d7c2b2afc96980b46d206dd2e79120baf285`, marker `3a37730e932f29c5928b930263d6c745c6c9f9171b9c42b34990320de329e12d`. Test Mean means remain `0.768738/0.764781/0.755323`; direct deltas remain `-0.037265101865/-0.050567885186/-0.071159977989`, wins `0/5` each.

No original TASK-012A correctness defect was demonstrated. The corrected audit is intentionally INCONCLUSIVE because raw predictions, sample IDs/membership fingerprints, exact historical command transcript, and exact autocast dtype are unavailable. No production evaluator, DataModule/model construction, forward/inference, training, tuning, Test rerun, or follow-up experiment occurred; `src/audio` stayed unchanged and Test metrics were not used for selection.
