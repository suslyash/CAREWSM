# TASK-007D: Final Research / Program Closure Ledger

## Authority and branch

This task follows **MANAGER-DECISION-081**.

Required branch:

    codex/task-007d

Start from current manager-updated `origin/main`.

This task is **documentation-only**.

No training, inference, Test invocation, cache build, checkpoint loading, metric recomputation, new statistical test, source/config/script/dependency change, model/threshold/role revision, or new scientific claim.

## Required reading

Read only:

1. `AGENTS.md`
2. `docs/PROJECT_REQUIREMENTS.md`
3. `docs/PROGRESS_EN.md`
4. `docs/STAGE3_CLAIM_LEDGER_EN.md`
5. `docs/STAGE6_CLAIM_LEDGER_EN.md`
6. `docs/STAGE7_FINAL_FREEZE_EN.md`
7. `docs/STAGE7_FINAL_DEV_EVIDENCE_EN.md`
8. `docs/STAGE7_FINAL_TEST_EVIDENCE_EN.md`
9. `docs/NEXT_TASK_EN.md`

Do not reopen raw run logs unless these accepted ledgers are internally inconsistent. If an inconsistency is found, STOP and report it; do not reinterpret raw history.

## Allowed tracked files

Only:

- `docs/FINAL_RESEARCH_LEDGER_EN.md`
- `docs/PROGRESS_EN.md`
- `docs/README.md`

No other tracked file may change.

## Final frozen roles

Record exactly:

1. **Primary parsimonious paper candidate:** equal-parameter shared fusion.
2. **Secondary multimodal reference:** full R4 trial012.
3. **Baseline:** frozen temporal audio.
4. **Standalone text evidence:** T1 transcript model; not evidence that text adds to A+V.
5. **T2:** blocked before cache/model/training under its frozen description-generation gate.

Clarify that shared's primary role was frozen before Test from DEV-only parsimony evidence and is not a statistical-superiority claim.

## Stage-7 evidence

Copy the exact five-seed DEV and Final-Test summaries, paired deltas/CIs, evaluator firewall SHA, evaluator SHA256, report path/SHA256, protocol counts, and role boundaries from the accepted Stage-7 evidence documents. Do not recompute them.

Required interpretation:

- both multimodal finalists had positive paired DEV Mean advantage over audio under the frozen descriptive CI plan;
- shared and full R4 were not separated by paired DEV Mean CI;
- shared has the highest five-seed Mean point estimate on TEST_NONE, TEST_SOFT, and TEST_HARD;
- every predeclared paired Test Mean 95% CI includes zero;
- therefore no Final-Test statistical-superiority claim is authorized;
- Test did not revise roles;
- no post-Test revision occurred.

## Stage-6 claims

Preserve exactly the supported conclusions:

1. `DIRECT PSEUDO-SUPERVISION CONTRIBUTION SUPPORTED`
2. `UNCERTAINTY/RELIABILITY CONTRIBUTION SUPPORTED`
3. `ONLINE AUDIO INPUT MODALITY CONTRIBUTION SUPPORTED`
4. `SAMPLE-SPECIFIC PSEUDO ALIGNMENT CLAIM SUPPORTED BY THIS NEGATIVE CONTROL`

Preserve exactly the unsupported conclusions:

1. `SPARSE MTL JOINT-TRAINING CONTRIBUTION NOT SUPPORTED`
2. `TASK-AWARE FUSION CONTRIBUTION NOT SUPPORTED`
3. `SEMANTIC-EVIDENCE CONTRIBUTION VIA DEPRESSION PSEUDO ACCEPTANCE NOT SUPPORTED`
4. `RA-STCH BALANCING CONTRIBUTION OVER EQUAL NOT SUPPORTED`
5. `RA-STCH ADVANTAGE OVER SIMPLER BALANCING NOT SUPPORTED`
6. `VIDEO MODALITY CONTRIBUTION NOT SUPPORTED`
7. `FUSION AMPLIFICATION OF CORPUS DECODABILITY NOT SUPPORTED`

Diagnostic-only: `STRONG CORPUS-ID DECODABILITY FLAG NOT TRIGGERED`.

Keep every conclusion in its narrow scope. Unsupported contribution does not mean universal uselessness.

## Stage-3 evidence

Preserve:

- `STAGE-3 TEXT/DESCRIPTION EVIDENCE MATRIX COMPLETE`
- `STAGE-3 TWO-FAMILY BUDGET EXHAUSTED`
- `T1 THREE-SEED TEXT BASELINE COMPLETE`
- T1 D/P/Mean `0.6268063333/0.8447246667/0.7357653333`
- T1 is standalone text evidence only
- `T2 DESCRIPTION GENERATION CONTRACT BLOCKED`
- T2 failed its frozen preflight on demographic/identity inference `4/24` and concision `18/24`
- no T2 performance claim exists

## Critical limitations

Explicitly state:

- unknown labels remain masked, never treated as negative;
- observed ground truth overrides pseudo supervision;
- no verified missing-label correctness claim;
- no verified comorbidity recovery claim;
- no claim that pseudo labels are clinically correct;
- no claim that T1 text improves A+V;
- no claim that T2 improves anything;
- no statistical-superiority claim among shared/full/audio on Final Test;
- no causal corpus-shortcut claim;
- no universal uselessness claim for unsupported components;
- no post-Test revision;
- no further experiment is authorized by the current programme.

## Traceability

Point to the Stage-3/6 ledgers, Stage-7 freeze/DEV/Test dossiers, final evaluator path/SHA, final report path/SHA, and the 15-checkpoint ledger in the Stage-7 DEV evidence.

## Exact final conclusion

If the accepted ledgers are internally consistent and the synthesis is faithful, record exactly:

    WSM RESEARCH EVIDENCE PROGRAM COMPLETE

and:

    NO FURTHER EXPERIMENT AUTHORIZED BY THE CURRENT PROGRAMME

If not, record:

    WSM RESEARCH EVIDENCE PROGRAM CLOSURE INCOMPLETE

and name the inconsistency. Do not run code to resolve it.

## PROGRESS / README

Update PROGRESS to show Stage 7 COMPLETE, no active experimental task, final roles, Final-Test completion, pointer to the final ledger, and no further experiment/Test authorization.

Add `FINAL_RESEARCH_LEDGER_EN.md` to README as the primary final-results/claims entrypoint.

## Final scope checks

Run:

    git diff --check
    git status --short
    git diff --stat origin/main...HEAD
    git log -12 --oneline --decorate

Only the three authorized documentation files may differ.

## Acceptance criteria

TASK-007D passes only if:

- branch exactly `codex/task-007d`;
- documentation-only scope;
- every metric/CI/claim matches accepted ledgers;
- roles remain frozen;
- supported/unsupported/blocked distinctions remain intact;
- no new scientific claim;
- no compute/Test/log inspection beyond reading committed docs;
- exact completion strings are correct;
- branch pushed;
- main/master untouched.

Passing TASK-007D closes the current WSM research evidence programme and authorizes no follow-up experiment.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Include branch, final SHA, pushed status, main/master untouched, final ledger path, exact completion strings, frozen roles, Final-Test completion summary, Stage-3/6 conclusion summary, no compute/new claim, and programme closure status.

For section 6 write only:

    Manager final review of TASK-007D; do not start another task.

Stop.
