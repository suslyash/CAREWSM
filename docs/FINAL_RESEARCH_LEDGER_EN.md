# WSM Final Research / Program Closure Ledger

`WSM RESEARCH EVIDENCE PROGRAM COMPLETE`

`NO FURTHER EXPERIMENT AUTHORIZED BY THE CURRENT PROGRAMME`

This is the authoritative final-results and claims entrypoint. It synthesizes
accepted Stage-3, Stage-6, and Stage-7 dossiers only. No training, inference,
Test invocation, checkpoint loading, metric recomputation, model selection,
threshold fitting, or new scientific claim was performed for this ledger.

## Frozen roles

1. **Primary parsimonious paper candidate:** equal-parameter shared fusion.
2. **Secondary multimodal reference:** full R4 trial012.
3. **Baseline:** frozen temporal audio.
4. **Standalone text evidence:** T1 transcript model; not evidence that text
   adds to A+V.
5. **T2:** blocked before cache/model/training under its frozen
   description-generation gate.

The shared primary role was frozen before Final Test from DEV-only parsimony
evidence: shared and full R4 each have 295,239 trainable parameters, and the
five-seed DEV paired Mean CI did not separate them. This is not a
statistical-superiority claim.

## Stage-7 DEV evidence

The frozen descriptive plan uses arithmetic mean, sample standard deviation,
and the two-sided 95% Student-t interval with multiplier
`2.7764451051977987`. These intervals are not multiplicity-corrected
significance tests.

Values below are `mean/std/min/max/range/[95% CI]`.

| Method | D | P | Mean |
|---|---|---|---|
| temporal audio | `0.721720/0.020065/0.697398/0.747918/0.050520/[0.696805,0.746634]` | `0.808569/0.012800/0.794924/0.827735/0.032811/[0.792676,0.824463]` | `0.765145/0.015234/0.748216/0.787827/0.039611/[0.746230,0.784059]` |
| full R4 trial012 | `0.731250/0.015928/0.718527/0.759025/0.040498/[0.711473,0.751027]` | `0.848863/0.030547/0.804363/0.879259/0.074896/[0.810934,0.886792]` | `0.790057/0.021364/0.761445/0.819142/0.057697/[0.763530,0.816583]` |
| equal-parameter shared fusion | `0.737690/0.016475/0.724496/0.765324/0.040828/[0.717234,0.758147]` | `0.844139/0.024284/0.818152/0.883606/0.065454/[0.813986,0.874292]` | `0.790914/0.019857/0.771324/0.824465/0.053141/[0.766258,0.815571]` |

Both multimodal finalists had positive paired DEV Mean advantage over audio:
R4-minus-audio mean `0.024912`, CI `[0.001169,0.048655]`; shared-minus-audio
mean `0.025769`, CI `[0.009556,0.041984]`. Shared-minus-R4 mean was `0.000858`,
CI `[-0.009704,0.011419]`; therefore shared and R4 were not separated by the
paired DEV Mean CI.

## Stage-7 Final Test evidence

Exact result: `STAGE-7 FINAL TEST EVALUATION COMPLETE`.

- Evaluator firewall commit: `bfaae7f812904dd9ae9a7460b394117867db5439`
- Evaluator: `scripts/common/evaluate_stage7_final_test.py`
- Evaluator SHA256: `b691c9563bb6ffd63316b664949dcf1bef02d8d3f4e770110c2e70adbc4cee1e`
- Report: `/media/maxim/Programs/Features/WSM/stage7_final_test/final_test_v1.json`
- Report SHA256: `59f763f04843c8e54d5d3f9893d3abe0530b47a32ea11f5b9585390edeb515bc`
- Protocol counts: `TEST_NONE=1364`, `TEST_SOFT=1208`, `TEST_HARD=1014`.
- The 15-checkpoint ledger is preserved in [Stage-7 DEV evidence](STAGE7_FINAL_DEV_EVIDENCE_EN.md).

### Five-seed Test summaries

| Protocol | Method | D mean [95% CI] | P mean [95% CI] | Mean mean [95% CI] |
|---|---|---:|---:|---:|
| TEST_NONE | temporal audio | `0.752954 [0.723614,0.782294]` | `0.848230 [0.832241,0.864219]` | `0.800592 [0.781012,0.820172]` |
| TEST_NONE | full R4 trial012 | `0.790977 [0.746733,0.835221]` | `0.816028 [0.784314,0.847742]` | `0.803502 [0.780018,0.826987]` |
| TEST_NONE | equal-parameter shared fusion | `0.799233 [0.763389,0.835076]` | `0.812774 [0.782704,0.842844]` | `0.806003 [0.794417,0.817590]` |
| TEST_SOFT | temporal audio | `0.775925 [0.746616,0.805233]` | `0.850212 [0.836083,0.864341]` | `0.813068 [0.793383,0.832754]` |
| TEST_SOFT | full R4 trial012 | `0.802296 [0.753472,0.851120]` | `0.820389 [0.784586,0.856192]` | `0.811343 [0.785432,0.837253]` |
| TEST_SOFT | equal-parameter shared fusion | `0.812757 [0.771212,0.854303]` | `0.817941 [0.783750,0.852131]` | `0.815349 [0.801248,0.829450]` |
| TEST_HARD | temporal audio | `0.788289 [0.766950,0.809628]` | `0.860246 [0.845660,0.874832]` | `0.824268 [0.808625,0.839910]` |
| TEST_HARD | full R4 trial012 | `0.806044 [0.753113,0.858975]` | `0.837490 [0.810052,0.864927]` | `0.821767 [0.790908,0.852626]` |
| TEST_HARD | equal-parameter shared fusion | `0.815916 [0.772020,0.859812]` | `0.837049 [0.805236,0.868863]` | `0.826483 [0.810207,0.842758]` |

Shared has the highest five-seed Mean point estimate on all three Test
protocols. Every predeclared paired Test Mean 95% CI includes zero, so no
Final-Test statistical-superiority claim is authorized. Test did not revise
roles; no post-Test revision occurred.

### Paired Test Mean deltas and CIs

Values are ordered by seeds `42,43,44,45,46`; each cell is
`deltas; mean; sample std; 95% CI`.

| Protocol | R4 − audio | shared − audio | shared − R4 |
|---|---|---|---|
| TEST_NONE | `-0.007925,0.030535,0.010928,0.003457,-0.022443; 0.002910; 0.019924; [-0.021829,0.027649]` | `-0.008362,0.027439,-0.003705,0.016980,-0.005296; 0.005411; 0.015863; [-0.014285,0.025108]` | `-0.000437,-0.003095,-0.014633,0.013523,0.017147; 0.002501; 0.012937; [-0.013563,0.018565]` |
| TEST_SOFT | `-0.003659,0.025540,0.013102,-0.011522,-0.032090; -0.001726; 0.022277; [-0.029386,0.025934]` | `-0.004154,0.023634,-0.002277,0.004759,-0.010559; 0.002281; 0.013125; [-0.014017,0.018578]` | `-0.000494,-0.001907,-0.015380,0.016281,0.021531; 0.004006; 0.014907; [-0.014503,0.022515]` |
| TEST_HARD | `-0.018381,0.023630,0.019476,-0.015863,-0.021366; -0.002501; 0.022093; [-0.029933,0.024931]` | `-0.012589,0.021361,0.000081,0.004083,-0.001860; 0.002215; 0.012353; [-0.013123,0.017553]` | `0.005791,-0.002269,-0.019395,0.019946,0.019506; 0.004716; 0.016447; [-0.015706,0.025137]` |

## Stage-6 frozen conclusions

Supported:

1. `DIRECT PSEUDO-SUPERVISION CONTRIBUTION SUPPORTED`
2. `UNCERTAINTY/RELIABILITY CONTRIBUTION SUPPORTED`
3. `ONLINE AUDIO INPUT MODALITY CONTRIBUTION SUPPORTED`
4. `SAMPLE-SPECIFIC PSEUDO ALIGNMENT CLAIM SUPPORTED BY THIS NEGATIVE CONTROL`

Unsupported:

1. `SPARSE MTL JOINT-TRAINING CONTRIBUTION NOT SUPPORTED`
2. `TASK-AWARE FUSION CONTRIBUTION NOT SUPPORTED`
3. `SEMANTIC-EVIDENCE CONTRIBUTION VIA DEPRESSION PSEUDO ACCEPTANCE NOT SUPPORTED`
4. `RA-STCH BALANCING CONTRIBUTION OVER EQUAL NOT SUPPORTED`
5. `RA-STCH ADVANTAGE OVER SIMPLER BALANCING NOT SUPPORTED`
6. `VIDEO MODALITY CONTRIBUTION NOT SUPPORTED`
7. `FUSION AMPLIFICATION OF CORPUS DECODABILITY NOT SUPPORTED`

Diagnostic only: `STRONG CORPUS-ID DECODABILITY FLAG NOT TRIGGERED`.
Unsupported contribution does not mean universal uselessness; all conclusions
remain narrow to their stated ablation boundaries.

## Stage-3 frozen conclusions

- `STAGE-3 TEXT/DESCRIPTION EVIDENCE MATRIX COMPLETE`
- `STAGE-3 TWO-FAMILY BUDGET EXHAUSTED`
- `T1 THREE-SEED TEXT BASELINE COMPLETE`; T1 D/P/Mean:
  `0.6268063333/0.8447246667/0.7357653333`.
- T1 is standalone text evidence only; no claim says T1 improves A+V.
- `T2 DESCRIPTION GENERATION CONTRACT BLOCKED`; demographic/identity inference
  was `4/24` and concision was `18/24`; no T2 performance claim exists.

## Mandatory limitations

- Unknown labels remain masked, never treated as negative.
- Observed ground truth overrides pseudo supervision.
- No verified missing-label correctness claim, comorbidity recovery claim, or
  clinically correct pseudo-label claim exists.
- T2 is not claimed to improve anything.
- No causal corpus-shortcut claim and no universal uselessness claim for
  unsupported components is made.
- No post-Test revision occurred; no further experiment is authorized.

## Traceability

- [Stage-3 claim ledger](STAGE3_CLAIM_LEDGER_EN.md)
- [Stage-6 claim ledger](STAGE6_CLAIM_LEDGER_EN.md)
- [Stage-7 final freeze](STAGE7_FINAL_FREEZE_EN.md)
- [Stage-7 final DEV evidence and 15-checkpoint ledger](STAGE7_FINAL_DEV_EVIDENCE_EN.md)
- [Stage-7 final Test evidence](STAGE7_FINAL_TEST_EVIDENCE_EN.md)

