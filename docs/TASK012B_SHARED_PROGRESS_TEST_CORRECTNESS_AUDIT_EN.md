# TASK-012B-C1 — Corrective independent correctness audit

## Scope and Codex audit verdict — pending manager review

This corrective audit fixes defects in the first TASK-012B audit itself. It remains strictly offline: no production evaluator entrypoint, DataModule construction, model construction/forward, real inference, training, tuning, calibration, Test rerun, or new experiment was performed. The original TASK-012A evaluator, closure evaluator, saved artifacts, and Stage-7 evidence were not modified.

**Codex audit verdict — pending manager review: INCONCLUSIVE.** No original TASK-012A production defect was demonstrated. The retained scalar evidence and source contracts are internally consistent, but raw predictions, sample IDs/membership fingerprints, exact historical command transcript, and exact autocast dtype are unavailable. Those gaps are material to a historical Test correctness judgment; therefore this audit does not claim `VERIFIED WITH STATED LIMITATIONS` or historical reconstruction.

## 1. Input identity and strict validator

The production CLI now enforces exact identity before accepting the saved inputs, while structural/arithmetic functions remain directly testable with synthetic malformed payloads. It uses explicit `ValueError` failures rather than assertions.

| Input | Bytes | Expected SHA-256 | Status |
|---|---:|---|---|
| `dev_preflight.json` | 11,291 | `67a7c70754070dd0594262ab706aac8d3d853d9b2a84f349adf97e08e8ac2e64` | pass |
| `test_results.json` | 12,875 | `ddcae2835c04a346c1a27d3e9fa9d7c2b2afc96980b46d206dd2e79120baf285` | pass |
| `test_invocation.marker` | 48 | `3a37730e932f29c5928b930263d6c745c6c9f9171b9c42b34990320de329e12d` | pass; `COMPLETED`, count 15 |

The validator checks:

- DEV schema, `test_iteration=false`, exactly five unique seeds, five ledger rows, frozen run/MLflow/epoch/config/checkpoint identities, model/recipe/pseudo-cache invariants, and per-seed config/checkpoint SHA files;
- DEV counts 933 with D/P 621/312, all finite in-range metrics, reconstructed score/Mean arithmetic, expected-value gates within 0.0005, and retained `reproduction_difference` values recomputed from the scalar values;
- Test schema, `test_iteration=true`, invocation count 15, exactly 15 rows before indexing, the exact 5×3 seed/protocol Cartesian set, protocol counts 1364/1208/1014, observed D/P counts 827/537, 710/498, 654/360, identity joins to DEV/frozen ledger, raw-evidence declarations, finite ranges, and tight per-row score arithmetic;
- counts as integer non-negative values bounded by protocol membership; retained device/autocast/model/output-dtype fields as metadata, not proof of historical execution.

The negative fixtures now reject: altered UAR with stale Score, a 16th duplicate row, empty DEV results/ledger, invalid artifact SHA, NaN/out-of-range metrics, wrong observed counts, and bad identities. These checks are covered in the synthetic test suite.

## 2. Independent metric contract

The audit oracle and the unchanged production function were run on the same fabricated tensors. The production function was imported only for pure metric computation; no model, dataset, runtime, or evaluator was initialized. Production source identity: `src/common/callbacks/wsm_segment_callback.py` at accepted source/snapshot hash `5e3ca71cc1e9f65be0137ca0e434366cc949ae769324d39a5358010d5558961d`, lines 18–70.

The exact contract is: logits `>=0` are positive; targets must be finite binary values on observed rows; undefined class recall/F1 terms are omitted rather than replaced with zero; empty observed tasks are rejected; `Score=(UAR+MF1)/2`; joint Mean is the arithmetic mean of the two task Scores. The all-negative perfectly predicted task therefore scores 1.0 under both implementations, while a false-positive-only all-negative task exposes the expected undefined-positive-class policy.

Synthetic coverage includes exact zero and tiny positive/negative logits, unequal observed D/P counts, sparse `[N,2]` masks with NaN unobserved values, head-order swaps, imperfect confusion counts, perfect and false-positive one-class tasks, empty-task rejection, malformed target/shape/mask rejection, two differently sized fake batches where global concatenation differs from unweighted batch averaging, and a nonconstant five-value Student-t CI fixture. Result: **8 passed**.

## 3. Historical parsing and aggregates

The C2 synthetic coverage correction is explicit: the five-row sparse fixture has unequal observed counts D=4 and P=3. Expected confusion is D (tn=0, fp=2, fn=1, tp=1), UAR 0.25, macro-F1 0.20, Score 0.225; P (tn=1, fp=0, fn=0, tp=2), UAR/macro-F1/Score 1.0; joint Mean 0.6125. The global fake-batch fixture has sizes 2 and 3, expected per-task (tn=1, fp=3, fn=0, tp=1), UAR 0.625, macro-F1 0.4, Score 0.5125, joint Mean 0.5125, versus unweighted batch Score 0.5. C2 corrects audit completeness and coverage only; it does not change scientific numbers or the verdict.

Stage-7 rows are parsed from explicit `#### test_none`, `#### test_soft`, and `#### test_hard` boundaries. Rows are joined by `(protocol, exact method identity, seed)`; missing or duplicate rows fail. Reordering the protocol sections still produces the same 15 records; duplicate and missing rows are rejected. Historical six-decimal Mean fields are retained directly for paired deltas; task decomposition reports the rounding residual rather than silently substituting the average of rounded task Scores.

All seven series are recomputed with mean, sample SD (`ddof=1`), min, max, range, and Student-t CI using `2.7764451051977987` with df=4. The independent saved-row report percentages are:

| Protocol | D UAR/F1/Score % | P UAR/F1/Score % | Mean % |
|---|---:|---:|---:|
| DEV | 73.70/73.56/73.63 | 86.34/87.14/86.74 | 80.18 |
| TEST_NONE | 76.64/75.72/76.18 | 78.73/76.41/77.57 | 76.87 |
| TEST_SOFT | 75.70/74.72/75.21 | 78.77/76.72/77.75 | 76.48 |
| TEST_HARD | 75.78/74.62/75.20 | 77.09/74.64/75.86 | 75.53 |

The retained C2 output schema is aggregates[split][depression|parkinson][uar|mf1|score] plus aggregates[split].mean.mean. Every series contains mean, sample_sd, min, max, range, ci95_low, and ci95_high. Values below are mean / sample SD / [95% CI]:

| Split | D UAR | D MF1 | D Score | P UAR | P MF1 | P Score | Joint Mean |
|---|---|---|---|---|---|---|---|
| DEV | .736965452848/.016990911558/[.715868449670,.758062456026] | .735570428311/.016284969940/[.715349967425,.755790889198] | .736267940580/.016620068683/[.715631399648,.756904481511] | .863395445135/.047093690349/[.804920905286,.921869984984] | .871371195195/.039822125361/[.821925495803,.920816894588] | .867383320165/.043437926848/[.813448010241,.921318630089] | .801825630372/.029207897030/[.765559236167,.838092024578] |
| TEST_NONE | .766406382721/.029645432857/[.729596716031,.803216049410] | .757185046034/.035552493689/[.713040794573,.801329297494] | .761795714377/.032486185974/[.721458786815,.802132641940] | .787270155587/.031447781872/[.748222577013,.826317734161] | .764092408200/.039047892385/[.715608046020,.812576770380] | .775681281894/.034424412219/[.732937728582,.818424835205] | .768738498135/.030396127732/[.730996720645,.806480275625] |
| TEST_SOFT | .757000607165/.038446437272/[.709263050145,.804738164184] | .747216347202/.044477174098/[.691990641383,.802442053021] | .752108477183/.041380547928/[.700727740596,.803489213771] | .787691838158/.030873560615/[.749357249447,.826026426870] | .767216466731/.039845934958/[.717741203819,.816691729643] | .777454152445/.034560739710/[.734541326195,.820366978694] | .764781314814/.034859987506/[.721496923351,.808065706278] |
| TEST_HARD | .757826455189/.043252472312/[.704121417487,.811531492892] | .746200928639/.049707855746/[.684480473731,.807921383546] | .752013691914/.046387678194/[.694415781940,.809611601888] | .770881914446/.031242662349/[.732089025399,.809674803493] | .746381989770/.044539696660/[.691078651936,.801685327604] | .758631952108/.037815675558/[.711677589200,.805586315017] | .755322822011/.041687958453/[.703560384842,.807085259180] |

| Protocol | Test Mean mean | Direct Test−Stage-7 Mean delta mean | Paired wins |
|---|---:|---:|---:|
| TEST_NONE | 0.768738498135 | -0.037265101865 | 0/5 |
| TEST_SOFT | 0.764781314814 | -0.050567885186 | 0/5 |
| TEST_HARD | 0.755322822011 | -0.071159977989 | 0/5 |

Task decomposition is retained in the utility output and dossier source: each D/P Score delta is divided by two, summed, and compared with the direct historical-Mean delta. Rounding residuals are approximately 0, ±0.0000005, or machine epsilon. These are arithmetic descriptions only; no causal explanation is assigned to Progress, overfitting, or dataset shift.

New untracked output: logs/task012b_shared_progress_test_correctness_audit/audit_summary_c2.json; schema includes status, verdict, artifact hashes/bytes, independent oracle, historical row count, and aggregates. It is 14,902 bytes with SHA-256 94543e6c0fc8be4e9baf1c55382deed3668139c002c7ee312bc8a51fd475e29 and is separate from the original evidence directory.

## 4. Source/config/snapshot comparison

The original TASK-012A firewall is revision `00e4569d988d6fbd95194d9d4c0e1ff4c20f1a1b`, evaluator SHA `7157f32253b6f3e831a00992c180f39fdd0d86d47c4718517b0b7e6584d2a5b7`. The permanent closure revision is evaluator SHA `108079eb382ac580b3293f8716a58848e4d6e1123f952141f48aa3041a247317`; its only semantic change is the unconditional closed-task guard at `test_pass` line 219. Stage-7 evaluator source is `scripts/common/evaluate_stage7_final_test.py` at the accepted Stage-7 revision; relevant lines are cited below.

Snapshot archives were inspected as inert text only. Seed 42 `code.zip` SHA is `b2b92c58f6209b1df19aa7ef0de1249f7e987c23248b278e7adc9a67726568f2`; seeds 43–46 share `52de31a38c779b052c8ef06163727c7cf6d1cb2186dd7ced39f16d8ea51a93de`. The model/data/callback members were checked against the seed-43–46 snapshot family; audio/video member hashes below were checked in seed 42 and seed 43 archives.

| Snapshot member | SHA-256 |
|---|---|
| `src/common/callbacks/wsm_segment_callback.py` | `5e3ca71cc1e9f65be0137ca0e434366cc949ae769324d39a5358010d5558961d` |
| `src/fusion/models/av_r3_disease_query.py` | `3e87cebab421df5ee4a307dcae0c12dc630bc765fbbddfde05deb0783f624f16` |
| `src/fusion/data/wsm_av_fusion_datamodule.py` | `406f7f23a12117de311883b42caa4f5da5d56456cda252430181462d9542d4cc` |
| `src/fusion/data/wsm_ramps_semantic_datamodule.py` | `3311e62845022b44a1f1d0ff442cc3a255a6628e5acfa160ed2c47058fe47659` |
| `src/audio/features/wsm_audio_feature_extractor.py` | `f932c122c666a83ce562022715c2047f78ae2b73b49ec27d6bf56f3aa71429ee` |
| `src/video/data/wsm_video_cache_datamodule.py` | `e2d1b03101996a3ab3344173cc351988c90ec72222fbc1c187361728df3555fa` |
| `src/video/features/clip_video_features.py` | `d224cefd23368ebf6ef91ebf261b2850f4e0c333f133c838b3beba00a901c414` |

| Stage | Original TASK-012A source at firewall | Stage-7 source | Training validation / snapshot source | Actual difference | Evidence/status |
|---|---|---|---|---|---|
| Model construction/loading | `evaluate_postclosure_shared_progress_test.py:126–137`; config-native params; `weights_only=False`, strict state load, `eval()` | `evaluate_stage7_final_test.py:150–153`; hardcoded 768/512/160/96 and method-aware flag | Snapshot model member above; checkpoint payloads statically inspected CPU-only with `weights_only=True` | Stage-7 is a different historical model family/config; TASK-012A uses 256/160 | Static payloads: 51 keys, float32, epochs 7/9/5/19/3; strict load not executed |
| Head order/output | Model snapshot/current `av_r3_disease_query.py:133–148`; independent heads and explicit D/P mapping | Stage-7 calls same model class but different dimensions | Snapshot member hash matches current relevant source | No task-order difference identified | Confirmed source; no forward run |
| Eval mode/no-grad | Original `:140–153` | Stage-7 `:170–205` | Snapshot callback source hash above | Both use no-grad, eval model, CPU float accumulation | Source-confirmed intent only |
| Saved config vs evaluator args/defaults | Config seed42 `task...yaml:6–18`; seeds43–46 equivalent with DEV-only name; evaluator `:175–187` passes batch32, workers0, pin false, persistent false, shuffle false, drop-last false | Stage-7 `:207–210` directly constructs fusion DM with those eval defaults | Snapshot configs preserve saved training values batch32/workers4/pin true/persistent true/shuffle true/drop-last false | Training-loader fields are overridden; eval DataLoader explicitly batch32, workers0, shuffle false, drop_last false at original `:142` and Stage-7 `:198` | Eval-relevant loader semantics source-confirmed; historical runtime not reconstructed |
| Base vs DEV-only module | Base `wsm_av_fusion_datamodule.py:175–230`; RAMPS wrapper `wsm_ramps_semantic_datamodule.py:85–192` | Stage-7 `:207–214` uses base RAMPS module | Snapshot hashes match | DEV-only wrapper only restricts trainer `val_dataloader` at `:201–211`; evaluator directly uses `test_dataset` | Intentional distinction; no production DM instantiated |
| NONE/SOFT/HARD | Base module `:210–217`, filters raw flags; counts `:21–27` | Stage-7 dispatch `:213–224` | Snapshot base-module hash matches | NONE is all non-train/test rows; SOFT/HARD are flag filters | Source contract confirmed; exact membership unavailable |
| Audio input | Base constants `wsm_av_fusion_datamodule.py:21–27`; layer9/pool4; extractor `wsm_audio_feature_extractor.py:18–66` uses WavLM and temporal mean for `audio_cls` | Stage-7 audio path uses separate audio model/evaluation branch `:143–192` | Snapshot audio extractor hash `f932c122…` | Fusion Test uses cached audio features; Stage-7 audio baseline is not the fusion audio path | Cache membership/content unavailable |
| Video representation/fallback | Video cache constants `wsm_video_cache_datamodule.py:18–23`; validation `:70–143`; CLIP extractor `clip_video_features.py:11–25,138–180` | Stage-7 fusion uses same RAMPS/base video cache path | Snapshot video member hashes match | Body ROI when detected, full-frame fallback otherwise; 60 frames, 512 dims | Source/config contract confirmed; no cache read |
| Transforms/augmentation | Cache extraction is frozen; fusion collate `wsm_av_fusion_datamodule.py:152–172` pads and masks; no training augmentation in evaluator | Stage-7 uses same collate through DM | Snapshot collate/data hashes match | Evaluation has padding/valid masks only; training augmentation is not invoked by evaluator | Source-confirmed intent |
| Observed/pseudo targets | `_targets`/collate base `:120–170`; RAMPS pseudo validation `wsm_ramps_semantic_datamodule.py:125–163`; eval pseudo fields neutral `:184–192` | Stage-7 uses sparse observed masks via `:139–141` | Snapshot RAMPS hash matches | Pseudo fields are training-only; observed ground truth drives Test metrics | Source contract confirmed; no labels loaded |
| Logits/threshold/accumulation/row pairing | Original `:143–159`; concatenates all batches before metric call | Stage-7 `:198–205`; same concatenation and sparse metric | Snapshot callback hash matches | No per-batch metric averaging; zero threshold and D/P pairing agree | Synthetic proof; historical predictions unavailable |
| Loader ordering | Original `:142`, Stage-7 `:198`: batch32, shuffle false, drop_last false, workers0 | Same | Snapshot source hashes match | No loader difference found on eval path | Source-confirmed intent; membership/order not retained |
| Device/autocast | Original `:90–95,136–137,148`; config-native CUDA fallback and AMP gate; output cast float32 | Stage-7 `:155–161,196–205`; same gate | Saved artifacts retain CUDA, AMP enabled, float32 model/output | Exact autocast dtype and historical command transcript unavailable | Metadata retained; execution provenance unresolved |
| Stage-7 report runtime | Stage-7 effective path `:155–161,196–205` can use CUDA/AMP, but report construction `:252` writes `device:"cpu", dtype:"float32"` | Same file | Stage-7 evidence text separately states float16 autocast assertion | Report header is not a reliable record of effective fusion runtime | Provenance/reporting limitation, not demonstrated TASK-012A defect |

The DEV gate is useful evidence for model/config/metric compatibility, but it cannot exclude Test-specific filtering, preprocessing, membership, or runtime differences. Counts are not membership fingerprints. No saved metadata permits exact sample-membership or historical row-order verification.

## 5. Findings and limitations

### Corrective audit defects fixed

- hash/length identity is now enforced rather than merely printed;
- DEV/test cardinality, uniqueness, Cartesian identity, counts, finite ranges, arithmetic, and identity joins are now explicit;
- historical parsing is heading-based and key-joined rather than ordinal;
- production metric policy is matched for undefined class terms;
- synthetic tests compare the unchanged production metric implementation and independent oracle;
- seven aggregates, direct paired deltas, wins, task contributions, and rounding residuals are retained.

### Original evaluation defects

No original TASK-012A production correctness defect was demonstrated. The original evaluator’s source-level limitations remain documented: retained scalar evidence does not prove sample membership or exact historical runtime, and the closure guard is administrative only.

### Harmless/intentional differences

The DEV-only wrapper, evaluator loader overrides, cached video fallback policy, and Stage-7’s different hardcoded model dimensions are identifiable source/config differences, not silently conflated as the same run.

### Historical provenance limitations

No raw predictions/logits/labels/sample IDs were retained. Exact autocast dtype, exact UTC command/push transcript, and exact membership/row order are unavailable. The first TASK-012B execution log was not retained and is not reconstructed here. The corrective commands below are the only timestamps recorded for this audit.

## 6. Corrective command record

At `2026-10-09T11:06:11Z`, the real UTC timestamp was recorded before the final corrective verification sequence. The first TASK-012B command transcript is unavailable; no historical timestamps are invented.

The final recorded commands and results are:

2026-10-09T11:28:55Z–11:29:05Z  .venv/bin/python -m pytest -q tests/test_task012b_shared_progress_test_correctness_audit.py -> exit 0, 8 passed (C2 final synthetic run).
2026-10-09T11:25:45Z  .venv/bin/python scripts/common/audit_task012a_saved_evidence.py --dev logs/task012a_postclosure_shared_progress_test/dev_preflight.json --test logs/task012a_postclosure_shared_progress_test/test_results.json --marker logs/task012a_postclosure_shared_progress_test/test_invocation.marker --stage7 docs/STAGE7_FINAL_TEST_EVIDENCE_EN.md --output logs/task012b_shared_progress_test_correctness_audit/audit_summary_c2.json -> exit 0, OFFLINE_CHECKS_PASS; output 14902 bytes, SHA-256 94543e6c0fc8be4e9baf1c55382deed3668139c002c7ee312bc8a51fd475e29.

```text
2026-10-09T11:07:06Z–11:07:16Z  .venv/bin/python -m pytest -q tests/test_task012b_shared_progress_test_correctness_audit.py  -> exit 0, 8 passed
2026-10-09T11:07:26Z  .venv/bin/python scripts/common/audit_task012a_saved_evidence.py --dev ... --test ... --marker ... --stage7 ... --output /tmp/task012b_c1_audit.json -> exit 0, OFFLINE_CHECKS_PASS
```

The authentic input/output schemas, bytes, and SHA-256 values are listed in Section 1. The audit output is a new untracked `/tmp` file; no original artifact directory was overwritten.

## 7. Final status

This C2 commit corrects audit aggregate completeness and synthetic coverage only; it does not alter the original numbers or scientific verdict.

The corrected audit is **INCONCLUSIVE pending manager review**: the saved scalars pass strict structural/arithmetic validation and no original production defect was found, but material historical Test provenance is absent. Original negative numbers remain unchanged. No causal attribution or follow-up authorization is made.
