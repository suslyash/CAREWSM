# TASK-011 — Multi-feature static deep fusion infrastructure

## Goal

Extend the current joint CARE+WSM native audio pipeline so one experiment can consume **multiple static/T=1 feature families** in a deterministic order.

This task is infrastructure + coverage + smoke validation only.

Do not launch a new full five-fold combination matrix.

The first target families are:

- `emotionmsp_embeds` — 1024;
- `trillsson4_1` — 1024;
- `wav2vec_xlsr300m` — 1024;
- `disvoice_articulation_static` — 488.

The first three are the strongest single-family representations by the manager-reviewed VAL ranking. The DisVoice family is included only as a potential complementary handcrafted feature.

## Base / branch

Repository:

`suslyash/CAREWSM`

Start from current `origin/main`.

Create:

`codex/task-011-static-deep-fusion`

Normal non-force push is authorized.

Do not merge.

## Required reading

Read:

1. `README.md`
2. `AGENTS.md`
3. `docs/PROJECT_REQUIREMENTS.md`
4. `docs/METHOD.md`
5. `docs/EXPERIMENT_PROTOCOL.md`
6. `docs/PLAN.md`
7. `docs/PROGRESS_EN.md`
8. `docs/evidence/MANAGER_AUDIO_EXPERIMENT_REPORT_2026-11-04.md`
9. this task

Inspect `git status --short` before edits.

## Scientific constraints

The task must preserve:

- joint CARE+WSM fold construction;
- missing-label masks;
- VAL-only selection policy;
- TEST statistics policy;
- feature-aware filtering;
- existing single-family configs and behavior.

Static vectors must remain static.

Do **not** repeat T=1 embeddings along time.

A multi-static representation should remain conceptually one static sample vector, e.g. shape:

`[1, sum(component_dims)]`

for the current native model interface.

## A. Add a generic multi-feature input contract

Extend the data/config path to accept an ordered list such as:

```yaml
feature_families:
  - emotionmsp_embeds
  - trillsson4_1
```

Requirements:

- preserve the exact configured order;
- validate every family exists in the registry;
- validate selected families are compatible with this static fusion path;
- load each component independently;
- concatenate only along feature/channel dimension;
- expose deterministic total input dimension;
- expose component names and component dimensions in sample/run metadata;
- do not silently sort or deduplicate user input;
- reject duplicate family names explicitly.

Backward compatibility with the existing singular `feature_family` configs is required unless there is a compelling technical reason; if not possible, stop and report rather than mass-rewriting old configs.

## B. Availability / filtering policy

For selected family list `S`, include a sample only when every family in `S` is valid according to the existing feature availability contract.

Do not fill unavailable features with zero.

Do not globally delete samples.

The resulting fold views must still preserve:

- corpus identity;
- split/fold identity;
- profile identity;
- Depression/Parkinson label masks.

## C. Candidate combination coverage

Using real CARE and WSM metadata, report retained counts and percentages for at least:

1. `emotionmsp_embeds + trillsson4_1`
2. `emotionmsp_embeds + wav2vec_xlsr300m`
3. `trillsson4_1 + wav2vec_xlsr300m`
4. `emotionmsp_embeds + trillsson4_1 + wav2vec_xlsr300m`
5. `emotionmsp_embeds + trillsson4_1 + disvoice_articulation_static`

For each combination report, per fold where applicable:

- joint TRAIN retained/expected;
- joint VAL retained/expected;
- CARE TEST retained/expected;
- WSM TEST retained/expected;
- Depression observed counts;
- Parkinson observed counts.

No training is required to produce these numbers.

## D. Model compatibility

The generic fused static vector must work with:

- `carewsm_pooled_mlp`;
- `carewsm_md_former`.

Do not use MPTSNet for these T=1-only combinations.

Do not add fake temporal tokens merely to make MD-Former look temporal. Preserve the existing static/T=1 interpretation and document that MD-Former is a comparative architecture, not evidence of temporal dynamics.

## E. Smoke configs

Add only bounded smoke configs, not the full matrix.

At minimum create fold-0 smoke configs for:

- emotion + TRILLsson with MLP;
- emotion + TRILLsson with MD-Former;
- emotion + TRILLsson + XLS-R with MLP.

Use small bounded sample limits / epochs sufficient to verify:

- loading;
- collation;
- forward;
- backward;
- metrics;
- callbacks;
- checkpoint monitor;
- TEST evaluation plumbing.

These smoke runs must not be presented as scientific performance evidence.

## F. Tests

Add focused tests covering at least:

- ordered concatenation;
- expected summed dimension;
- singular `feature_family` backward compatibility;
- duplicate family rejection;
- unknown family rejection;
- missing one component causes sample exclusion;
- labels/observed masks unchanged;
- CARE/WSM corpus metadata preserved;
- MLP forward/backward on fused vector;
- MD-Former forward/backward on fused vector.

Synthetic tests must not require the real datasets.

## G. Evidence

Create:

`docs/evidence/TASK-011_STATIC_DEEP_FUSION_EN.md`

Include:

- implementation summary;
- API/config contract;
- exact component dimensions;
- five requested coverage tables;
- smoke commands/results;
- tests;
- limitations;
- recommendation for the bounded TASK-012 combination matrix.

Update:

`docs/PROGRESS_EN.md`

Do not rewrite the manager experiment report.

## H. Prohibitions

Do not:

- run the full five-fold feature-combination matrix;
- run loss-balancing comparisons;
- run Optuna;
- install/enable Mantis;
- add TSCMamba dependencies;
- change single-feature scientific results;
- use TEST to select a combination;
- change thresholds;
- change canonical splits;
- re-extract features;
- modify raw CARE/WSM data.

## Verification

Before commit:

- focused unit tests pass;
- bounded fold-0 smoke runs pass;
- no old single-family config is broken;
- `scripts/run_audio_experiments.sh --check --all` still passes for the existing 110 single-feature configs;
- `git diff --check`;
- worktree clean after commit.

Commit message:

`TASK-011: add static deep feature fusion infrastructure`

## Required handoff

Use exactly:

1. Outcome
2. Changed files
3. Verification
4. Real-data coverage
5. Smoke evidence
6. Git handoff
7. Blockers and risks
8. Next atomic step

Section 8 must say exactly:

`Manager review of TASK-011; do not launch the full feature-combination matrix, loss-balancing study, Mantis, TSCMamba dependency work, or Optuna yet.`
