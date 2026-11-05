# DEEP-005 — Fusion Launcher and CUDA Canary Evidence

## Outcome

The sequential launcher was added and exactly one approved native-turn fusion
canary was executed on the verified local GPU host. The canary completed
successfully through a persistent tmux session with the original YAML budget:
35 executed epochs, early stopping patience 7, and maximum 50 epochs. The
remaining 109 configurations were not launched.

No existing YAML, Dataset, DataModule, model, metric, callback, dependency, or
feature artifact was modified. The unrelated local run-mlflow.sh modification
was preserved.

## Environment and preflight

- Required base: 9656f0a41a64d9547bd98d1e25d80565de401c07 (origin/main at branch
  creation).
- Branch: codex/deep-005-fusion-launcher.
- Host execution: direct local execution on the verified GPU host; no SSH hop
  was introduced.
- GPU: one NVIDIA GeForce RTX 4080, CUDA available through PyTorch
  2.11.0+cu128.
- Python: .venv/bin/python, Python 3.12.3.
- Chimera-ML: .venv/bin/chimera-ml available with train and validate-config.
- tmux: /usr/bin/tmux, version 3.4.
- Manifest: exactly 110 YAML files under
  configs/audio_deep_features/native_turn_fusion/, processed in lexical path
  order.
- No duplicate CAREWSM training process was present before launch. A separate
  pre-existing read-only feature audit process was observed and left
  untouched.
- The approved canary output directory did not exist before launch.

The launcher is scripts/run_audio_deep_fusion.sh. Its modes are:

- --check: validates the 110-file manifest and inspects collisions without
  training;
- --canary: selects only the approved canary;
- --all: processes the same sorted 110-file manifest sequentially, skipping
  verified completed runs and stopping on failed or ambiguous runs.

The launcher uses CUDA_VISIBLE_DEVICES=0, the repository .venv Chimera-ML
binary, explicit exit-status handling, and progress records under
logs/audio_deep_features/fusion_launcher/. Runtime logs are outside Git.

## tmux operation

The canary was started locally on the GPU host with:

    tmux new-session -d -s carewsm-deep-fusion \
      'cd /media/maxim/Programs/Projects/CAREWSM && exec scripts/run_audio_deep_fusion.sh --canary'

Useful operational commands are:

    tmux has-session -t carewsm-deep-fusion
    tmux capture-pane -pt carewsm-deep-fusion:0 -S -40
    tmux attach -t carewsm-deep-fusion
    # detach safely inside tmux: Ctrl-b, then d
    tail -f logs/audio_deep_features/fusion_launcher/launcher.log
    cat logs/audio_deep_features/fusion_launcher/progress.tsv

After an interruption, first verify that no training process is running and
that the tmux session is absent or contains the existing launcher. Then rerun
the same --canary command. The launcher resolves the generated runtime
directory, skips a run only when train.log, summary.txt, a checkpoint, and
Training complete are all present, and stops on collisions or incomplete
evidence. It never starts a second copy of a completed canary.

## Canary invocation and runtime result

The exact configuration was:

    configs/audio_deep_features/native_turn_fusion/emotionmsp_embeds__trillsson4_1/mlp_fold0.yaml

The actual resolved runtime directory was:

    logs/audio_deep_features/audio_deep_features_native_turn_fusion_emotionmsp_embeds__trillsson4_1/native_turn_fusion_emotionmsp_embeds__trillsson4_1_carewsm_pooled_mlp_fold0_2026-11-05_17-21_carewsm_pooled_mlp_4c5040b8/

The runtime completed with MLflow status FINISHED, run ID
f9da32597cc44bec9dccbe67a9aa95db, and no active tmux or Chimera training
process remained. The launcher progress record is SKIPPED_COMPLETED when
re-invoked after completion; no second training run was started.

Runtime artifacts verified:

- train.log ends with Training complete;
- summary.txt contains 35 epoch rows;
- checkpoints/last.pt exists;
- selected checkpoint:
  checkpoints/epoch=28_val_mean_corpora_tasks_macro_recall=0.8293.pt;
- native counts: joint train 6998, CARE VAL 352, WSM VAL 1429, CARE TEST
  318, WSM TEST 1364;
- best validation monitor:
  val_mean_corpora_tasks/macro_recall = 0.829263 at epoch 28;
- early stopping occurred at epoch 35 with last validation value 0.822465.

The TEST aggregate at the final logged epoch was 0.750241; it was not used
for checkpoint selection or any configuration decision.

## Native fusion verification

The live DataModule and Dataset were read directly for the approved ordered
family list [emotionmsp_embeds, trillsson4_1]. Representative real samples
produced:

| Corpus/sample | Native shape | dtype | Component dimensions | Turn IDs |
|---|---:|---|---:|---|
| CARE CONTROL__CLEFT-LIP-PALATE__PROFILE_075__VIDEO_002 | [5, 2048] | torch.float32 | 1024 + 1024 | 5, equal and ordered |
| WSM DEPRESSION__DEPRESSION__eGHKmLtV0uo__eGHKmLtV0uo_003 | [1, 2048] | torch.float32 | 1024 + 1024 | 1, equal and ordered |

Both fused tensors were finite. The Dataset concatenates along channel axis 1
and preserves native T; no input-side pooling, interpolation, padding, or
resampling was used. CARE and WSM retain their observed variable/single-turn
semantics. The runtime uses the existing joint CARE+WSM fold-0 populations,
separate corpus validation/test loaders, and original observed-label masks.

## MLflow verification

The actual SQLite backend was inspected read-only at
sqlite:///logs/audio_deep_features/mlflow.db. The canary run contained:

- original val_care/... and val_wsm/... series;
- original test_care/... and test_wsm/... series;
- val_mean_corpora/<task>/<metric>;
- test_mean_corpora/<task>/<metric>;
- val_mean_corpora_tasks/<metric>;
- test_mean_corpora_tasks/<metric>.

Representative canonical histories had steps 1 through 35. The validation
task aggregate reached its maximum at step 28 with 0.829263; the TEST task
aggregate history was also present through step 35. No dev_mean_*, val_average/*
or test_average/* legacy MLflow metric series were present. The summary's
dev_mean_macro_recall value at epoch 28 was 0.8293, the rounded legacy alias
of the canonical validation monitor.

## Launcher validation and limitations

- bash -n scripts/run_audio_deep_fusion.sh: passed.
- The 110 YAML manifest count and lexical ordering check: passed.
- Canary re-entry detection: passed; the completed canary was skipped.
- Collision handling: passed; an isolated temporary second runtime-directory
  marker caused --canary to exit 1 before any training command, and the
  marker was removed.
- No remaining 109-run execution was attempted.

An early launcher version assumed that Chimera would honor the literal
run_name. The actual CLI appended a timestamp/model hash, so the first
successful process was conservatively recorded as incomplete by that version.
The launcher was corrected to resolve the generated suffix and was then
revalidated by skipping the completed canary. This did not trigger a second
training run and is retained in the runtime launcher log as an audit trail.

No SSH companion script was added because the Codex environment is already the
verified GPU host; an SSH hop would have been unnecessary and would have
required inventing no authorized connection details.

## Exact completed experiment count

**1 of 110** approved native-turn fusion experiments completed successfully.

This evidence contains no comparative performance claim and does not authorize
the remaining matrix.
