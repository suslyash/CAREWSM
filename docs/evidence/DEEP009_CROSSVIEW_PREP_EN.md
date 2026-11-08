# DEEP-009 CrossView CPU-Only Preparation Evidence

Date: 2026-11-08
Base: `71a47ff4033d63f13c985bbf322094ba9aee3695`
Scope: implementation and reproducibility preparation only

## 1. Outcome

Implemented the opt-in `carewsm_crossview` Chimera-ML model and prepared five
isolated fold configurations. The historical `carewsm_gated_fusion` model,
losses, metrics, Dataset, DataModule, callbacks and Trainer interfaces were not
changed. No GPU training, real-data inference, TEST access, audio access,
feature extraction, model download or additional architecture was used.

The CrossView model keeps the existing per-family LayerNorm/linear/GELU
projections, gates, masked mean pooling, task heads and `[Depression,
Parkinson]` output order. It adds only the approved rank-16 interaction:

```text
e = projection_emotion(emotion)
t = projection_trillsson(trillsson)
base = sigmoid(gate_emotion(emotion)) * e
      + sigmoid(gate_trillsson(trillsson)) * t
cross = W_out(U_e(e) * U_t(t))
fused = base + alpha * cross
```

`U_e`, `U_t` and `W_out` are bias-free linear layers with Xavier initialization;
`alpha` is a learnable scalar initialized to `0.1`. Rows with no valid turns
raise a clear error. Padded turns are excluded from pooling.

## 2. Changed implementation

- `src/audio/models/carewsm_representation_pilot_models.py`
  - Added `CAREWSMCrossViewModel`.
  - Added explicit `@MODELS.register("carewsm_crossview")` factory.
  - Preserved the existing gated-fusion class unchanged.
- `tests/test_deep009_crossview_preparation.py`
  - Added synthetic CPU tests for registry resolution, exact parameter count,
    T=1, variable-length padding invariance, all-invalid rejection,
    missing-label masking, gradients, CPU bfloat16 autocast, baseline forward
    contract and YAML parity.
- `scripts/run_audio_deep009_crossview.sh`
  - Added a sequential five-fold launcher following the existing conventions.
  - `--check` validates all configs and collision gates without launching
    training; the training path was not executed.
- `configs/audio_representation_pilot/deep009/crossview/`
  - Added exactly five fold configurations.

## 3. Model accounting

At hidden dimension 64, the interaction branch contains exactly:

| Component | Parameters |
|---|---:|
| `U_e`, 64 → 16, no bias | 1,024 |
| `U_t`, 64 → 16, no bias | 1,024 |
| `W_out`, 16 → 64, no bias | 1,024 |
| `alpha` | 1 |
| **Interaction total** | **3,073** |

The complete CrossView model contains exactly **278,275 trainable parameters**.
The first pilot uses fixed rank 16 and does not introduce attention,
convolution, recurrence, positional encoding, corpus-specific heads, a third
feature family or a new loss.

## 4. Prepared five-fold experiments

All configurations use joint CARE+WSM training, native aligned
`emotionmsp_embeds` + `trillsson4_1`, fixed fold isolation, seed 5, hidden size
64, dropout 0.1, batch size 64, AdamW with learning rate `0.0001` and weight
decay `0`, weighted Parkinson BCE, maximum 50 epochs, patience 7, mixed
precision, gradient clipping and the canonical
`val_mean_corpora_tasks/macro_recall` monitor.

| Fold | Configuration | Run name | Parkinson `pos_weight` |
|---:|---|---|---:|
| 0 | `F_crossview_fold0.yaml` | `F_crossview_fold0` | 2.358878504672897 |
| 1 | `F_crossview_fold1.yaml` | `F_crossview_fold1` | 2.375234521575985 |
| 2 | `F_crossview_fold2.yaml` | `F_crossview_fold2` | 2.3653308480894686 |
| 3 | `F_crossview_fold3.yaml` | `F_crossview_fold3` | 2.4134980988593155 |
| 4 | `F_crossview_fold4.yaml` | `F_crossview_fold4` | 2.3594771241830066 |

Artifacts are isolated under:

```text
logs/audio_representation_pilot/deep009_crossview/
```

The five MLflow runs use the dedicated experiment
`audio_representation_pilot_deep009_crossview` and the dedicated SQLite URI
`sqlite:///logs/audio_representation_pilot/deep009_crossview/mlflow.db`.

## 5. CPU verification evidence

Passed checks:

- `9 passed` focused CrossView synthetic CPU tests.
- Exact interaction and complete-model parameter counts.
- T=1 finite forward and `[B,2]` output.
- T>1 padding-value invariance.
- All-invalid sequence rejection.
- Missing-label loss masking.
- Finite backward gradients through both interaction projections, residual
  scale and both task heads.
- CPU bfloat16 autocast forward/backward.
- Historical gated-fusion deterministic forward contract.
- Five-fold YAML parity and registry resolution.
- Launcher `--check`: `Preflight passed; no training was launched.`
- Historical regression suites: `30 passed` across representation-pilot,
  task-attention preparation and Batch 02 preparation tests.
- Five explicit Chimera config validations passed.
- Python compilation passed for the touched Python files.
- `bash -n scripts/run_audio_deep009_crossview.sh` passed.
- `git diff --check` passed after staging.

Warnings were limited to existing third-party deprecation/plugin-load warnings;
no test failed because of them.

## 6. Frozen future VAL decision

The future comparison is CrossView versus the historical weighted-F gated-fusion
control. It must retain the same five joint CARE+WSM folds, participant
isolation, features, seed, optimizer, batch size, Parkinson TRAIN-derived
weights, loss, training budget, callbacks, metrics and checkpoint monitor.

Accept CrossView only if all conditions hold over the five paired folds:

1. mean VAL UAR improves by at least `+0.010` over weighted-F;
2. no CARE/WSM × task component mean decreases by more than `0.020` absolute;
3. at least three of five paired folds improve; and
4. all five runs have valid completion evidence and all required metrics,
   including Patient and Healthy Recall.

TEST remains descriptive only and cannot select the architecture, rank,
initialization, checkpoint, threshold or continuation. This preparation does
not claim any classification improvement.

## 7. Hard stop and limitations

The launcher’s future training path remains opt-in and was not executed. The
configuration `train.device: cuda` is preserved for protocol parity, but this
task performed no CUDA operation. The CPU `--check` path stops before the GPU
check and training command.

CrossView tests within-turn complementarity. It does not create temporal
structure for WSM, whose valid records are single-turn, and it does not invent
timestamps or acoustic frames.
