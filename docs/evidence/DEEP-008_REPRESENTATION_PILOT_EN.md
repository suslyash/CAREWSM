# DEEP-008 — Stronger Pooling, Gated Fusion, and MantisV2 Pilot

Date: 2026-11-06
Acceptance base commit: `72216ebab5db9af592365e5fe935538bed8839bb`
Branch: `codex/deep-008-representation-pilot`

## Isolated experiment series

DEEP-008 is isolated from the historical `audio_deep_features` series. The six
fold-0 configurations live under `configs/audio_representation_pilot/deep008/`.
They use experiment name `audio_representation_pilot_deep008`, filesystem roots
`logs/audio_representation_pilot`, and MLflow URI
`sqlite:///logs/audio_representation_pilot/mlflow.db`. No historical logs or
MLflow database were migrated or modified.

The original `src/audio/models/carewsm_audio_models.py` matches the required
main-base content byte-for-byte. New registrations are isolated in
`src/audio/models/carewsm_representation_pilot_models.py` and imported by
`src/chimera_plugin.py` beside the original module. The pilot module reuses the
historical projection/head abstractions through lazy resolution to avoid an
entry-point import cycle.

## Configurations

- `A_native_mlp64.yaml`
- `B_mean_mlp64.yaml`
- `C_mean_std_mlp64.yaml`
- `D_attentive_mlp64.yaml`
- `E_native_mlp128.yaml`
- `F_gated_emotion_trillsson_native64.yaml`

All retain fold 0, seed 5, 50 epochs, patience 7, masked multitask BCE, fixed
TRAIN/VAL/TEST populations, six approved metrics, and the validation-only
`val_mean_corpora_tasks/macro_recall` monitor.

## Corrected gated fusion

The gated arm uses exactly the ordered native families:

```yaml
feature_family:
  - emotionmsp_embeds
  - trillsson4_1
```

Its input is native turn-aligned `[T, 2048]`, with separate 1024-channel
projection and learnable gate per family. Gates are applied at every native
turn, then valid turns are mask-averaged. Variable-length sequences remain
intact; no first-turn truncation, interpolation, input-side pooling, or fake
timestamps are used.

## Runtime acceptance

Environment used:

- GPU: NVIDIA GeForce RTX 4080, compute capability 8.9
- PyTorch: `2.11.0+cu128`
- CUDA runtime: `12.8`
- Chimera-ML: `0.3.0`

Native `chimera-ml validate-config` passed for all six configurations:

- A: PASS
- B: PASS
- C: PASS
- D: PASS
- E: PASS
- F: PASS

The CUDA smoke script passed for both new models. It covered T=1, T=3, and T=5
variable-length batches, finite `[B,2]` logits, backward passes, finite
parameter/input gradients, zero gradients on padded positions, evaluation-mode
padding invariance, both gated family projections and gates, and mean/mean_std
pooling dimensions for single-turn and variable-length inputs. Registration was
loaded twice successfully without the prior circular-import warning.

The existing `tests/test_deep008_representation_pilot.py` passed: **6 passed**
with three unrelated dependency deprecation warnings.

## Correction made during acceptance

The first CUDA smoke exposed a circular-import warning during Chimera entry-point
loading: the pilot module imported the historical model module while it was
still initializing. The minimal fix moved those shared imports to lazy helper
resolution inside the isolated module. The historical model module was not
changed and remains byte-identical to the required base.

## Preservation and limits

No training, checkpoint creation, TEST evaluation, dependency installation, or
research-result modification was performed. No performance claim is made.
Historical DEEP-003 through DEEP-007 configurations, outputs, MLflow databases,
and results remain preserved.

**Final acceptance: PASS.**
