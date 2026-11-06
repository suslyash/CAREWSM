# DEEP-008 — Stronger Pooling, Gated Fusion, and MantisV2 Pilot

Date: 2026-11-06  
Required base: `7b6dcc7c43535b1a226f4c7ebe73caf3b190869c`  
Branch: `codex/deep-008-representation-pilot`

## Isolated experiment series

DEEP-008 is isolated from the historical `audio_deep_features` series. The six
fold-0 configurations now live under
`configs/audio_representation_pilot/deep008/`. They use experiment name
`audio_representation_pilot_deep008`, filesystem roots
`logs/audio_representation_pilot`, and MLflow URI
`sqlite:///logs/audio_representation_pilot/mlflow.db`. No historical logs or
MLflow database were migrated or modified.

The original `src/audio/models/carewsm_audio_models.py` matches the required
main-base content byte-for-byte. New registrations are isolated in
`src/audio/models/carewsm_representation_pilot_models.py` and imported by
`src/chimera_plugin.py` beside the original module. The pilot module reuses the
original `_unpack`, `CAREWSMSequenceModel`, and `TaskHeads` abstractions.

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

The gated arm now uses exactly the ordered native families:

```yaml
feature_family:
  - emotionmsp_embeds
  - trillsson4_1
```

Its input is native turn-aligned `[T, 2048]`, with separate 1024-channel
projection and learnable gate per family. Gates are applied at every native
turn, then valid turns are mask-averaged. Variable-length sequences remain
intact; no first-turn truncation, interpolation, input-side pooling, or fake
timestamps are used. This replaces the previous draft's incorrect four-family
static-mean design.

## Verification

- Original model module matches the required main-base file exactly.
- No DEEP-008 YAML remains under `configs/audio_deep_features`.
- All six new YAMLs parse and use only the new filesystem and MLflow roots.
- New registration names are `carewsm_attentive_mlp` and
  `carewsm_gated_fusion`; the plugin registration guard remains idempotent.
- Python compilation and `git diff --check` pass.
- No training, GPU run, checkpoint creation, TEST evaluation, or dependency
  installation was performed.
- Runtime pytest and native Chimera validation remain blocked in this checkout
  because system Python lacks PyTorch, pytest, and Chimera-ML.

No performance claim is made. Existing DEEP-003 through DEEP-007 experiments and
results remain unchanged.
