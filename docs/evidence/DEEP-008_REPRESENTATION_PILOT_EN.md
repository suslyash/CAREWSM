# DEEP-008 — Stronger Pooling, Gated Fusion, and MantisV2 Pilot

Date: 2026-11-06  
Required base: `7b6dcc7c43535b1a226f4c7ebe73caf3b190869c`  
Branch: `codex/deep-008-representation-pilot`

## Outcome

Prepared six fold-0 pilot configurations on the strongest prior single family,
`emotionmsp_embeds`, without changing native artifacts, fixed splits, labels,
losses, or prior DEEP-001–DEEP-007 evidence. The arms are deliberately small:

| Arm | Representation/model | Width |
|---|---|---:|
| A | Existing native pooled MLP | 64 |
| B | Input-side masked turn mean, MLP | 64 |
| C | Input-side turn mean + standard deviation, MLP | 64 |
| D | Learned mask-safe attentive pooling, MLP | 64 |
| E | Existing native pooled MLP capacity control | 128 |
| F | Four-family static mean + learned gated fusion | 64 |

A–E are the direct `emotionmsp_embeds` comparison. F is a separate gated-fusion
pilot using the audited ordered families and dimensions
`[1024, 3, 1024, 1024]`. All arms retain seed 5, 50 maximum epochs, patience 7,
masked multitask BCE, validation-only checkpoint selection, and fold 0.

## Implementation

The Dataset/DataModule now support explicit `pooling: native | mean | mean_std`.
`mean` and `mean_std` reduce only the observed native turns at input time; they
do not fabricate timestamps. `mean_std` concatenates the per-channel turn mean
and standard deviation. The new attentive pooler applies a learned scalar score
and sequence mask before softmax. The gated model projects each static family
separately and learns a per-family sigmoid gate before summation.

The configurations are under `configs/audio_deep_features/deep008_pilot/`.
No DEEP-003–DEEP-007 configuration or result was modified.

## MantisV2 feasibility

The official Mantis documentation says inputs must be time series with sequence
length a multiple of 32 and recommends resizing to 512; its frozen feature
recipe uses an intermediate transformer layer and `output_token='combined'`.
The official MantisV2 recommendation is layer index 2. See:

- https://github.com/vfeofanov/mantis
- https://github.com/vfeofanov/mantis/blob/main/src/mantis/architecture/version2.py
- https://arxiv.org/abs/2602.17868
- https://huggingface.co/paris-noah/MantisV2

This pilot did not download the checkpoint or install `mantis-tsfm`; therefore
there is no downloaded revision or package installation to pin. The feasibility
candidate is `MantisV2` from `paris-noah/MantisV2`, layer 2, combined output,
but it is not scientifically admissible for the current NPZ families without a
manager-approved time-axis contract: the audited embedding artifacts expose
ordered turn IDs but no timestamps, and WSM is single-turn in the live data.
Repeating or interpolating those turns into 1-second windows would violate the
representation contract. No Mantis-derived result is claimed.

## Verification

- Required base was fetched and the branch was created from the exact commit.
- Six YAML files parse successfully; each retains seed 5 and a 50-epoch budget.
- Python compilation passed for the three touched modules.
- `git diff --check` passed.
- Synthetic Torch forward checks were not runnable because the available system
  Python has no `torch`; pytest and Chimera-ML were likewise unavailable.
- No real training, checkpoint download, feature extraction, split generation,
  or TEST-based decision was performed.

## Limits

This is configuration and implementation preparation, not evidence of a
validation improvement. The pilot does not guarantee 0.82–0.85 Macro Recall.
The arms must be executed in a compatible Chimera/PyTorch environment, and any
selection must use `val_mean_corpora_tasks/macro_recall` only.
