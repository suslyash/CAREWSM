# MantisV2 eGeMAPS Real TRAIN Pilot

Date: 2026-11-06
Required main: `ace53c1f8d95d38b1f4bd876c5312bca3cd1f728`
Branch: `codex/mantisv2-egemaps-real-train-pilot`

## 1. Decision

**CONDITIONAL GO for this bounded frozen-inference pilot only.** The pinned
MantisV2 checkpoint produced finite, reproducible `[num_windows,25,512]`
embeddings from real CARE and WSM fold-0 TRAIN eGeMAPS windows. This is not a
real cache builder, classifier experiment, predictive evaluation, or approval
for full-corpus inference.

## 2. Data boundary and selection

Only canonical `fold_0_train.csv` populations were used. No VAL, DEV, or TEST
files were read. Selection did not inspect disease labels: it used manifest
eGeMAPS validity, temporal integrity, and deterministic source-row order. The
pilot selected three artifacts from each corpus and retained at most two
accepted windows per artifact.

| Corpus | Fold-0 TRAIN population | Artifacts selected | Windows accepted | Policy C frames excluded |
|---|---:|---:|---:|---:|
| CARE | 1361 | 3 | 6 | 1 |
| WSM | 5640 | 3 | 6 | 0 |
| Total | 7001 | 6 | 12 | 1 |

The selected artifacts contained 23,991 CARE source frame rows and 17,222 WSM
source frame rows. The selection scan examined 92 CARE and 3 WSM candidate rows
before reaching the bounded selection. No identifiers, labels, participant
metadata, or feature values were printed or saved.

The single CARE exclusion reason was
`excluded_terminal_out_of_turn_1sample`; WSM had no exclusions.

## 3. Temporalization and interpolation

The authoritative 25-channel registry order was verified. The script applied
Policy C in memory using the accepted one-sample terminal-frame rule. The
excluded CARE frame was represented by an explicit exclusion count/reason in
memory and was not retained for window construction. Source CSVs and timestamps
were unchanged.

Windows were one second with a 0.5-second hop, independently within turns. A
frame was eligible only when its complete interval was inside both the window
and declared turn. Each accepted window required at least 95 consecutive valid
frame starts at the verified 10 ms cadence, with no missing interior cadence
position. The pilot therefore uses a stricter frame-containment/consecutive-run
rule than the validator's descriptive interval-union coverage convention.

Each accepted window was linearly interpolated in float32 from actual observed
frame-start times to 512 points per channel. The target grid was bounded by the
first and last observed source frame starts; no extrapolation or fabricated
frame was used. The 512 points are a deterministic regridding, not 512
independent source observations. No normalization, PCA, feature selection, or
learned channel adapter was applied.

## 4. Checkpoint and environment

- Package: `mantis-tsfm==1.1.0`
- Model: `paris-noah/MantisV2`
- Revision: `99fe0f548960e272fbfa4b82fd9b5b5956779dfd`
- `model.safetensors` SHA-256:
  `49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1`
- Configuration: `return_transf_layer=2`, `output_token='combined'`
- Device: NVIDIA GeForce RTX 4080, CUDA runtime 12.8
- Peak CUDA allocated memory: 73,448,960 bytes
- Peak CUDA reserved memory: 88,080,384 bytes

The official `MantisTrainer.transform(three_dim=True)` processed channels
independently. No direct 25-channel model call, channel averaging, or learned
fusion was used.

## 5. Inference verification

| Check | Result |
|---|---|
| Synthetic temporalization fixture | PASS |
| Real batched input | `[12,25,512]` float32 |
| Real batched output | `[12,25,512]` |
| Finite input/output values | PASS |
| Repeated frozen inference identical | PASS |
| Parameters unchanged | PASS |
| Gradients absent | PASS |
| Policy C frame absent from input | PASS |
| Python compilation | PASS |
| `git diff --check` | PASS |

No real embedding cache or intermediate tensor file was written. The command
printed aggregate JSON only.

## 6. Reproduction

```bash
PYTHONPATH=/media/maxim/Programs/Projects/CAREWSM/.venv/lib/python3.12/site-packages \
  /tmp/mantisv2-isolated/bin/python \
  scripts/smoke_mantisv2_egemaps_train.py \
  --checkpoint /tmp/mantisv2-checkpoint
```

The checkpoint hash, registry channel count, temporal integrity, interpolation
bounds, tensor shapes, finiteness, determinism, parameter immutability, and
gradient absence are asserted by the script.

## 7. Limitations

This is six real artifacts and twelve windows, not a representative model
evaluation. Disease labels were not used, and no disease prediction metric was
calculated. The result does not establish predictive benefit, generalization,
normalization suitability, or scientific equivalence of interpolation to new
measurements. Full-corpus temporal caching remains a separate authorization
gate.

## 8. Next gate

Before any expanded inference, obtain approval for the explicit channel-wise
cache schema, preprocessing contract, storage budget, and coverage policy. No
classifier training, threshold selection, MLflow logging, or TEST evaluation is
authorized by this pilot.
