# MantisV2 Isolated Synthetic Smoke Test

Date: 2026-11-06
Required main: `48905953e559c112ee6b83eea5d303e32900a15b`
Branch: `codex/mantisv2-isolated-synthetic-smoke`

## 1. Scope and decision

**CONDITIONAL GO for isolated synthetic inference only.** The authentic pinned
MantisV2 checkpoint loaded and produced finite, deterministic embeddings on CPU
and CUDA. This does not establish CARE/WSM preprocessing correctness, predictive
benefit, real-data compatibility, or permission to generate real caches.

## 2. Environment

The repository `.venv` was not modified. A separate environment was created at
`/tmp/mantisv2-isolated`; it exposes the existing `.venv` Torch installation
read-only through `PYTHONPATH` so that a second 539 MB CUDA Torch wheel was not
installed. Mantis and its non-Torch dependencies were installed only in the
isolated environment.

- Python: 3.12.3
- Torch: 2.11.0+cu128
- CUDA runtime reported by Torch: 12.8
- GPU: NVIDIA GeForce RTX 4080, driver 580.178.04, 16,376 MiB
- `mantis-tsfm`: 1.1.0
- `einops`: 0.8.2
- `datasets`: 5.1.0
- `numpy`: 2.5.4
- `pandas`: 2.3.3
- `safetensors`: 0.8.0
- `scikit-learn`: 1.9.1
- `scipy`: 1.17.1

## 3. Checkpoint

- Repository: `paris-noah/MantisV2`
- Required revision: `99fe0f548960e272fbfa4b82fd9b5b5956779dfd`
- `model.safetensors` SHA-256:
  `49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1`
- `config.json` SHA-256:
  `f28885b7aa662cedcaf5627659fc0fba9cf8ee49d5b991280522e04fa5b1ae80`
- `.gitattributes` SHA-256:
  `11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361`

The checkpoint files are stored only under `/tmp` and are not tracked by Git.

## 4. Synthetic test configuration and outcomes

The script uses `return_transf_layer=2`, `output_token='combined'`, `eval()`,
`torch.no_grad()`, and frozen parameters. It generated `[2,25,512]` tensors
with seed 5 and tested CPU plus the available CUDA device.

| Check | CPU | CUDA |
|---|---|---|
| Checkpoint load | PASS | PASS |
| Direct `[2,25,512]` model call | Correctly rejected | Correctly rejected |
| Single-channel input | `[2,1,512]` | `[2,1,512]` |
| Single-channel output | `[2,512]` | `[2,512]` |
| Official trainer multichannel output | `[2,25,512]` | `[2,25,512]` |
| Finite outputs | PASS | PASS |
| Repeated inference identical | PASS | PASS |
| Parameters unchanged / no gradients | PASS | PASS |

## 5. Channel and interface finding

The direct `MantisV2` convolution has one input channel and rejects a raw
25-channel tensor. The official `MantisTrainer.transform()` path handles
multichannel input by sending each channel independently through the frozen
network and returning one 512-dimensional embedding per channel. It does not
learn or perform cross-channel fusion. A future CARE/WSM implementation must
therefore explicitly choose and document this channel-wise representation or a
separately authorized adapter. No adapter, PCA, normalization, resampling, or
real feature processing was implemented here.

## 6. Reproduction

With the checkpoint in `/tmp/mantisv2-checkpoint` and the isolated environment:

```bash
PYTHONPATH=/media/maxim/Programs/Projects/CAREWSM/.venv/lib/python3.12/site-packages \
  /tmp/mantisv2-isolated/bin/python \
  scripts/smoke_mantisv2_synthetic.py \
  --checkpoint /tmp/mantisv2-checkpoint
```

The smoke script verifies the checkpoint hash before loading and fails on any
unexpected shape, nonfinite result, nondeterminism, gradient, or parameter
mutation.

## 7. Restrictions and limitations

No CARE/WSM files, splits, feature artifacts, TEST data, audio, temporal cache,
Mantis real-data inference, training, MLflow database, or temporal-validator
rule was accessed or modified. Synthetic shape compatibility does not prove that
Policy C eGeMAPS windows are correctly normalized or scientifically suitable for
MantisV2, and no predictive or clinical conclusion can be drawn.

## 8. Next gate

Before any real-data operation, obtain separate authorization for the channel
contract, preprocessing, and cache builder. The next bounded technical step is
synthetic tensor testing of that explicitly selected channel interface only.
