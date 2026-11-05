# CLEANUP-001 TSCMamba Cleanup and CUDA Dependency

## Outcome

The requested minimal TSCMamba cleanup is complete. `mamba-ssm` 2.3.2.post1 was installed in the existing `.venv` with CUDA selective-scan enabled. Native Mamba and the registered `carewsm_tscmamba` model both passed synthetic CUDA forward/backward checks.

No CARE/WSM data, training epoch, experiment matrix, benchmark, or TASK-011 run was performed.

## Source changes

- Required `sequence_mask`; missing input now raises a clear `ValueError`.
- Removed the `native_mask` and all-ones mask fallbacks.
- Removed unused TSCMamba `mamba_required` and `num_heads` parameters and forwarding.
- Removed constant `encoder_type`, unreachable Transformer branch, and `safe_valid_mask()`.
- Preserved the Mamba temporal/spectral path, bidirectional blocks, and padding masking behavior.
- MLP, MD-Former, MPTSNet, configurations, and research hyperparameters were unchanged.

## Environment and installation

Before installation:

- Python 3.12.3
- PyTorch 2.11.0+cu128; PyTorch CUDA 12.8
- CUDA available: true; GPU: NVIDIA GeForce RTX 4080; driver 580.178.04; 16,376 MiB
- `nvcc` was not on `PATH`; the CUDA compiler at `/usr/local/cuda/bin/nvcc` was used by the build
- `mamba_ssm` was not importable
- Relevant packages included `triton 3.6.0`, `einops 0.8.2`, and `setuptools 78.1.0`

Installation command:

```text
MAMBA_KEEP_CUDA_BUILD=TRUE .venv/bin/python -m pip install mamba-ssm==2.3.2.post1 --no-build-isolation
```

After installation:

- `mamba-ssm 2.3.2.post1`
- PyTorch remained `2.11.0+cu128`; PyTorch CUDA remained `12.8`
- NumPy changed from `1.26.4` to `2.5.4` as an installed package dependency
- The build produced a 284,304,754-byte Linux CPython 3.12 wheel with the CUDA selective-scan build enabled

The installation followed the official Mamba instructions: https://github.com/state-spaces/mamba

## Verification

- `from mamba_ssm.modules.mamba_simple import Mamba`: passed.
- Native Mamba, input `(2, 8, 8)`: finite forward output and successful finite backward pass on CUDA.
- Registered `carewsm_tscmamba`, output `(2, 2)`: finite forward output and successful backward pass on CUDA with explicit `sequence_mask`.
- Missing `sequence_mask`: clear `ValueError` passed.
- Focused regression suite: **18 passed, 5 warnings**.
- `bash scripts/run_audio_experiments.sh --check --all`: all 110 canonical configurations passed.
- Python compilation and `git diff --check`: passed.

The remaining warnings were existing/dependency warnings; no dependency blocker remains for the verified TSCMamba CUDA path.
