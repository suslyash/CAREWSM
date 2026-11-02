# TASK-004 model infrastructure evidence

Implemented and registered three native sequence classifier families: TSCMamba (FFT spectral view plus required `mamba_ssm` SSM blocks), MPTSNet (FFT period cue, three local convolution scales and GRU global dependency path), and MD-Former (dual temporal/cross-channel Transformer branches with late fusion). All use variable-length `inputs["native"]`, sequence masks, a shared representation, separate Depression/Parkinson layers and two logits. Pooled MLP remains a reference.

The focused smoke parses `configs/audio/task004_models.yaml`, registers factories through `src/chimera_plugin.py`, and runs forward/masked BCE/backward on a synthetic padded batch. It checks finite logits and gradients and explicitly reports TSCMamba as BLOCKED when `mamba_ssm` is unavailable; no Transformer substitution is made. No training, Mantis download, TEST selection or results are claimed.
