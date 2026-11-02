# TASK-004 model infrastructure evidence

Implemented and explicitly registered three native sequence classifier families: TSCMamba (FFT spectral view plus legacy-derived mask-safe BiMamba/TemporalEncoder blocks requiring `mamba_ssm`), MPTSNet (FFT period selection, period-folded multiscale Conv2d branches and GRU global dependency path), and MD-Former (independent temporal and cross-channel Transformer branches with late fusion). All use variable-length `inputs["native"]`, sequence masks, a shared representation, separate Depression/Parkinson layers and two logits. Pooled MLP remains a reference. The joint DataModule, masked BCE loss, and each of the six metrics use explicit Chimera-ML decorators and are imported by `chimera_plugin.py`.

The focused smoke parses `configs/audio/task004_models.yaml`, registers components through `src/chimera_plugin.py`, and runs forward/masked BCE/backward on a synthetic padded batch. A short fold-0 CARE+WSM batch from `carewsm_joint` (two samples, first 32 timesteps) was loaded and passed through MPTSNet and MD-Former with finite logits/gradients. All six registered stateful metrics were created and updated through the Chimera-ML `BaseMetric` interface. TSCMamba explicitly reports BLOCKED when `mamba_ssm` is unavailable; no Transformer substitution is made. No training, Mantis download, TEST selection or results are claimed.


The final smoke also checks all-padding/partially padded inputs for finite outputs. Padding values are excluded from masked pooling and the MD-Former cross-channel branch.
