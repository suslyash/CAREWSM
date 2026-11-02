# CAREWSM Recent Temporal Model Shortlist

The audio programme starts with three complementary 2025 time-series classification architectures. They are chosen because the project contains heterogeneous temporal acoustic descriptors and the owner explicitly wants temporal modelling.

These are architectural references. CAREWSM implementations must be clean Chimera-ML adaptations to the audited feature contract, not unverifiable copies.

## 1. TSCMamba-inspired model

Paper: Md Atik Ahamed, Qiang Cheng, **TSCMamba: Mamba Meets Multi-View Learning for Time Series Classification**, Information Fusion 120 (2025), 103079.

DOI: https://doi.org/10.1016/j.inffus.2025.103079

Why it fits:

- direct multivariate time-series classification;
- combines temporal and spectral views;
- state-space/Mamba sequence modelling is attractive for long acoustic trajectories;
- reported strong performance across 30 benchmark datasets.

CAREWSM adaptation:

- family-specific input projection;
- masked variable-length temporal sequence;
- temporal + spectral view where the feature semantics support it;
- masked pooling / classification into two binary disease heads;
- static feature tokens enter only through an explicit adapter.

## 2. MPTSNet-inspired model

Paper: Y. Mu, M. Shahzad, X. X. Zhu, **MPTSNet: Integrating Multiscale Periodic Local Patterns and Global Dependencies for Multivariate Time Series Classification**, AAAI 2025.

Paper: https://ojs.aaai.org/index.php/AAAI/article/view/34155

Why it fits:

- direct MTSC architecture;
- Fourier analysis identifies dominant periods;
- multiscale periodic blocks combine CNN local pattern extraction with attention for global dependencies;
- reported to outperform 21 advanced baselines on UEA MTSC benchmarks.

CAREWSM adaptation:

- use only for truly temporal families;
- robust masked/padded implementation;
- family projection before periodic blocks;
- standardized pooled representation for feature-family fusion.

## 3. MD-Former-inspired model

Paper: Y. Du et al., **MD-Former: Multiscale Dual Branch Transformer for Multivariate Time Series Classification**, Sensors 25(5), 1487 (2025).

DOI: https://doi.org/10.3390/s25051487

Why it fits:

- direct MTSC;
- explicitly models multiscale temporal and cross-channel relations;
- Transformer branch gives a different inductive bias from Mamba and periodic CNN/attention.

CAREWSM adaptation:

- temporal branch models time;
- channel/family branch is adapted carefully because CAREWSM channels are acoustic descriptors, not arbitrary sensor axes;
- masks are mandatory;
- static families use separate tokens rather than fake repeated timelines.

## Reference baseline

A small pooled MLP is retained only to answer:

“Does the temporal model outperform a simple aggregate representation?”

It is not a fourth SOTA candidate.

## Replacement rule

Do not replace these three major model families during routine implementation.

A replacement requires manager review and owner approval, ideally based on one of:

- model cannot be adapted faithfully to variable-length acoustic features;
- unacceptable compute/runtime;
- a clearly stronger 2025–2026 architecture with available methodological details/code and a better fit to the audited feature contract.
