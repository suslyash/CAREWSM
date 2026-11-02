# CAREWSM Temporal Model Shortlist

The project uses a frozen Mantis-family time-series encoder as a **derived feature representation** where the audited feature block has a defensible temporal axis.

Mantis itself is not counted as one of the three downstream disease-classifier families.

The exact Mantis checkpoint/version/layer/pooling is frozen in a dedicated post-audit task. Do not silently switch between Mantis variants.

## 1. TSCMamba-inspired classifier

Reference:

Md Atik Ahamed, Qiang Cheng, **TSCMamba: Mamba Meets Multi-View Learning for Time Series Classification**, Information Fusion 120 (2025), 103079.

DOI:

https://doi.org/10.1016/j.inffus.2025.103079

Role:

- sequence classifier for native temporal features and Mantis window embeddings;
- temporal/state-space inductive bias;
- spectral view used only where meaningful for the audited representation.

CAREWSM adaptation:

- explicit masks;
- variable-length handling;
- common trunk;
- separate Depression/Parkinson feature layers and heads.

## 2. MPTSNet-inspired classifier

Reference:

Y. Mu, M. Shahzad, X. X. Zhu, **MPTSNet: Integrating Multiscale Periodic Local Patterns and Global Dependencies for Multivariate Time Series Classification**, AAAI 2025.

Paper:

https://ojs.aaai.org/index.php/AAAI/article/view/34155

Role:

- multiscale periodic/local pattern alternative;
- complementary inductive bias to Mamba;
- useful for temporal descriptors and Mantis window sequences.

CAREWSM adaptation:

- mask-safe padded batching;
- no assumption that every feature family has identical sampling rate;
- family projection before temporal blocks;
- task-specific output branches.

## 3. MD-Former-inspired classifier

Reference:

Y. Du et al., **MD-Former: Multiscale Dual Branch Transformer for Multivariate Time Series Classification**, Sensors 25(5), 1487 (2025).

DOI:

https://doi.org/10.3390/s25051487

Role:

- Transformer-based comparison;
- multiscale temporal and cross-channel modelling;
- prevents the study from becoming only Mamba variants.

CAREWSM adaptation:

- masks mandatory;
- audited feature channels only;
- no fake repeated timeline for static vectors;
- task-specific output branches.

## Mantis representation reference

Mantis is an open-source foundation model family for time-series classification/representation learning. The 2026 ICML publication describes frozen feature extraction and fine-tuning; later Mantis-family work also studies intermediate-layer representations and aggregation.

Official implementation:

https://github.com/vfeofanov/mantis

The project must pin the exact checkpoint/version before feature derivation.

## Reference baseline

A small pooled MLP/reference model is allowed to answer:

“Does temporal modelling outperform a simple aggregate representation?”

It is not a fourth SOTA model.

## Replacement rule

Replacing a major model family requires manager review and owner approval.

Valid reasons include:

- incompatible with audited representations;
- unacceptable compute/runtime;
- implementation cannot be reproduced faithfully;
- a clearly stronger 2025–2026 architecture has become reproducibly available and materially better fits the project.
