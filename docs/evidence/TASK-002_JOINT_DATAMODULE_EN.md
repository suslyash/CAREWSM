# TASK-002 Joint DataModule evidence

The data-only layer uses `(corpus, sample_id)` as key, joins each fold directly from canonical fold CSVs, and keeps DEV metadata-only. It exposes joint TRAIN/VAL, corpus-specific VAL, and separate fixed CARE/WSM TEST views. It does not fit preprocessing, calculate class weights, train models, or temporalize features.

Run `PYTHONPATH=. .venv/bin/python scripts/smoke_joint_datamodule.py`. Real smoke counts after `wav2vec_xlsr300m` availability filtering were fold 0 `6998/1781`, fold 1 `7027/1752`, fold 2 `7045/1734`, fold 3 `7020/1759`, fold 4 `7026/1753` (TRAIN/VAL); unfiltered canonical counts remain 7001/1781, 7029/1753, 7048/1734, 7022/1760, 7028/1754. The smoke also exercised `egemaps_func + wav2vec_xlsr300m` (6998/1780) and native CSV loading with shape `(1, 88)` plus turn metadata. Fixed TEST views are CARE 318 and WSM 1364 before filtering.

The module rejects duplicate/missing manifest keys and profile mismatches, preserves explicit observed masks, and filters only requested feature sets. No scaler, PCA, Mantis, or fake time axis is implemented. CARE's missing `splits/videos.csv` is not required.
