# TASK-003 Dual representation evidence

## Outcome

Implemented paired `native` and timestamp-respecting window adapters. Native mode preserves the audited sequence and metadata. Mantis preparation uses only real frame/segment timestamps, keeps turn boundaries, and emits ordered 1.0 s windows at 0.5 s hop with valid-observation fractions. Static and turn-embedding families remain native-only.

## Eligible families

`egemaps_lld` is verified on real CARE and WSM data: frame timestamps are present at approximately 50 Hz, with turn IDs and turn start/end bounds. The audit also identifies timestamped dynamic/trajectory CSV families as candidates; static/turn-level DisVoice, XLS-R, TRILLsson and emotion embeddings are excluded from Mantis mode because repeating them would fabricate time.

## Official Mantis compatibility

The official implementation documents inputs whose sequence length is a multiple of 32, commonly resized to 512, and frozen extraction from a pinned intermediate layer. The project environment has no `mantis-tsfm` installation (`import mantis` fails). Therefore this task is **BLOCKED for frozen Mantis inference**: no checkpoint, revision, layer or pooling was guessed, no unsuitable encoder was run, and no Mantis cache was generated. Installing and pinning the official package/checkpoint requires manager review before continuing.

## Real-data smoke

Command:

`PYTHONPATH=. .venv/bin/python scripts/smoke_dual_representation.py`

Observed fold-0 TRAIN samples:

| corpus | sample | native shape | window shape | first/last window |
|---|---|---:|---:|---|
| CARE | `CONTROL__ASTHMA__PROFILE_054__VIDEO_000` | `(1,12190,25)` | `(1,242,25)` | `9.113–10.113` / `129.613–130.613`, turn 1 |
| WSM | `CONTROL__DEPRESSION__0efS0axfX0E__0efS0axfX0E_001` | `(1,6120,25)` | `(1,121,25)` | `0–1` / `60–61`, turn 1 |

The smoke uses the same `(corpus, sample_id)` for both modes, checks ordered windows, first/last boundaries and non-empty masks. No training, TEST selection, source modification or full-corpus generation was performed.

## Reproducibility and limitations

Representation fingerprints include mode, family, source artifact, window/hop settings and sample identity in the adapter metadata. The current task intentionally does not persist a cache because official frozen inference is blocked. Future Mantis output must add checkpoint/revision, selected layer, pooling, preprocessing and code fingerprints and fit any learned preprocessing on TRAIN only.

## Chimera-ML integration

Installed `chimera-ml==0.3.0` uses `DATAMODULES.register`, `build_datamodule`, and the `chimera_ml.plugins` entry-point group. `src/chimera_plugin.py` registers `carewsm_joint`; the factory subclasses Chimera standard `DataModule`, uses standard `MaskingCollate`, and exposes the native adapter as the `native` input.

YAML: `configs/audio/joint_native_fold0.yaml`. Command: `PYTHONPATH=src:. .venv/bin/python scripts/smoke_chimera_datamodule.py`. Real fold-0 sizes: TRAIN 6998, joint VAL 1780 (CARE 352 / WSM 1428), CARE TEST 318, WSM TEST 1364. Every loader produced a batch with native tensors and sequence masks, without training. Entry-point registration also succeeds without manually calling `chimera_plugin.register()`.
