# AV-005 Fusion Sweep

`UPSTREAM_PROVENANCE=UNKNOWN — EXPLORATORY RESULTS ONLY`

## Status

AV-005 implementation and matrix preparation are complete. No GPU training was
started. Existing AV-004 15-run configurations remain unchanged and the new
sweep writes only under `logs/fusion/sweep/`.

## Architecture inventory

- `carewsm_bimodal_gated_late`: task-wise gated late fusion, adapted from the
  historical F0 design.
- `carewsm_bimodal_shared_mtl`: shared multitask trunk with residual modality
  paths, adapted from F1.
- `carewsm_bimodal_task_aware`: two directed relation paths with task-specific
  relation weights, adapted from F2.
- `carewsm_text_guided_dual_gated`: existing AV-003 comparison.
- `carewsm_audio_guided_dual_gated`: project-specific adaptation; audio
  conditions video and text channel gates, followed by learned modality weights.
  This is not a reproduction of a published audio-guided paper.
- `carewsm_trimodal_cross_attention`: existing AV-003 comparison with a
  singleton pooled path and mask-safe sequence attention.
- `carewsm_shared_private_trimodal`: existing AV-003 comparison.

All models use two independent Depression/Parkinson heads, explicit dimensions,
sequence masks, and real singleton pooling. Bimodal batches contain exactly two
selected modalities; no third modality is zero-filled.

## Feature combinations and exact matrix

The audited families are A=`audio_id1`, `audio_id2`, `audio_id4`;
V=`video_4`; and T=`text_id9`, `text_id10`, all using `shared_features`.
The 11 bimodal combinations are A+V (3), A+T (6), and V+T (2). The six
trimodal combinations are 3 audio versions × `video_4` × 2 text versions.

| Matrix | Architectures | Feature combinations | Folds | Jobs |
|---|---:|---:|---:|---:|
| Bimodal | 3 | 11 | 5 | 165 |
| Trimodal | 4 | 6 | 5 | 120 |
| Total | 7 | 17 compatible selections | 5 | **285** |

There are 57 identities per fold, no duplicate run names, no same-modality
pairs, and every compatible architecture/feature selection is represented.
The generator is `scripts/fusion/generate_sweep.py`; committed YAMLs are in
`configs/fusion/sweep/`.

## Common complete-case cohort

The DataModule now requires the six-family complete case
(`audio_id1`, `audio_id2`, `audio_id4`, `video_4`, `text_id9`, `text_id10`)
while loading only the selected subset into each batch. Live read-only counts
from the canonical folds are:

| Fold | CARE train | CARE val | CARE test | WSM train | WSM val | WSM test | Joint train |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1299 | 327 | 304 | 5499 | 1403 | 1341 | 6798 |
| 1 | 1296 | 330 | 304 | 5530 | 1372 | 1341 | 6826 |
| 2 | 1308 | 318 | 304 | 5544 | 1358 | 1341 | 6852 |
| 3 | 1312 | 314 | 304 | 5505 | 1397 | 1341 | 6817 |
| 4 | 1289 | 337 | 304 | 5530 | 1372 | 1341 | 6819 |

The source identity remains `(corpus, sample_id)`, and profile isolation is
checked across TRAIN, VAL, and TEST views.

## Logging, launcher, and MLflow

Every sweep config targets `logs/fusion/sweep/` with one `train.log`,
`summary.txt`, checkpoint directory, snapshot, and MLflow run identity. MLflow
uses `sqlite:///logs/fusion/sweep/mlflow.db`. Checkpoints, logs, and databases
are not committed.

Use the single entrypoint as follows:

```bash
bash scripts/fusion/run_all.sh --list --matrix sweep
bash scripts/fusion/run_all.sh --check --matrix sweep
tmux new -s carewsm_av005
bash scripts/fusion/run_all.sh --run --matrix sweep
# detach: Ctrl+B, D
bash scripts/fusion/run_all.sh --resume --matrix sweep
```

`--check` performs config/identity/path preflight without training. `--run`
refuses existing runtime identities, and `--resume` skips only verified
completed directories, archives incomplete artifacts, stops on ambiguous runtime
directories, and records launcher status. The original one-argument 15-run
commands remain supported.

## Verification and evidence

- `bash scripts/fusion/run_all.sh --check --matrix sweep` validated 285 unique
  configs and the 55/55/55 bimodal plus 30/30/30/30 trimodal per-model counts.
- Focused tests: 22 passed in `tests/test_av005_fusion_sweep.py`,
  `tests/test_av002_trimodal_data.py`, `tests/test_av003_trimodal_models.py`,
  and `tests/test_av004_fusion_launch.py`.
- The live DataModule was constructed for all five folds and a representative
  forward/backward test was run on synthetic masked batches. No training or
  optimizer step was run.

## Workload estimate

The planned workload is 285 independent jobs, each capped at 50 epochs with
early stopping patience 7, batch size 32, and the AV-004 AdamW settings. Runtime
is intentionally not estimated without measured machine throughput; actual
wall time depends on observed epoch duration, early stopping, and GPU
availability.

## Provenance limitation and next action

The extracted upstream FIT/SELECTION rosters and provenance remain UNKNOWN, so
these jobs are exploratory and must not be interpreted as leakage-free evidence.
After independent review and merge, the owner may launch the complete sweep in
tmux. This task does not start jobs.
