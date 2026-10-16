# NEXT TASK

## TASK-001A — repository, corpus, and CARE feature-contract audit

**Status: NOT STARTED — manager must issue the implementation prompt**

The next manager should issue one read-only Codex audit covering:

### CARE

- `/media/maxim/Databases/CARE/splits/videos.csv`
- `train.csv`, `dev.csv`, `test.csv`
- all five `fold_{0..4}_{train,val}.csv` pairs
- `splits/features.json`
- `splits/care_features.py`
- `/media/maxim/Databases/CARE/scripts`
- representative files under `extracted/`

### WSM

- equivalent split/fold CSVs
- current `extracted/` coverage
- raw `depression/` and `parkinson/` source organization

### Repository

- `src/`
- `configs/`
- `tests/`
- retained `scripts/`
- `pyproject.toml`

## Required audit outputs

1. split/fold counts;
2. participant/profile overlap audit;
3. exact label columns, values, availability, and missingness by corpus/split;
4. exact CARE feature registry and representative feature-file schemas/dtypes/shapes;
5. CARE extractor script and dependency inventory;
6. exact WSM feature-coverage gap relative to CARE;
7. repository component inventory with KEEP / ADAPT / LEGACY / DELETE recommendations;
8. recommended new active package/config layout;
9. a **small** future DiariZen benchmark design only;
10. proposed documentation corrections where observed facts conflict with owner expectations.

## Prohibitions

TASK-001A must not:

- train a model;
- extract/recompute features;
- run the DiariZen benchmark;
- modify CARE or WSM datasets;
- delete/move repository code/configs;
- start cleanup before manager review.

The manager reviews TASK-001A before authorizing the cleanup/extraction stages.
