# TASK-003 — Paired native vs frozen-Mantis temporal representations

## Objective
Prepare **both** representation paths for a controlled CARE+WSM comparison:
- **native**: feed actual audited temporal feature sequences directly to future Mamba / Transformer classifiers;
- **mantis**: pass defensibly windowed time-series features through a **frozen** Mantis encoder and feed its ordered embeddings to those **same** future classifiers.

Mantis is optional, not a mandatory preprocessing step. The experimental question is whether Mantis improves validation metrics compared with native features on matched samples/folds. **This task builds/validates representation adapters and a tiny real-data smoke, not disease classifiers or training.**

## Operating procedure
Read `AGENTS.md`, the current project contracts, TASK-001 evidence and TASK-002 DataModule. Begin from updated `origin/main`; work on `codex/task-003-dual-representation`. Normal commit + push; no PR/merge.

## Data and alignment
Use the existing `JointDataModule` and the audited 11-family registry, CARE `/media/maxim/Databases/CARE/`, WSM `/media/maxim/Databases/WSM_NEW/`. **Only** canonical `fold_{k}_train.csv` / `fold_{k}_val.csv` for joint TRAIN/VAL; separately fixed CARE/WSM TEST. `dev.csv` is not an extra training or validation source.

Make two explicit feature modes `native` and `mantis` in a small interface. For matched evaluation, both modes must use identical `(corpus,sample_id)` membership and identical observed-label masks for each feature-set/fold. Preserve profiles, corpus, block/turn identity, temporal order, and VAL/TEST isolation; report paired retention and exclusions. No TEST-driven decisions.

## Windowing and Mantis validity
- Determine eligible families **from real schemas**, not family name alone. At minimum inspect `egemaps_lld`, dynamic DisVoice, and timestamped trajectories; record eligibility and reasons for exclusions. Do not assume every registry family has a usable sampling grid.
- For Mantis only: construct **1.0-second windows / 0.5-second hop** from defensible timestamps; retain window start/end, valid-observation fraction, validity mask, original identity, source family and preprocessing metadata. Respect turn boundaries and missing intervals. State explicitly how frame/segment/turn timestamps align; no fabricated times or pseudo-time expansion.
- Keep turn-level/static embeddings (including pre-aggregated wav2vec/Trillsson/emotion embeddings) in native mode only unless a real 1-second axis is justified. They should not be silently repeated into Mantis windows.
- Research the **official Mantis implementation** (`https://github.com/vfeofanov/mantis`) and pin a reproducible model checkpoint/version, expected input shape, length, sampling assumptions, selected layer and pooling. Record an executable small compatibility probe. If the proposed 1-second acoustic windows are incompatible with the official encoder, **report BLOCKED with evidence** instead of inventing an input contract, resampling without justification, or running an unsuitable encoder.
- If compatible, implement small frozen inference with `eval()` / no gradients; deterministic caching of **per-window embedding sequences** `[num_windows, embedding_dim]`. Store metadata and fingerprints (checkpoint, code/config, source artifact, feature family, windowing, preprocessing), never a model copy per sample. Cache only a tiny representative real-data sample in this task, not all CARE/WSM.
- Avoid leakage: fixed, non-learned transformations may be shared; any fitted scaling/PCA must be TRAIN-only for each fold, never global/VAL/TEST. Do not introduce learned preprocessing in TASK-003.

## Native comparator and future model contract
- Preserve original temporal sequences in `native` mode with sequence masks, lengths and explicit feature dimensions. No forced 1-second windowing for native mode, no global averaging, and no fake time axis for static inputs.
- Both modes must expose a consistent *interface* usable by future **TSCMamba-inspired** and **MD-Former-inspired** classifiers (and later MPTSNet-inspired); different input dimensions are allowed and must be recorded. This task does **not** implement those classifiers.
- Define a paired experiment config contract: mode, feature family, fixed fold, same samples/labels, seed, model family, cache/representation fingerprint; comparisons must select using VAL only. Model architecture and training belong to TASK-004 and beyond.

## Deliverables and verification
- Minimal code under `src/audio/features/` or `src/audio/data/` plus a short smoke wrapper under `scripts/`; no new large framework.
- `docs/evidence/TASK-003_DUAL_REPRESENTATION_EN.md` showing official Mantis compatibility, eligible families, exact timings, one real native sample shape, Mantis sample shape **only if compatible**, reproducibility command, cache metadata, matching sample IDs, and limitations. Record BLOCKED explicitly if official Mantis cannot be used on these inputs.
- One small real-data smoke for at least one defensible temporal family, preferably in both corpora; check first/last window, no off-by-one, masks, and paired sample IDs. No full-corpus Mantis generation.
- Only brief focused checks (no expanding unit-test programme): `py_compile`, tiny smoke and `git diff --check`. Update `docs/PROGRESS_EN.md` only among manager docs.
- Normal commit/push, clean worktree, eight-section handoff per `AGENTS.md`; stop for manager review.

## Forbidden
No models/classifiers, model training, feature screening, Optuna, loss balancing, global Mantis embeddings, source feature re-extraction, split editing, DEV substitution, or PR/merge.

**Stop at TASK-003 handoff.**
