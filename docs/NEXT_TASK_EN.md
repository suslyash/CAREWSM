# NEXT TASK

## TASK-002A — WSM audio input and CARE feature-contract preflight

**Status: READY after TASK-001D manager PASS (PR #12 merged).** Repo cleanup is complete. Stage 2 begins with a small, read-only and reproducible **audio preflight**; no extraction or diarization is authorized in this task.

### Startup
Run `git fetch origin`, `git switch main`, `git pull --ff-only origin main`. Verify `git status --short`. Create **new** `codex/task-002a-wsm-audio-preflight` from current `origin/main`. Read current `AGENTS.md`, active `docs/` contracts in their required order, the accepted `docs/TASK-001A_CORPUS_FEATURE_REPO_AUDIT_EN.md`, and the observed CARE extraction scripts at `/media/maxim/Databases/CARE/scripts/`. Read-only access only to CARE and WSM corpus roots. Report blockers if roots are unavailable; do not invent files.

### Goal
Produce one concise, *verifiable* preflight for **WSM audio feature parity against CARE**, addressing actual raw input paths, a deterministic bounded TRAIN-only sample, audio metadata and exact output schemas/dependency constraints. The user wants practical progress toward audio extraction and DiariZen, not another repository cleanup or legacy WSM experiment.

### Authorized tracked changes
- `scripts/audio/wsm_audio_preflight.py`: a minimal standard-library CLI (if external `ffprobe` is available, use it via subprocess; no ffmpeg conversion/writing) that reads WSM fixed TRAIN manifest, resolves existing raw video/audio paths **from observed columns and paths**, selects no more than **5 stable TRAIN examples** (e.g. deterministic stable-key ordering, without user-specific assumptions), probes stream presence, duration, sample rate and channel count when available, and prints a JSON summary to stdout **only**. It must have a `--wsm-root` option and a `--limit` with hard cap 5; fail clearly on ambiguous column/path mappings rather than guessing. It must not open or enumerate TEST data.
- `docs/TASK-002A_WSM_AUDIO_PREFLIGHT_EN.md`: report exact observations and commands. Inspect CARE source extraction scripts for 16 kHz mono preparation, respondent segmentation/turn policy and representative audio feature formats (eGeMAPS v2, DisVoice, XLS-R and TRILLsson as applicable), and identify which details are measured, inferred, or still unverified. Specify output filenames, parent paths, shapes/columns, callable scripts, dependencies, missing pieces and one well-scoped next extraction smoke candidate. Explicitly explain the open `linguistic_turnlevel` registry 41-vs-observed-39 discrepancy without silently resolving it.
- `docs/PROGRESS_EN.md`: record TASK-002A results *pending manager review*, with no assertion that feature parity was achieved.
- Any future audio preparation scripts belong exclusively in `scripts/audio/`, independently of ChimeraML. In this task, the **only** script is `scripts/audio/wsm_audio_preflight.py`.

No other tracked modifications. Do not touch `src/` (except reading), `configs/`, package metadata, active requirements or plan, or legacy model/callback source.

### Audio-processing architecture boundary
All current and future one-time **audio** preprocessing, media probing, respondent-turn preparation, diarization/DiariZen gate, feature extraction, aggregation and feature-cache creation scripts belong under **`scripts/audio/`**, not under `src/` or the root of `scripts/`. Treat CARE reference scripts as sources of extraction logic, but place any adapted WSM executables under `scripts/audio/`. These scripts must be standalone CLI/offline jobs: **no imports from `chimera_ml`**, no ChimeraML registry, plugin, trainer, callback, DataModule or training-config coupling. Training and evaluation code may later read the produced CARE-compatible features through approved ChimeraML components under `src/audio/`, but feature production itself remains strictly outside ChimeraML. Do not create any extraction or diarization code in TASK-002A beyond the bounded read-only preflight.

### Restrictions
- **No** CARE or WSM data writes, extraction output/caches, audio re-encoding, DiariZen invocation, transcript changes, model loading/downloading, package installation, training, evaluation, TEST metadata/model metrics, or huge directory scans.
- Do not install or execute heavyweight feature extractors. A read-only, bounded `ffprobe` of at most five TRAIN media files is permitted. Do not create or run tests; only execute the bounded preflight and static checks. Do not create README files outside repository root/docs.
- Do not change the owner-approved split/label/fold policy or infer unknown labels as negatives. No full feature parity claim until a separately authorized extraction task.
- If the WSM raw media location, TRAIN file link or CARE scripts are inaccessible/ambiguous, document the exact evidence and blocker instead of inventing a data layout or running alternate workloads.

### Manager acceptance evidence
Give `git diff --check`, complete changed-path list, exact CLI invocation/output (or exact unavailable `ffprobe`/data-root blocker), count of selected TRAIN files and media metadata where available, explicit CARE script-to-schema mapping, scope of read-only access, base SHA, task commit full SHA, clean worktree and non-force push. No PR — manager reviews and merges on PASS.

Required Codex handoff sections: Outcome; Changed files; Verification; Data/Test access; Git handoff; Blockers and risks; Next atomic step. End at manager review; **do not run feature extraction, DiariZen, training or write to either corpus**.
