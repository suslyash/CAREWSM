# NEXT TASK

## TASK-002B — bounded WSM DiariZen feasibility and participant-turn gate

**Status: READY after TASK-002A manager PASS (PR #15 merged).** This is a one-off engineering gate, **not** full diarization, feature extraction or training. No general CARE/WSM dataset writes are permitted.

### Source of truth / startup
Run `git fetch origin`, `git switch main`, `git pull --ff-only origin main`, check `git status --short` and create NEW branch `codex/task-002b-diarizen-gate`. Read `AGENTS.md`, active docs in their prescribed order, accepted `docs/TASK-001A_CORPUS_FEATURE_REPO_AUDIT_EN.md` and `docs/TASK-002A_WSM_AUDIO_PREFLIGHT_EN.md`. Inspect the CARE reference `/media/maxim/Databases/CARE/scripts/extract_diarizen.py` and `turn_utils.py` read-only; do not assume that their CLI or pretrained weights work unmodified on WSM.

### Goal
Answer for **at most three** short, deterministic, existing WSM TRAIN recordings whether local DiariZen can reliably identify useful speaker/participant turns, at what real-time cost, and which exact CARE semantics would need adaptation. Include different source-task/control-positive groups **where feasible**, rather than using only the first five depression/CONTROL preflight records. If an observed group is unavailable, report the actual sample-selection limitation without widening scope.

### Authorized tracked changes
- Create **only** `scripts/audio/wsm_diarizen_gate.py`, a standalone bounded CLI outside ChimeraML. It must require an explicit WSM root, select at most 3 deterministic TRAIN rows with existing media using observed manifest columns, reject traversal and non-TRAIN paths, cap each selected recording to at most 120 s of media, and print machine-readable JSON summary to stdout (or a clear `blocked` result). No script in `src/`; absolutely no `chimera_ml` imports, registry, trainer, DataModule, or training config.
- Create `docs/TASK-002B_DIARIZEN_GATE_EN.md`: exact commands/CLI, evidence for sample selection and duration, setup/runtime dependencies and weights, wall-clock and real-time factors, detected speakers/segments/turn durations, and cautious respondent/interviewer sanity against CARE source policy. Identify anything unmeasured or blocked; recommend **retain / narrow-optimize / drop** with observed justification but leave the decision to manager/owner. No invented successful diarization.
- Update `docs/PROGRESS_EN.md` only to record outcome **pending manager review**.

### Small runtime authorization
- First perform read-only environment/CARE source inspection. Run the gate **only if DiariZen, its exact dependencies and model weights are already available locally**, with no new installation or implicit download. Explicitly disable relevant automatic network/model downloads; if this cannot be guaranteed, stop at a documented blocked preflight.
- For supported input, perform only bounded decoding/conversion to **16 kHz mono** and actual DiariZen processing for **at most 3** TRAIN media files, up to **120 seconds each**; capture any generated WAV/RTTM/logs **only in an ephemeral non-corpus temporary directory** (e.g., Python `tempfile.TemporaryDirectory`) and clean it afterward. Do not write to CARE or WSM `extracted/`, `splits/`, raw files or anywhere in their roots. Store only report/CLI source in Git.
- Log source duration, time elapsed and real-time factor; inspect turn assignments, number of speakers/segments, participant-turn coverage and whether the CARE longest-speaker=`PARTICIPANT` heuristic looks reasonable. Do not silently assume longest speaker is the respondent. If no reliable reference for respondent identity exists, label the sanity determination `unverified`, not `pass`.
- Use one bounded attempt; if missing weights, unavailable GPU, model downloads required, or clear failures occur, record exact blocker and stop. **No extended tuning or automated retries.**

### Prohibited
No other scripts/config/source edits, no model dependency installation, no training, no Test access/evaluation, no feature-family extraction, no full-corpus diarization or expensive scans, no changes to CARE/WSM data, split/label semantics, or saved feature outputs. Do not create or run tests (owner preference). No README outside repository root or docs.

### Acceptance / handoff
Give exact TRAIN sample IDs/source-task categories, CLI commands and JSON results (or explicit blocked evidence), elapsed durations/RTF, selected segment summaries, artifact cleanup confirmation, Git diff/check/scope, branch/base/full commit SHA, normal non-force push, clean worktree. Do not claim DiariZen is usable without measured runtime plus observed speech-turn sanity; do not claim negative evidence proves it unusable corpus-wide. Manager will decide retain/narrow-optimize/drop and authorize the first eGeMAPS extraction smoke separately.

Codex handoff: Outcome; Changed files; Verification; Data/Test access; Git handoff; Blockers and risks; Next atomic step. **No PR or merge by Codex. Stop for manager review.**
