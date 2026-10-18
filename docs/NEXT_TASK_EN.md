# NEXT TASK

## TASK-002C — WSM TRAIN speaker-count plausibility and DiariZen decision evidence

**Status: READY after TASK-002B manager PASS for bounded GPU runtime, PR #21 merged.** DiariZen was fast on three TRAIN clips, but it returned exactly **one speaker in every file**, with respondent/interviewer identity **unverified**. This task is a **small listening/annotation sanity gate**, not full diarization, extraction, experiment or training. No new downloads.

### Startup and scope
Fetch `origin`, update `main` using `git pull --ff-only`, verify clean Git status, and create a fresh `codex/task-002c-speaker-sanity` branch from latest `origin/main`. Read `AGENTS.md`, active `docs/` contracts, and the TASK-002A / TASK-002B reports before accessing any local corpus file.

Reuse **only** the same three WSM `splits/train.csv` sample IDs documented in `docs/TASK-002B_DIARIZEN_GATE_EN.md`: one DEPRESSION, one CONTROL, one PARKINSON. Do not enumerate DEV or TEST; do not inspect additional WSM media. Original raw files are read-only.

### Goal
Provide the owner with a **short, reproducible manual listening checklist** to decide whether the three clips really contain one speaker or whether DiariZen missed an interviewer. Distinguish what the automatic model detected from what a human actually confirms. No automatic diarization accuracy claim without verified manual/ground-truth speaker labels.

### Authorized work
- Read the accepted TASK-002B summary and source paths. For each of the same three videos, prepare **2–3 bounded owner-runnable playback commands** (`ffplay` or equivalent) for distinct, short regions (e.g. early/middle/late, each no more than 15 seconds), selected to have actual observed speech where possible. Use explicit input paths from the TRAIN manifest; use `-ss`/`-t` and `-nodisp`/audio-only playback flags as supported. No new media export is needed. If no trustworthy speech-interval timestamps are available, select conservative evenly spaced windows and mark that limitation.
- If useful, Codex may execute at most **one** rerun of existing `scripts/audio/wsm_diarizen_gate.py` using **already-cached** weights in the **already-existing** `.venv-diarizen` (owner-approved exception) to observe detailed segment times. Reuse existing script; no new scripts, installations, new models or caches. Absolutely **no network/model download**, CPU-only PyTorch, or broad scanning. The gate itself already limits to 3 TRAIN inputs and <=120 seconds/media and uses temporary 16 kHz mono scratch files. Do not modify the corpus, track RTTM/WAV, or claim manual confirmation.
- Create `docs/TASK-002C_SPEAKER_SANITY_EN.md` with exact source recordings, source/model-observed speaker count, suggested playback windows and **copy/paste listening commands**, what to listen for, and a small owner response protocol: for each recording `one speaker / two or more / uncertain`, plus `respondent identifiable? yes / no / uncertain`. The owner may reply in chat; do not invent their response.
- Update `docs/PROGRESS_EN.md` to record that *manual owner confirmation is pending*. No DiariZen retain/drop decision before feedback unless compelling source-grounded evidence emerges and is clearly labeled as inference.
- No other tracked changes. No tests (owner preference), training, feature extraction, dataset writes, model downloads, new venvs, config/metadata changes or TEST access. Keep extraction scripts under `scripts/audio/` and independent of ChimeraML.

### PERMANENT OWNER BANDWIDTH RULE
For any current or future potentially significant dependency or weight download, **Codex first provides the owner verified commands, official source, revision/version, expected transfer size or explicitly unknown size, cache destination and verification commands; owner performs the download manually**. Never silently pull a model via `from_pretrained` or `pip install`. This TASK-002C authorizes **zero downloads**.

### Handoff
Provide brief 7-section handoff: Outcome; Changed files; Verification; Data/Test access; Git handoff; Blockers and risks; Next atomic step. Include copy/paste `ffplay` commands in the handoff itself for the owner, plus exact branch/base/full commit and normal non-force push, clean worktree, diff scope/`git diff --check`. No PR: manager handles review/merge. **Stop for owner listening and manager decision; do not proceed to extraction.**
