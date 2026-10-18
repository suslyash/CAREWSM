# NEXT TASK

## TASK-003B — four deep-audio CARE-compatible WSM feature families

**Status: READY — urgent audio-only feature extraction.** TASK-003A was manager-accepted and merged in [PR #27](https://github.com/suslyash/CAREWSM/pull/27), merge SHA `4f6fadffc29ade18bc252dff43fd9a7d469d7f60`. The implementer reported 21 real CSVs on three WSM TRAIN recordings (7 eGeMAPS/DisVoice families each); a read-only validator reported **PASS on all rows**, exact CARE column/order/metadata schema, no NaN/Inf, four blank prosody feature cells documented. Manager checked the committed source/reports and could not independently access the local corpus.

**OWNER RESOLUTION — no CARE audio source exists.** Existing CARE descriptor CSV/RTTM archives contain no original media/audio suitable for exact same-input extraction. Do **not** re-audit, request, reconstruct, download or manufacture those recordings. Numeric equality to original CARE producer values is **unverifiable**; mark that permanent scientific limitation **UNVERIFIED**, not PASS. Matching schema and use of reference algorithms is accepted as the practical engineering contract. The discarded previous CARE numeric replay is **not a blocking gate** for TASK-003B or eventual audio-only batch. Human TASK-002C listening remains deferred due to SSH access.

### Goal

Implement and, when locally cached dependencies/weights permit, actually extract the **remaining four CARE audio feature families** on the **same three deterministic WSM TRAIN recordings**, preserving exact CARE file names, turn-level representation, numeric semantics, dtype, keys/shapes and WSM `extracted/` destination layout:

| family | exact file | reference contract |
|---|---|---|
| `wav2vec_xlsr300m` | `wav2vec_xlsr300m.npz` | `mean`, `std` float32 `(n_turns,1024)`; `turn_ids` int64 `(n_turns,)` |
| `trillsson4_1` | `trillsson4_1.npz` | `mean`, `std` float32 `(n_turns,1024)`; `turn_ids` int64 `(n_turns,)` |
| `emotionmsp_trajectories` | `emotionmsp_trajectories.csv` | metadata `turn_id,turn_start_sec,turn_end_sec` + `arousal,dominance,valence` (3 numeric) in exact CARE order |
| `emotionmsp_embeds` | `emotionmsp_embeds.npz` | `embed` float32 `(n_turns,1024)`, `turn_ids` int64 `(n_turns,)` |

These shapes are from the accepted bounded CARE example, not proof of all variants. The `mean/std` naming must reflect the actual CARE extraction/aggregation logic; **do not fake a per-turn vector by duplicating, concatenating or zero-filling embeddings.**

### Startup and authorized Git scope

Fetch `origin`, fast-forward `main`, read `AGENTS.md`, `docs/PROJECT_REQUIREMENTS.md`, `docs/PLAN.md`, `docs/PROGRESS_EN.md`, `docs/FEATURE_PARITY_TARGET_EN.md`, `docs/TASK-003A_AUDIO_DESCRIPTOR_EXTRACTION_EN.md` and source files needed for this task. Start fresh `codex/task-003b-deep-audio` from current `origin/main`, clean worktree; do not keep working on TASK-003A branch or reopen PR #27. Codex implements one task, commits/pushes branch normally, **no PR**.

**Allowed tracked changes only:** new `scripts/audio/wsm_extract_care_deep_audio.py` and minimal reusable audio-only helpers in `scripts/audio/` if essential; `scripts/audio/validate_care_audio_parity.py` for NPZ/key/dtype/shape/metadata validation; `docs/TASK-003B_DEEP_AUDIO_EXTRACTION_EN.md` (new) and `docs/PROGRESS_EN.md`. No `src/`, video/text extractors, training configs, tests, project package/dependency files, CARE code edits, or unrelated files.

### Implementation and CARE algorithm semantics

1. **Inspect actual CARE producer implementations read-only:** `/media/maxim/Databases/CARE/scripts/audio_descriptors/extract_wav2vec.py`, `extract_trillsson.py`, `extract_audeering_emotionmsp.py`, `/media/maxim/Databases/CARE/scripts/turn_utils.py`; and genuine representative CARE CSV/NPZ headers (one known directory is `CARE/extracted/CONTROL/ASTHMA/PROFILE_053/VIDEO_000/`). Establish actual model IDs/revisions/weights, frame windows/sampling, pooling/turn selection, speech filtering, embedding layers, emotion column order and `turn_ids` dtype from source. Record any mismatch or uncertainty rather than guessing. Do not extract CARE audio (none exists).
2. Use WSM `splits/train.csv` source `video_path` and immutable real manifest fields `label_id,subgroup,profile_id,video_id` for the **canonical output** `WSM_NEW/extracted/<label_id>/<subgroup>/PROFILE_<profile_id>/VIDEO_<video_id>/`. `feature_dir` points to the **legacy** `*/train_labels/*/features/*` location and is NOT the destination; never edit WSM CSVs. Confirm existing three 7-descriptor outputs and safe path/ID containment. Preserve existing transcripts, 7 CSVs, raw media, external-team video/text feature files and all valid outputs.
3. Match TASK-003A's actual sample selection (DEPRESSION, CONTROL, PARKINSON, one each) and respondent-turn policy as implemented in CARE, using already observed local cached DiariZen when necessary, *only* for these three WSM TRAIN clips. One detected speaker per sample and longest-speaker `PARTICIPANT` selection are **provisional** and scientifically unverified; do not assert human-verified respondent identity. Audio may be decoded/converted to temporary **mono 16 kHz**; other model-specific resampling/normalization must match actual CARE code.
4. Modular `--dry-run` and `--split train --limit 3` CLI, explicit WSM/CARE roots, explicit local model/cache paths where possible, offline/no-network loading and stable device choice; no implicit `from_pretrained` network download. Make calls compatible with existing approved Python environments; **do not create any additional venv**. If two model stacks cannot coexist with existing compatible environments, report blocker and owner-run options rather than silently changing torch/TensorFlow.
5. Stage each sample's four outputs in temporary scratch **on the destination filesystem**, validate **all** expected CARE column headers, NPZ keys, dtypes, turn IDs, shapes and values (not simply matching file dimensions), then atomically publish with **no-clobber**. Valid existing files must be reused (skip); malformed existing files must not be silently overwritten. Missing model/input/turn resources should be reported as `blocked_missing_...` per family rather than creating fake/empty features. Report wall time, GPU/CPU backend and basic resource/use counts; never run full batch.

### OWNER-ONLY DOWNLOAD RULE (permanent)

**Do not install, download, or trigger downloads of TensorFlow, TensorFlow Hub, Transformers, Torch, pretrained XLS-R, TRILLsson or emotion MSP checkpoints, other packages or weights.** Before any potentially large transfer, provide **exact verified copy/paste owner commands**, official source/model IDs and revision, known/unknown transfer and transitive-size estimate, cache target and post-install verification. Owner performs downloads personally and shares resulting paths. Reuse locally cached packages/models. Do not launch model constructors that may silently fetch data. If weights or dependencies missing, **complete the safe implementation/documentation and stop with explicit provisioning commands**. No irrelevant new ML environments, pip changes or broad local cache scans.

### Authorized bounded local execution

**Owner authorizes new deep-audio output files for at most the same three WSM TRAIN clips** under their already-verified canonical `extracted/` directories, only when packages and model weights are **already present locally**. Run exactly one small offline extraction smoke and a thorough read-only validation of the resulting four families. No DEV or TEST access during implementation/smoke, no model training, feature scaling fitted on TEST, evaluation, test suite or other modality extraction. No corpus-wide extraction yet. If a dependency/weight is absent, do not attempt inference or create files for that family; report blockers and owner-download commands, and stop.

### Evidence and acceptance

Write `docs/TASK-003B_DEEP_AUDIO_EXTRACTION_EN.md` with the exact CARE source algorithms/parameters, model IDs/revisions/licenses when verified, environment/library versions, offline policy, original input and canonical output paths, turns/provenance, per-family produced/skip/blocked status, keys/dtypes/shapes, row/turn counts, numeric finiteness/blank handling, wall-clock/device, cleanup and total bytes. Explicitly state that **11-audio-family CARE structural compatibility** is achieved only when seven TASK-003A outputs and all four TASK-003B outputs pass on the selected TRAIN files. **Numerical equality to original CARE reference outputs remains untestable without original CARE audio and is NOT a blocker for engineering progression.**

Finish seven headings: Outcome; Changed files; Verification; Data/Test access; Git handoff; Blockers and risks; Next atomic step. Include commands, `git diff --check`, branch/full SHA, non-force push, clean worktree. No tests created/run, no training, no PR, no unauthorized downloads or bulk extraction. Manager reviews before approving full WSM audio batch.
