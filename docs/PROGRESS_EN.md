# CAREWSM Active Progress

**LATEST OWNER CORRECTION — TASK-004:** Existing 11-family outputs created via DiariZen turn selection are now **obsolete and scheduled for exact-file deletion**. WSM source VIDEO media are already segmented; current intended output is 11 CARE-schema audio families per **whole VIDEO**, single `turn_id=1` with 0-to-decoded-duration metadata, without DiariZen or RTTM. The former 517+ output count and currently active TASK-003D production progress are **legacy evidence only, not accepted coverage for new TASK-004 segment-native features**. Owner must stop old `tmux` and all writers before any reset. TASK-004 authorizes code and bounded scratch TRAIN smoke in isolated worktree; owner later runs one reset-and-reextract all TRAIN/DEV/TEST command from new runtime, deleting only the 11 exact old audio files and old `_carewsm_audio_turns.json` in canonical manifest-mapped destinations; all other files preserved. Do not count new-mode progress until actual new outputs are generated and validated. No DiariZen downloads/inference, tests, training or other modalities.


## Project state

Project: **CAREWSM**

Repository bootstrap source: a one-commit copy of the completed WSM codebase.

The previous WSM evidence/task history is intentionally excluded from the active CAREWSM research record. No old WSM metric or claim is an accepted CAREWSM result.

## Current stage

Stage 1 — corpus, feature, and legacy-code audit.

Status: **TASK-001A ACCEPTED (PR #3)**, **TASK-001B ACCEPTED (PR #5)**, **TASK-001C ACCEPTED (PR #9)** and **TASK-001D ACCEPTED (PR #12)**. Stage 2: **TASK-002A manager PASS, PR #15 merged**; **TASK-002B manager PASS for bounded GPU runtime (PR #21)**; TASK-002C playback checklist **accepted and deferred** due to SSH audio limitations. **TASK-003A/B accepted (PR #27/#30); TASK-003C corrective accepted (PR #32): 46 complete WSM TRAIN records × 11 audio files = 506 validator-reported outputs; six of seven blocked attempts recovered, one VUV/Praat failure remains. TASK-003D owner-operated production release preparation is NEXT, Codex read-only TRAIN preflight only.** Five video and three linguistic families are supplied by other teams for future fusion; no video/text extraction or training.

No CAREWSM scientific experiment has been run.

No CAREWSM baseline has been accepted.

TASK-001C removed obsolete WSM source, tracked old scripts and copied tests. TASK-001D removed 157 old config files, flattened 14 historical model files into passive `src/legacy/{audio_models,fusion_models,video_models,text_models}/`, and archived two callback sources under `src/legacy/callbacks/`. Direct package dependencies are now only `chimera-ml>=0.2.4`; old WSM methods remain inactive.

## Data roots

- CARE: `/media/maxim/Databases/CARE/`
- WSM: `/media/maxim/Databases/WSM_NEW/`

TASK-001A observations: fixed train/dev/test and five TRAIN CV folds were consistent and sample/profile-disjoint in both corpora; CARE `videos.csv` and `care_features.py` were absent; WSM manifest has 34 columns and per-task missing labels; 0/19 CARE feature families currently appear in WSM. The CARE `linguistic_turnlevel` declared-vs-observed feature dimension discrepancy remains unresolved.

## Evaluation policy

CV folds are used for development/hyperparameter selection. DEV is used for final model/configuration selection. TEST is visible and stored as descriptive/reporting evidence for completed DEV-selected checkpoints, but it must not influence model, checkpoint, threshold, hyperparameter, architecture, sweep, or retry decisions.

## Completed task

TASK-001A — read-only corpus/feature/repository audit; manager **PASS**. Evidence: `docs/TASK-001A_CORPUS_FEATURE_REPO_AUDIT_EN.md`. Merged PR: https://github.com/suslyash/CAREWSM/pull/3 ; merge SHA: `4097bd52adc3d46551f2439dd24c040f06a55e47`. No models or TEST metrics produced.

**LATEST OWNER OVERRIDE:** TASK-003D now authorizes one single-target Parkinson DisVoice/Praat source diagnosis plus **at most one real extraction attempt** for `PARKINSON__PARKINSON__-1N99QsfWRk__-1N99QsfWRk_003`, preserving 506 existing validated audio files. Codex will produce a **single complete owner-operated `tmux` launcher command** for all 11 CARE-style audio feature families over all eligible WSM TRAIN+DEV+TEST input media. Codex may not launch bulk extraction, touch nonaudio features or access DEV/TEST before owner starts the full workflow. One sample may remain blocked if repair isn't scientifically safe; flag it explicitly and do not block unrelated data. This supersedes the previous TASK-003D no-retry/no-DEV-TEST-launch wording only for the owner-executed all-splits future run. Numerical equality to old CARE CSV values remains UNVERIFIABLE; current CARE structural/NPZ/column parity is validated by existing files/reports.

## Next task

TASK-003D — prepare safe owner-executed full WSM AUDIO extraction mode and exact finite TRAIN chunk commands; no data writes/inference by Codex in this task, no DEV/TEST access, no downloads/tests/training. Existing 506 audio files remain intact; one Parkinson `PRAAT_EXIT 255` sample must be tracked as blocked, not silently retried. Full TRAIN production requires manager code review, then explicit owner execution. See `docs/NEXT_TASK_EN.md`.

**Latest TASK-002B owner correction:** Ubuntu reboot resolved NVIDIA driver/NVML mismatch. Cancel attempted CPU `torch==2.11.0+cpu`/`torchaudio==2.11.0+cpu` installation; preserve/recover verified matching CUDA-enabled PyTorch and torchaudio in the existing `.venv`, confirm `torch.cuda.is_available() == True` before bounded DiariZen GPU inference. CPU wheels, CPU fallback and additional environments are prohibited; `pyproject.toml` stays unchanged. Actual DiariZen results still pending.

**Latest TASK-002B workflow:** Owner will install dependencies and download weights personally. Codex is restricted to preparing an exact verified shell-command handoff and an **unexecuted** `scripts/audio/cache_diarizen_weights.py` helper plus bounded gate code/report. Await owner's GPU/CUDA dependencies and model-cache outputs before manager authorizes any DiariZen inference; no Codex installations/downloads/inference in preparation phase. Existing `.venv` only; no CPU-only Torch.

## TASK-001B outcome

Created the inactive `src/carewsm/` namespace, future `configs/carewsm/` root,
legacy-isolation documentation, and bounded scaffold tests. Existing WSM-derived
code, configurations, registrations, scripts, and tests remain untouched.
TASK-001B accepted and merged in PR #5: https://github.com/suslyash/CAREWSM/pull/5 ; merge SHA `b23b686fe4a143d0cd8804e232110ccb8731924a`. Codex reported standalone import PASS and `git diff --check` PASS; `pytest` was unavailable (`No module named pytest`), so a pytest run is not claimed. **Owner corrected the source layout afterward; TASK-001C removes the unwanted scaffold.** Extraction, DiariZen, training, evaluation and CARE/WSM writes were not performed.


## TASK-001C outcome

Removed obsolete WSM source, top-level scripts, copied tests, nested README files, and the TASK-001B scaffold. Archived only historical model implementations under passive `src/legacy/`; preserved `src/common/callbacks/wsm_summary_callback.py` and `src/common/callbacks/wsm_segment_callback.py` as non-activating future candidates. The ChimeraML plugin entry point is retained as a no-op, and the package identity is CAREWSM with archived legacy packages excluded from installation.

Future manager-approved training configurations must document `checkpoint_callback`, `snapshot_callback`, `early_stopping_callback`, `wsm_summary_callback`, `mlflow_logger`, and `console_file_logger`; `wsm_segment_metrics_callback` remains optional pending review. No tests were created or run. **Manager PASS**, merged PR: https://github.com/suslyash/CAREWSM/pull/9 ; merge SHA `d732b7a2e83f50fa1b06018e589cc36012797793`. Remaining nested legacy archive and old configs are the explicitly scoped TASK-001D follow-up.


## TASK-001D outcome

Removed all 157 tracked WSM-era configuration files. Flattened the passive archive into `src/legacy/audio_models/`, `fusion_models/`, `video_models/`, `text_models/`, and `callbacks/`, preserving all 14 model implementation files present in `origin/main` and both historical callback sources without behavior changes or activation; the task wording count of 15 does not match the audited inventory.

Reduced direct project dependencies to `chimera-ml>=0.2.4`; removed legacy Torch/audio/video/ML dependencies including `mamba-ssm`. Existing CAREWSM project identity, no-op plugin entrypoint, and `legacy*` package exclusion remain unchanged. TASK-001D manager **PASS**. Merged PR: https://github.com/suslyash/CAREWSM/pull/12 ; merge SHA `db92aa6eab4f10f4c141c6b730feeee7b0c679bc`. Codex reported `pip install -e . --dry-run` and exact `pip install -e .` PASS; manager independently checked Git changes and representative source blob preservation but did not execute pip on Codex's machine. No tests, extraction, DiariZen, evaluation or training were run.


## TASK-002A outcome

TASK-002A read-only WSM TRAIN audio preflight was **accepted (PR #15)**. A bounded deterministic sample of five existing TRAIN
media files was probed with `ffprobe` according to Codex evidence: all had audio/video streams, AAC audio,
16 kHz sample rate, two channels, and durations from 41.8 to 79.864 seconds.
The CARE 16 kHz mono, participant-turn, and representative eGeMAPS/DisVoice/
XLS-R/TRILLsson contracts are documented in
`docs/TASK-002A_WSM_AUDIO_PREFLIGHT_EN.md`. No feature parity claim is made;
extraction, DiariZen, training, evaluation, tests, and corpus writes were not
performed. Manager independently reviewed the code/report and accepted the preflight; merged PR: https://github.com/suslyash/CAREWSM/pull/15 ; merge SHA `5f97a73fb4bf6f52fb7b7320d794df6a8134b061`. Local external data roots were not independently accessed by the manager. The five samples are all depression/CONTROL and do not demonstrate task-balanced coverage.


## TASK-002B bounded GPU outcome (manager reviewed — bounded GPU execution accepted; turn identities unverified)

Owner-authorized local CUDA DiariZen runtime setup required a separate Python 3.10 `.venv-diarizen` because upstream packages conflicted with the CAREWSM `.venv`. The owner accepted this specific exception; both environments and downloaded weights remain untracked. According to the implementer's reported run, exactly three TRAIN clips spanning DEPRESSION, CONTROL and PARKINSON were converted to temporary 16 kHz mono WAV and processed with locally cached DiariZen weights on GPU: inference-only RTF 0.023261–0.028673, 21–24 diarized segments and exactly one speaker detected in each. Respondent-vs-interviewer labels remain **unverified**; output is runtime feasibility, not turn-quality validation or across-corpus parity. The WSM/CARE files were reported unchanged, scratch WAV/RTTM files cleaned, no TEST access, tests, feature extraction or training. Manager reviewed Git source/report but could not independently rerun the local data or GPU inference.


**TASK-002C manager outcome:** PASS for the documented manual listening checklist, merged PR https://github.com/suslyash/CAREWSM/pull/23 ; merge SHA `9629135645c8f10745e3537ad006c83fb27d69bb`. The owner accesses this machine via SSH and cannot currently hear audio, so the human one-vs-multiple-speaker and respondent-identity checks are DEFERRED; none are claimed as verified. This limitation does not stop supervised, explicit provisional feature extraction; keep semantic parity qualified.

**Latest owner scope — audio extraction only, plus audio/multimodal modelling:** our extraction target is **11 CARE audio features**, not all 19 registry families. TASK-003A immediately implements seven eGeMAPS/DisVoice families with a bounded three-TRAIN-file WSM smoke, exact CARE descriptor content/schema and WSM `feature_dir` paths; TASK-003B handles four deep audio outputs and TASK-003C the audio-only batch after manager approval. The other **8 nonaudio features (5 video, 3 linguistic) are supplied by other teams**; later multimodal models consume those artifacts, but we do not generate, alter or provision their extraction. No CARE/WSM raw/manifest writes, no unapproved bulk run, and all potentially large downloads remain owner-executed. Speaker identity remains unverified.

**Manager outcome, TASK-002B:** PASS for bounded GPU feasibility, merged PR https://github.com/suslyash/CAREWSM/pull/21 ; SHA `6c5056bfe6b65424d386c1dfe0872b160e2433be`. Measured run results were documented by Codex; the manager verified committed source/report and preserved newer manager contracts during the merge, without independently rerunning external data. The three files all received one detected speaker, so DiariZen remains provisional pending qualitative speaker sanity. The owner approved a separate `.venv-diarizen` to resolve real Python/runtime incompatibilities. **Owner's permanent traffic rule:** Codex must supply commands, sources, versions and expected/unknown download sizes for any potentially large dependencies or weights; owner executes those downloads, not Codex.

## TASK-002C outcome

TASK-002C prepared a bounded owner listening checklist for exactly the same
three TRAIN recordings used by TASK-002B. It provides three short audio-only
`ffplay` windows per recording, centered on previously observed DiariZen speech
segments, and a response table distinguishing audible speaker count from
respondent identifiability. The intervals are model-derived and not ground
truth. Manual owner confirmation is **pending**. No new media, downloads,
installations, inference reruns, tests, extraction, training, or dataset writes
were performed. Evidence: `docs/TASK-002C_SPEAKER_SANITY_EN.md`.


**LATEST OWNER DECISION — no CARE replay:** the original CARE audio recordings are unavailable. Same-input numerical replay is **impossible and not required**. Numeric equivalence between our extraction outputs and historical CARE numeric values is **UNVERIFIED**, not PASS. Do not ask for original CARE audio again or block the remaining WSM audio feature extraction. The bounded 21 CSV files were verified for exact CARE column/order/schema and complete numeric finiteness (four blank values in one prosody file), and PR #27 merged, SHA `4f6fadffc29ade18bc252dff43fd9a7d469d7f60`. TASK-003B starts now; no unauthorized big downloads.

## TASK-003A historical first-pass review (resolved by corrective PASS/PR #27)

Codex commit `f04678bfa34cfaee0db7948f090b64a6b895dd2e` prepared the seven-family extractor, audio-only validator and report but **wrote no WSM features**. The audited TRAIN manifest `feature_dir` values are legacy `*/train_labels/*/features/*` inputs/metadata, not the authorized canonical CARE-compatible `WSM_NEW/extracted/<STRATUM>/<SUBGROUP>/PROFILE_<id>/VIDEO_<segment>/` destination. Do **not** edit manifests or write into legacy `feature_dir`. Resolve target output mapping with checked manifest columns against existing transcripts and validate paths/collisions.

Manager review found blocking code issues besides owner-uninstalled `opensmile` and `disvoice`: DisVoice file API must use `fmt='dataframe'`, numeric column headers must exactly match CARE rather than numbered placeholders, existing RTTM participant segments must apply CARE turn merging and interviewer boundaries, and all seven results need schema validation before same-filesystem atomic no-clobber output. Preserve `transcript.txt`, raw media, existing feature files and external-team modalities. Historical first-pass **FAIL** was later resolved, corrected source merged in PR #27; owner authorizes TASK-003B but NOT the full batch.


## TASK-003A corrective review outcome

The bounded smoke was completed successfully for exactly three TRAIN records spanning DEPRESSION, CONTROL and PARKINSON. All seven requested eGeMAPS/DisVoice CSV families were written (21 files total) at the manifest-derived WSM extracted destinations. NumPy and Praat blockers were resolved by the owner; the headless runner now uses a temporary Praat `--run` wrapper. This evidence establishes successful bounded extraction and CARE-compatible schema checks only; it does **not** establish numerical equivalence of WSM feature values to CARE because CARE raw audio was unavailable for same-code re-extraction. Existing WSM outputs are now subject to read-only audit and no-overwrite protection.


## TASK-003A final manager acceptance

**PASS — bounded WSM audio descriptor smoke only**, merged PR #27: https://github.com/suslyash/CAREWSM/pull/27 ; merge SHA `4f6fadffc29ade18bc252dff43fd9a7d469d7f60`. According to Codex's actual-data run and read-only validation, 7 exact CARE-layout descriptor families were present for each of 3 TRAIN clips (21 CSVs), with verified headers/column order, frame/turn metadata and numeric integrity. Four blank prosody cells were counted, no NaN/Inf. The manager independently reviewed committed GitHub code/report, not the remote corpus bytes. DiariZen returned one speaker per file, respondent identity remains unverified. The owner confirmed original CARE media are unavailable: therefore original CARE numerical output equivalence is **not testable**. This limitation must be documented and must **not** block further WSM audio-only extraction. TASK-003B now targets 4 remaining audio families on same 3 TRAIN files, no new dependency/model downloads by Codex.


## TASK-003B bounded outcome

Prepared the offline, no-download extractor and read-only validator for the four
remaining deep-audio families: XLS-R 300M, TRILLsson 4.1, Emotion MSP trajectories,
and Emotion MSP embeddings. The implementation follows the inspected CARE model
IDs, pooling, population-standard-deviation, trajectory-column and NPZ contracts,
uses canonical manifest-derived WSM destinations, stages on the destination
filesystem, validates all outputs, and refuses clobbering. The bounded preflight
selected the same DEPRESSION, CONTROL and PARKINSON TRAIN records. No deep files
were written because local transformers/TensorFlow Hub/model assets and the local
DiariZen RTTMs were unavailable; explicit per-family blockers were returned. The locally cached
XLS-R checkpoint was then used for the same three records, creating and validating
three `wav2vec_xlsr300m.npz` files on CUDA; DiariZen turns were temporary and no
RTTM was retained. TRILLsson and Emotion MSP remain blocked. No downloads, tests,
training, non-audio work, or changes to the accepted 21 CSVs were made. Original CARE audio remains unavailable, so numerical equality is
UNVERIFIED rather than a blocker. Evidence: `docs/TASK-003B_DEEP_AUDIO_EXTRACTION_EN.md`.


## TASK-003B final bounded outcome

Owner-provided local caches enabled completion of all four deep-audio families on
the same three TRAIN records. Three `wav2vec_xlsr300m.npz`, three
`trillsson4_1.npz`, three `emotionmsp_embeds.npz`, and three
`emotionmsp_trajectories.csv` files are present and validated. The all-audio
read-only validator passed all 11 families with exit code 0. DiariZen used
temporary mono 16 kHz audio and explicit cached weights; no RTTM or scratch audio
was retained. TRILLsson ran on CPU to avoid GPU memory contention, while XLS-R and
Emotion MSP used CUDA. Existing 21 CSVs, raw data, transcripts and manifests
were not changed. Numerical equality to historical CARE remains UNVERIFIED because
CARE original audio is unavailable.


## TASK-003B manager PASS / TASK-003C activated

PR #30 merged: https://github.com/suslyash/CAREWSM/pull/30 ; merge SHA `595d4ebd871a36592e395320a4d5c982aee5a027`. According to implementer and read-only local validator evidence, three XLS-R NPZ, three TRILLsson NPZ, three Emotion MSP embeddings NPZ and three trajectories CSV were added for the same three original TRAIN clips. Alongside 21 accepted descriptor CSVs, this yields **33 validated audio outputs across 11 CARE-registered audio families** for three records. GitHub manager independently reviewed code, NPZ/CSV validator and docs but could not access actual local bytes. XLS-R/MSP used CUDA and TRILLsson CPU for compatibility; GPU DiariZen ran on scratch mono 16 kHz and retained no RTTM. CARE original media absent means numerical historical CARE equivalence UNVERIFIABLE; respondent identity provisional. The owner prioritizes quick complete WSM AUDIO extraction; TASK-003C authorizes an incremental pilot on **at most 30 further distinct TRAIN clips** and resumable full-split scripts, NOT a full corpus execution by Codex. Owner-only downloads; no tests, training or nonaudio extraction.


## TASK-003C first-pass manager HOLD and corrective scope

Codex commit `50cd8474eb74722c0eaa5dfe817f6d5fa047d1fe` reported two sequential pilots and **37 new complete WSM TRAIN samples** = 407 new audio files, exceeding the cumulative owner/manager cap of 30. Combined with 3 previous accepted TRAIN samples, **40 complete × 11 audio families = 440 validated files**, according to a read-only audit on the remote machine. Seven additional previously attempted TRAIN records remained incomplete after DisVoice/Praat missing `tempVUVprosody*.txt` errors; latest batch correctly exited nonzero. Per-complete-record mean 527.807 seconds; peak RSS 15,228.5 MiB, CUDA allocation 4,982.4 MiB. All existing validated WSM audio files must be kept; this is a documented **scope violation** and does not permit more new pilot records. Manager review found current CLI `--offset` is applied **after complete rows are filtered**, making published sequential chunk commands skip eligible rows; cumulative cap not persisted across invocations; NPZ `turn_ids` not cross-compared against CSV timing; sidecar checks incomplete; 30 WAVs may be retained until features finish. Draft PR #32 is HOLD until bounded corrective code and VUV analysis. The **only** authorized further WSM outputs are an at-most-once retry of the **seven exact previously attempted blocked TRAIN IDs**, after safe allowlist-only dry-run and source safeguards. No new recordings, bulk extraction, DEV/TEST reads, tests or downloads by Codex.

## TASK-003C corrective review outcome

The corrective worker now fails closed for new samples in this branch. It has a
persistent budget receipt, rejects ordinary selection without an approved
receipt, applies pagination to stable manifest order before existing-file
filtering, supports exact `--sample-ids-file` plus mandatory
`--retry-blocked-only`, diagnoses Praat exit/stderr/output paths, processes one
record at a time, reuses one worker-local DiariZen pipeline, and cross-checks
CSV turn metadata, NPZ `turn_ids`, trajectories and cached sidecar boundaries.
Praat is guarded by a timeout but failures are not treated as timeout-only.

A read-only audit and dry-run selected exactly the seven manager-approved IDs
from `docs/NEXT_TASK_EN.md`; all seven were incomplete and no other IDs were
selected. The single authorized retry completed six records, adding 66 files
(6 × 11). One record remains blocked:
`PARKINSON__PARKINSON__-1N99QsfWRk__-1N99QsfWRk_003`. Its diagnostic log
records `PRAAT_EXIT 255` for `vuv_praat.praat` and the expected
`tempVUVprosody*.txt` was absent. This is a reproducible DisVoice/Praat output
failure, not a timeout; no zero/blank substitute was written. Retry time was
303.325 s total, 43.332 s mean.

The final read-only TRAIN audit found 46 complete records × 11 = 506 valid
feature files, with no noncomplete destination containing feature files. This
is 43 new complete records / 473 new files plus the previous 33 accepted files;
the previous 440 files remain untouched. The retry receipt is marked one-time and
prevents a second retry. No new records, DEV/TEST data, tests, training,
downloads or installations were used.


## TASK-003C final manager acceptance and TASK-003D

PR #32 merged: https://github.com/suslyash/CAREWSM/pull/32 ; merge SHA `55dcb7b4b8688a97b5247b728ca3acae622fbdd3`. Corrected branch `codex/task-003c-audio-batch` contains immutable-order pagination, a fail-closed 30-record historical scope ledger, exact seven-ID one-time retry, NPZ/CSV/sidecar turn checks, one worker-local DiariZen pipeline, one-record scratch processing and Praat exit/stderr diagnostics. According to the implementer's remote audit, **46 WSM TRAIN recordings** have **11 valid audio files each** = **506 verified outputs** (43 new complete clips, 473 new files, plus the original three/33 files). Six of seven authorized retries completed without touching previous 440 files; **one** `PARKINSON__PARKINSON__-1N99QsfWRk__-1N99QsfWRk_003` remains `blocked_disvoice_vuv`, showing `PRAAT_EXIT 255` and missing `tempVUVprosody*.txt`. Retry permission is exhausted. Earlier cap violation remains documented. The manager independently reviewed source/report in GitHub but not local WSM files; status is **PASS for bounded/corrected implementation**, NOT full corpus feature coverage or proof of CARE numerical identity/participant identity. TASK-003D now prepares **owner-operated production** mode and finite TRAIN commands **without any Codex extraction or DEV/TEST access**. Owner-only model downloads, audio extraction scope, no tests/training still apply.


## TASK-003D outcome — owner-operated production launcher prepared

The single authorized corrective attempt for `PARKINSON__PARKINSON__-1N99QsfWRk__-1N99QsfWRk_003` succeeded: the stale sidecar ended at 50.2125 s while decoded audio ended at 49.9759375 s; duration validation clipped only that impossible end boundary, preserving feature algorithms. Praat returned EXIT 0 and all 11 families were written and read-only validated in 38.940 s. Existing 506 outputs were not overwritten; the WSM audio-file count is now 517.

Added a separate owner-only, explicit-opt-in launcher for TRAIN+DEV+TEST with deterministic manifest fingerprints, chunk checkpoints, resume, lock, no-clobber and per-sample failure accounting. The full-corpus command was NOT run by Codex; DEV/TEST were not read during development. No downloads, tests, training or nonaudio extraction. See `docs/TASK-003D_AUDIO_PRODUCTION_EN.md`.


## TASK-003D production correction after Draft PR #36

The production launcher was corrected without corpus extraction: Praat now has an explicit Xvfb/headless preflight and the owner command inherits `xvfb-run`; cached turn boundaries are bounded against the audio stream duration and compared to saved feature timestamps with container-timebase tolerance without rewriting published files; production workers inspect only their selected chunk; TRAIN-only preflight lists immutable positions, IDs, canonical destinations and all 11 family statuses; progress state and summary are atomically saved after each chunk; the cursor remains unchanged for incomplete worker reports. Existing 517 outputs were not modified. Full production remains unrun by Codex, with no DEV/TEST access during development.


## TASK-003E — live WSM audio production diagnostics (manager correction)

The owner executed TASK-003D all-audio production launcher through `tmux` and supplied live first-chunk evidence; TRAIN manifest contains **6,133 eligible rows** and the batch's first 100 records are being processed. `summary.json` reports zero until the first chunk finishes because progress is committed per chunk, while worker `production_train.jsonl` records per-sample results immediately. **Two distinct error types** were observed: (1) false `cached sidecar turn boundaries do not match feature metadata` even with eleven `valid_existing` audio feature files (e.g., raw cached sidecar and eGeMAPS CSV `[0.0125,64.3725]` match exactly while `ffprobe` reports audio stream duration `64.231`, a 0.1415 s gap; current read-sidecar validation clips before comparing), and (2) genuine Praat `EXIT 255` with absent DisVoice `tempVUVprosody*` or `tempVUVarticulation*` output, cause not yet determined. Scientific warnings (`RankWarning`, loss of precision, TF32) are not automatically fatal. These observed errors do not prove damaged raw audio or numerical inconsistency of all 11 feature files. **TASK-003E** prepares algorithm-preserving sidecar/timebase validation and source-evidence-based Praat correction. Codex shall not run a second extraction or change the active production checkout. All published audio artifacts/checkpoints preserved. Historical CARE numeric equality and human respondent identity remain UNVERIFIABLE/PROVISIONAL.
