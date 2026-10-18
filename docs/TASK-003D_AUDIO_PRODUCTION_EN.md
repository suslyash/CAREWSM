# TASK-003D — audio production release

## Outcome

The single authorized corrective attempt for `PARKINSON__PARKINSON__-1N99QsfWRk__-1N99QsfWRk_003` completed successfully. The source MP4 is 49.9759375 seconds at 16 kHz stereo AAC; decoded temporary audio is finite mono 16 kHz. The stale cached sidecar ended at 50.2125 seconds, beyond the decoded audio. The repair validates decoded duration and clips only an impossible cached end boundary to the available audio end; it does not alter CARE algorithms, frame settings, or substitute values. Praat then returned `EXIT 0` for all DisVoice calls.

The sample produced and validated all 11 families in 38.940 seconds. The previous 506 files were not overwritten; the read-only count is now 517 (506 prior + 11 target). Historical CARE numerical equality remains UNVERIFIED because original CARE audio is unavailable. Respondent identity remains provisional.

## Structural parity and destination

| Family | WSM output | Structural check |
|---|---|---|
| eGeMAPS functionals | `egemaps_func.csv` | CARE header/order, finite numeric cells |
| eGeMAPS LLD | `egemaps_lld.csv` | CARE header/order, frame metadata, finite numeric cells |
| DisVoice prosody static | `disvoice_prosody_static.csv` | CARE header/order, finite numeric cells |
| DisVoice prosody dynamic | `disvoice_prosody_dynamic.csv` | CARE header/order, frame metadata, finite numeric cells |
| DisVoice phonation static | `disvoice_phonation_static.csv` | CARE header/order, finite numeric cells |
| DisVoice phonation dynamic | `disvoice_phonation_dynamic.csv` | CARE header/order, frame metadata, finite numeric cells |
| DisVoice articulation static | `disvoice_articulation_static.csv` | CARE header/order, finite numeric cells |
| XLS-R wav2vec 300M | `wav2vec_xlsr300m.npz` | keys `mean,std,turn_ids`; float32 1024-wide arrays |
| TRILLsson 4.1 | `trillsson4_1.npz` | keys `mean,std,turn_ids`; float32 1024-wide arrays |
| Emotion MSP trajectories | `emotionmsp_trajectories.csv` | registered six-column turn schema |
| Emotion MSP embeddings | `emotionmsp_embeds.npz` | keys `embed,turn_ids`; float32 1024-wide arrays |

The canonical destination is:
`WSM_NEW/extracted/PARKINSON/PARKINSON/PROFILE_-1N99QsfWRk/VIDEO_-1N99QsfWRk_003/`.

## Production launcher

`scripts/audio/wsm_run_all_audio.py` is read-only by default. Actual execution requires both `--production --all-splits` and `CAREWSM_ALLOW_PRODUCTION=1`. It reads TRAIN, then DEV, then TEST at owner runtime, fingerprints each immutable manifest, processes deterministic chunks of at most 100, persists per-split high-water positions and sample records, uses an exclusive runtime lock, and calls the existing batch worker with local-only model paths. It never invokes the old mutable `--all --offset` flow. The worker validates only the selected chunk's 11 families; it does not audit every manifest destination on every chunk. Existing valid 11-family destinations are validated and skipped without model loading; outputs are staged/no-clobber. Individual worker failures remain recorded and do not abort later chunks.

Before production, run the meaningful TRAIN-only read-only preflight with the same local paths and `--preflight --preflight-limit 3`. It reports the immutable TRAIN digest, exact positions/sample IDs, canonical destinations, source-media presence, all 11 existing-family checks, CUDA/package prerequisites, local model paths, and headless Praat availability. It never opens DEV/TEST.

Owner-run command (not executed by Codex):

```bash
tmux new-session -d -s carewsm-audio-prod "cd /media/maxim/Programs/Projects/CAREWSM && export CAREWSM_ALLOW_PRODUCTION=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && xvfb-run -a --server-args='-screen 0 1024x768x24' .venv-diarizen/bin/python scripts/audio/wsm_run_all_audio.py --production --all-splits --wsm-root /media/maxim/Databases/WSM_NEW --care-root /media/maxim/Databases/CARE --runtime /media/maxim/Programs/CAREWSM-runtime/task003d --batch-size 100 --device cuda:0 --weights /media/maxim/Programs/ml_cache/huggingface/models--BUT-FIT--diarizen-wavlm-large-s80-md/snapshots/6285693ddd5b38e8229acb93f864f3d04a82bee1 --embedding-weights /media/maxim/Programs/ml_cache/huggingface/blobs/db/dbae20d3c4f8fc381db5d514a7d4069c87d09feba054c2b248f7333b073e08cd --xlsr-model /media/maxim/Programs/ml_cache/torch/hub/checkpoints/wav2vec2_xlsr_300m.pth --trillsson-model /media/maxim/Programs/ml_cache/tensorflow_hub/c2d7e6e34916517a6bfdcf0536dd4e4241529356 --emotion-model /media/maxim/Programs/ml_cache/huggingface/models--audeering--wav2vec2-large-robust-12-ft-emotion-msp-dim/snapshots/6eba34a2485ea31cb03600241787c3a5edab8626"
```

Progress:

```bash
tail -f /media/maxim/Programs/CAREWSM-runtime/task003d/logs/train_*.log
cat /media/maxim/Programs/CAREWSM-runtime/task003d/summary.json
```

After a power loss, rerun the same tmux command. The manifest fingerprint and saved high-water index prevent silent drift; an interrupted chunk is safely replayed and valid files are skipped. The cursor advances only after every eligible ID in the worker report is present; a partial/invalid worker report leaves the cursor unchanged. Per-chunk logs and `summary.json` are atomically updated immediately after each chunk, so progress is visible while the process runs. Do not delete the runtime state or change manifests between runs. A nonzero final exit means unresolved blockers remain; exact blocked/missing IDs remain in the per-split state for an owner-controlled later retry policy.

## Operational limits

Praat is run through the inherited Xvfb display from the command above; the worker also fails closed if neither `xvfb-run` nor `DISPLAY` is available. No downloads or installations are performed. The first pilot measured approximately 527.8 seconds per complete record with high variability and peak RSS about 15.2 GiB; the six-record corrective retry is not a cold-run estimate. Reserve substantial free disk for CSV/NPZ outputs and temporary WAVs, and monitor GPU memory. The launcher reports exact sample IDs and statuses including complete, skipped, blocked, invalid existing, missing media, and unreported failures.

Full-corpus execution was not run by Codex. No DEV/TEST data was read during development, and no tests, training, video or linguistic extraction was performed.
