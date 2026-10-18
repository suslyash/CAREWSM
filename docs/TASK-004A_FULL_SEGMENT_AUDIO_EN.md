# TASK-004A — full-VIDEO audio replacement

The active policy treats every existing WSM `VIDEO` as a complete audio segment. The new launcher decodes each video once to mono 16 kHz, uses one synthetic turn `turn_id=1`, `turn_start_sec=0`, and `turn_end_sec=decoded PCM duration`, then produces the 11 registered audio families. It does not import or invoke DiariZen, read RTTM files, select speakers, or require DiariZen weights.

`scripts/audio/wsm_reset_audio_outputs.py` has an exact allowlist containing only the 11 audio artifacts and `_carewsm_audio_turns.json`. It refuses symlinks, path escapes, missing confirmation, and any active legacy production session. It writes a reset receipt outside WSM and never removes directories, videos, transcripts, manifests, or non-audio artifacts.

The owner must first stop the old session:

```bash
tmux kill-session -t carewsm-audio-prod
```

The Codex read-only check found that session still present (a shell, with no active extractor descendant), so no deletion or extraction was performed. The new runtime is separate: `/media/maxim/Programs/CAREWSM-runtime/full-segment-audio`.

## Owner command

After stopping the old session and confirming the local model paths, run this single command. It performs the guarded allowlisted reset and then TRAIN, DEV and TEST in order. The same command resumes from atomic per-record state after an interruption; it does not overwrite existing files.

```bash
tmux new-session -d -s carewsm-full-segment-audio "cd /media/maxim/Programs/Projects/CAREWSM && export CAREWSM_ALLOW_PRODUCTION=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && xvfb-run -a --server-args='-screen 0 1024x768x24' .venv-diarizen/bin/python scripts/audio/wsm_run_all_audio.py --production --all-splits --reset-audio --confirm-replace-diarizen-outputs --wsm-root /media/maxim/Databases/WSM_NEW --care-root /media/maxim/Databases/CARE --runtime /media/maxim/Programs/CAREWSM-runtime/full-segment-audio --device cuda:0 --xlsr-model /media/maxim/Programs/ml_cache/torch/hub/checkpoints/wav2vec2_xlsr_300m.pth --trillsson-model /media/maxim/Programs/ml_cache/tensorflow_hub/c2d7e6e34916517a6bfdcf0536dd4e4241529356 --emotion-model /media/maxim/Programs/ml_cache/huggingface/models--audeering--wav2vec2-large-robust-12-ft-emotion-msp-dim/snapshots/6eba34a2485ea31cb03600241787c3a5edab8626"
```

The command is intentionally not run by Codex. Before launching, use read-only preflight mode with the same paths and omit all production/reset flags. Progress is in `summary.json` and `state/{train,dev,test}.json`; attach with `tmux attach -t carewsm-full-segment-audio`. Re-running the command after interruption uses the saved cursor and per-record results.

No full-corpus output has been produced under this policy by Codex. Historical DiariZen-based artifacts remain untouched until the owner executes the guarded reset. Historical CARE numerical equivalence is not claimed.
