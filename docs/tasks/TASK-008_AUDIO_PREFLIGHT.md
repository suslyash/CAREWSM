# TASK-008 — Audio experiment preflight and temporal/static reporting guard

## Goal
Make TASK-007's 110 literal YAMLs safe and understandable to launch by the owner. **Static-only** preparation: no training, no smoke runs, no MLflow runs, no cleanup, no TEST evaluations.

## Context
TASK-007 verified 11 audio feature families, including temporal sequences and turn-level static single-token `T=1` vectors. MPTSNet's period folding and MD-Former's temporal branch have different scientific meaning on `T=1`. We must retain the owner's 11-family config matrix without misleadingly treating static inputs as temporal-model validation. The owner runs experiments manually.

## Scope
1. Inspect current `docs/evidence/TASK-007_AUDIO_FEATURE_MATRIX_EN.md`, `src/audio/features/representations.py`, `src/audio/data/chimera_datamodule.py`, `src/audio/models/classifiers.py`, registry and manifest. Document a *checked* temporal vs static distinction for the 11 families. Show dimensions, timestamps/length semantics, eligibility, missingness counts if derivable from existing audit without loading all artifacts, and consequences for interpreting temporal-model runs. Never invent counts.
2. Add a lightweight **non-training** preflight mode to the one existing **Bash** launcher `scripts/run_audio_experiments.sh`, such as `--check` combined with `--feature` or `--all`, which validates the selected *literal* YAMLs, fold/model/family/dimension correctness, fixed paths, required registry names and expected callbacks. It should print which models/folds would run and clearly mark static `T=1` series as **STATIC_INPUT / TEMPORAL_INTERPRETATION_LIMITED**. Do not create a Python launcher or generate YAML dynamically. Minimal read-only Python helper may be used only for accurate YAML parsing if needed; launcher remains Bash.
3. Ensure the real launch path continues invoking the already existing YAMLs unchanged, five folds per chosen series, and respects user control; without args, help only. No seed sweep, no extra launchers, no monkey patches, no moving logs.
4. Provide one read-only command in README to preflight and one example owner-run command. Clarify static inputs are valid as an exploratory classifier setting but not comparable evidence of periodic/temporal architectural advantage. Do not delete static configs or automatically launch anything.
5. Verify `bash -n`, preflight for `--feature egemaps` and one static feature, optional `--all --check`, YAML validation where installed, and `git diff --check`. These are **read-only static checks**, not training. Modify only the launcher, README/evidence/progress and a tiny helper if demonstrably necessary.

## Handoff
Work on `codex/task-008-audio-preflight`, commit + normal push, no PR/merge, eight-heading handoff. Stop for manager review.

**All actual experiments remain owner-triggered only.**
