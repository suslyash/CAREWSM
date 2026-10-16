#!/usr/bin/env bash
set -euo pipefail

CHIMERA="${CHIMERA:-./.venv/bin/chimera-ml}"
BASE="${BASE:-configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml}"
MAX_TRIALS="${MAX_TRIALS:-}"
STAGE="${1:-all}"

mkdir -p logs/wsm_audio_optuna

run_sweep() {
  local name="$1"
  local sweep="$2"
  shift 2

  local args=(
    sweep
    --base-config "${BASE}"
    --sweep-config "${sweep}"
    --sweep-name "${name}"
  )
  if [[ -n "${MAX_TRIALS}" ]]; then
    args+=(--max-trials "${MAX_TRIALS}")
  fi
  args+=("$@")

  "${CHIMERA}" "${args[@]}"
}

run_smoke() {
  echo "[stage 0] smoke run"
  "${CHIMERA}" sweep     --base-config "${BASE}"     --sweep-config "configs/audio_experiments/01_ssl_extractor_grid.yaml"     --sweep-name "wsm_audio_smoke"     --max-trials 1
}

run_ssl() {
  echo "[stage 1] SSL extractor / layer / pooling grid"
  run_sweep "wsm_audio_ssl" "configs/audio_experiments/01_ssl_extractor_grid.yaml"
}

run_domain() {
  echo "[stage 1b] domain audio wrapper grid"
  run_sweep "wsm_audio_domain" "configs/audio_experiments/01b_domain_audio_wrapper_grid.yaml"
}

run_models() {
  echo "[stage 2] audio temporal model grid"
  run_sweep "wsm_audio_models" "configs/audio_experiments/02_audio_model_grid.yaml"
}

run_coarse() {
  echo "[stage 3] Optuna coarse search"
  run_sweep "wsm_audio_optuna_coarse" "configs/audio_experiments/03_audio_optuna_coarse.yaml"
}

run_refine() {
  echo "[stage 4] Optuna refine search"
  run_sweep "wsm_audio_optuna_refine" "configs/audio_experiments/04_audio_optuna_refine.yaml"
}

case "${STAGE}" in
  smoke) run_smoke ;;
  ssl) run_ssl ;;
  domain) run_domain ;;
  models) run_models ;;
  coarse) run_coarse ;;
  refine) run_refine ;;
  all)
    run_smoke
    run_ssl
    run_models
    run_coarse
    run_refine
    ;;
  *)
    echo "Usage: $0 [smoke|ssl|domain|models|coarse|refine|all]"
    exit 2
    ;;
esac

cat <<MSG
[done]
For staged selection, pass the best generated trial config to the next stage:
BASE=logs/.../_sweeps/.../trial_configs/best.yaml ./scripts/run_audio_experiments.sh models

The ssl stage already includes the domain-wrapper trials. Use domain only for a small isolated domain run.
MSG
