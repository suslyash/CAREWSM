#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_audio_experiments.sh --list
  scripts/run_audio_experiments.sh --feature egemaps
  scripts/run_audio_experiments.sh --feature egemaps --model mptsnet
  scripts/run_audio_experiments.sh --all

No arguments print this help and launch nothing.
USAGE
}

list_series() {
  printf '%s\n' "egemaps/mptsnet_fold0..fold4" "egemaps/mdformer_fold0..fold4"
}

run_configs() {
  local feature="$1"
  local model="${2:-}"
  local pattern="configs/audio/${feature}/*_fold[0-4].yaml"
  local configs=( $pattern )
  [[ -d "configs/audio/${feature}" ]] || { printf 'Unknown feature: %s\n' "$feature" >&2; return 2; }
  if [[ -n "$model" ]]; then
    configs=( "configs/audio/${feature}/${model}_fold0.yaml" \
              "configs/audio/${feature}/${model}_fold1.yaml" \
              "configs/audio/${feature}/${model}_fold2.yaml" \
              "configs/audio/${feature}/${model}_fold3.yaml" \
              "configs/audio/${feature}/${model}_fold4.yaml" )
  fi
  local expected=10
  [[ -n "$model" ]] && expected=5
  [[ "${#configs[@]}" -eq "$expected" ]] || { printf 'Expected %s fold configs for %s/%s\n' "$expected" "$feature" "${model:-all-models}" >&2; return 2; }
  for config in "${configs[@]}"; do
    [[ -f "$config" ]] || { printf 'Missing config: %s\n' "$config" >&2; return 2; }
    printf 'Starting %s\n' "$config"
    CUDA_VISIBLE_DEVICES=0 .venv/bin/chimera-ml train -c "$config"
  done
}

[[ $# -gt 0 ]] || { usage; exit 0; }
feature=""
model=""
all_series=false
list_only=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) list_only=true; shift ;;
    --all) all_series=true; shift ;;
    --feature) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; feature="$2"; shift 2 ;;
    --model) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; model="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if "$list_only"; then
  [[ -z "$feature" && -z "$model" && "$all_series" == false ]] || { usage >&2; exit 2; }
  list_series
  exit 0
fi
if "$all_series"; then
  [[ -z "$feature" && -z "$model" ]] || { usage >&2; exit 2; }
  run_configs egemaps
  exit 0
fi
[[ -n "$feature" ]] || { usage >&2; exit 2; }
run_configs "$feature" "$model"
