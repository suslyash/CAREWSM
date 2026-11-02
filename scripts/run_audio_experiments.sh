#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

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

list_features() {
  local directory
  for directory in configs/audio/*/; do
    [[ -d "$directory" ]] || continue
    printf '%s:' "${directory%/}" | sed 's#configs/audio/##'
    printf ' %s' "${directory}"*_fold[0-4].yaml
    printf '\n'
  done
}

run_feature() {
  local feature="$1"
  local model="${2:-}"
  local directory="configs/audio/${feature}"
  [[ -d "$directory" ]] || { printf 'Unknown feature folder: %s\n' "$feature" >&2; return 2; }
  local configs=()
  local fold config
  if [[ -n "$model" ]]; then
    for fold in 0 1 2 3 4; do
      config="${directory}/${model}_fold${fold}.yaml"
      [[ -f "$config" ]] || { printf 'Missing config: %s\n' "$config" >&2; return 2; }
      configs+=("$config")
    done
  else
    configs=("${directory}"/*_fold[0-4].yaml)
    [[ "${#configs[@]}" -gt 0 ]] || { printf 'No fold configs in %s\n' "$directory" >&2; return 2; }
  fi
  for config in "${configs[@]}"; do
    printf 'Starting %s\n' "$config"
    CUDA_VISIBLE_DEVICES=0 .venv/bin/chimera-ml train -c "$config"
  done
}

[[ $# -gt 0 ]] || { usage; exit 0; }
feature=""
model=""
all_features=false
list_only=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) list_only=true; shift ;;
    --all) all_features=true; shift ;;
    --feature) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; feature="$2"; shift 2 ;;
    --model) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; model="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if "$list_only"; then
  [[ -z "$feature" && -z "$model" && "$all_features" == false ]] || { usage >&2; exit 2; }
  list_features
  exit 0
fi
if "$all_features"; then
  [[ -z "$feature" && -z "$model" ]] || { usage >&2; exit 2; }
  for directory in configs/audio/*/; do
    [[ -d "$directory" ]] || continue
    run_feature "${directory#configs/audio/}"
  done
  exit 0
fi
[[ -n "$feature" ]] || { usage >&2; exit 2; }
run_feature "$feature" "$model"
