#!/usr/bin/env bash
set -euo pipefail

series_dir="configs/audio/series"
series_names=(mptsnet_native_egemaps_lld mdformer_native_egemaps_lld)

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_audio_experiments.sh --list
  scripts/run_audio_experiments.sh --series mptsnet_native_egemaps_lld
  scripts/run_audio_experiments.sh --series mdformer_native_egemaps_lld
  scripts/run_audio_experiments.sh --all

Runs one selected YAML series across canonical folds 0-4 through Chimera-ML.
No arguments print this help and launch nothing.
USAGE
}

list_series() {
  printf '%s\n' "${series_names[@]}"
}

config_for_series() {
  case "$1" in
    mptsnet_native_egemaps_lld)
      printf '%s\n' "$series_dir/audio_mptsnet_native_egemaps_lld.yaml" ;;
    mdformer_native_egemaps_lld)
      printf '%s\n' "$series_dir/audio_mdformer_native_egemaps_lld.yaml" ;;
    *)
      printf 'Unknown series: %s\n' "$1" >&2
      return 2 ;;
  esac
}

run_series() {
  local series="$1"
  local base_config
  base_config="$(config_for_series "$series")"

  for fold in 0 1 2 3 4; do
    local run_config
    run_config="$(mktemp --suffix=.yaml)"
    trap 'rm -f "$run_config"' RETURN
    sed "s/fold0/fold${fold}/g" "$base_config" > "$run_config"
    printf 'Starting %s fold %s\n' "$series" "$fold"
    CUDA_VISIBLE_DEVICES=0 .venv/bin/chimera-ml train -c "$run_config"
    rm -f "$run_config"
    trap - RETURN
  done
}

if [[ $# -eq 0 ]]; then
  usage
  exit 0
fi

case "$1" in
  --list)
    [[ $# -eq 1 ]] || { usage >&2; exit 2; }
    list_series
    ;;
  --series)
    [[ $# -eq 2 ]] || { usage >&2; exit 2; }
    run_series "$2"
    ;;
  --all)
    [[ $# -eq 1 ]] || { usage >&2; exit 2; }
    for series in "${series_names[@]}"; do
      run_series "$series"
    done
    ;;
  --help|-h)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
