#!/usr/bin/env bash
set -euo pipefail

base_config="configs/audio/series/audio_mdformer_native_egemaps_lld.yaml"
for fold in 0 1 2 3 4; do
  run_config="$(mktemp --suffix=.yaml)"
  trap 'rm -f "$run_config"' EXIT
  sed "s/fold0/fold${fold}/g" "$base_config" > "$run_config"
  echo "START audio_mdformer_native_egemaps_lld_fold${fold}"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/chimera-ml train -c "$run_config"
  source_dir="$(find "logs/audio_mdformer_native_egemaps_lld_fold${fold}" -mindepth 1 -maxdepth 1 -type d -print | sort | tail -n 1)"
  test -n "$source_dir"
  target="logs/audio_mdformer_native_egemaps_lld_fold${fold}"
  if [[ -e "$target/train.log" ]]; then
    suffix=2
    while [[ -e "${target}_${suffix}/train.log" ]]; do suffix=$((suffix + 1)); done
    target="${target}_${suffix}"
  fi
  mv "$source_dir" "$target"
  rmdir "logs/audio_mdformer_native_egemaps_lld_fold${fold}" 2>/dev/null || true
  rm -f "$run_config"
  trap - EXIT
  echo "DONE $target"
done
