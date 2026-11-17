#!/usr/bin/env bash
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"

mkdir -p logs/audio_feature_extraction

for pair in \
  'deep interp_B_mean_gated64_equal' \
  'deep_phon interp_BP_mean_gated64_equal' \
  'deep_egemaps_phon_artic interp_BEPA_mean_gated64_equal'; do
  read -r config_name ledger_name <<<"$pair"
  for fold in 0 1 2 3 4; do
    checkpoint=$(awk -F, -v name="$ledger_name" -v fold="$fold" \
      '$1 == name && $3 == fold { print $9 }' \
      reports/audio_interpretable_feature_fusion/provenance_ledger.csv)
    if [[ ! -f "$checkpoint" ]]; then
      echo "missing selected checkpoint: $ledger_name fold $fold" >&2
      exit 1
    fi
    echo "=== ${config_name} fold ${fold} ==="
    .venv/bin/chimera-ml eval \
      --config-path "configs/audio_feature_extraction/${config_name}_fold${fold}.yaml" \
      --checkpoint-path "$checkpoint" \
      --with-features 2>&1 | tee "logs/audio_feature_extraction/${config_name}_fold${fold}.eval.log"
  done
done
