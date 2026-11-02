#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_audio_experiments.sh --list
  scripts/run_audio_experiments.sh --check --feature egemaps
  scripts/run_audio_experiments.sh --check --all
  scripts/run_audio_experiments.sh --feature egemaps
  scripts/run_audio_experiments.sh --feature egemaps --model mptsnet
  scripts/run_audio_experiments.sh --feature egemaps_func --model mlp
  scripts/run_audio_experiments.sh --all

No arguments print this help and launch nothing.
USAGE
}

check_configs() {
  local configs=("$@")
  [[ "${#configs[@]}" -gt 0 ]] || { printf 'No configs selected\n' >&2; return 2; }
  python3 - "${configs[@]}" <<'PY2'
from pathlib import Path
import re
import sys
import yaml
expected = {
    "egemaps": ("egemaps_lld", 25, False), "egemaps_func": ("egemaps_func", 88, True),
    "disvoice_prosody_static": ("disvoice_prosody_static", 103, True), "disvoice_prosody_dynamic": ("disvoice_prosody_dynamic", 13, False),
    "disvoice_phonation_static": ("disvoice_phonation_static", 28, True), "disvoice_phonation_dynamic": ("disvoice_phonation_dynamic", 7, False),
    "disvoice_articulation_static": ("disvoice_articulation_static", 488, True), "wav2vec_xlsr300m": ("wav2vec_xlsr300m", 1024, True),
    "trillsson4_1": ("trillsson4_1", 1024, True), "emotionmsp_trajectories": ("emotionmsp_trajectories", 3, True), "emotionmsp_embeds": ("emotionmsp_embeds", 1024, True),
}
models = {"mptsnet": "carewsm_mptsnet", "mdformer": "carewsm_md_former", "mlp": "carewsm_pooled_mlp"}
callbacks = {"carewsm_val_checkpoint_callback", "snapshot_callback", "carewsm_summary_callback", "carewsm_val_early_stopping_callback"}
for raw_path in sys.argv[1:]:
    path = Path(raw_path); match = re.fullmatch(r"(.+)/(mptsnet|mdformer|mlp)_fold([0-4])\.yaml", str(path))
    if match is None: raise SystemExit(f"Invalid config filename: {path}")
    folder, model, fold_text = match.groups(); feature = Path(folder).name
    if feature not in expected: raise SystemExit(f"Unknown feature folder: {feature}")
    family, dimension, is_static = expected[feature]
    allowed = {"mdformer", "mlp"} if is_static else {"mptsnet", "mdformer"}
    if model not in allowed: raise SystemExit(f"{path}: model {model} is not allowed for {feature}")
    with path.open(encoding="utf-8") as handle: config = yaml.safe_load(handle)
    data = config["data"]["params"]; model_config = config["model"]
    if data.get("fold") != int(fold_text) or data.get("feature_family") != family: raise SystemExit(f"{path}: fold/family mismatch")
    if model_config.get("name") != models[model] or model_config["params"].get("input_dim") != dimension: raise SystemExit(f"{path}: model/dimension mismatch")
    if data.get("care_root") != "/media/maxim/Databases/CARE" or data.get("wsm_root") != "/media/maxim/Databases/WSM_NEW": raise SystemExit(f"{path}: data roots mismatch")
    if config.get("train", {}).get("params", {}).get("device") != "cuda": raise SystemExit(f"{path}: device is not CUDA")
    callback_map = {item.get("name"): item.get("params", {}) for item in config.get("callbacks", [])}
    if set(callback_map) != callbacks: raise SystemExit(f"{path}: callback set mismatch")
    for name in ("carewsm_val_checkpoint_callback", "carewsm_val_early_stopping_callback"):
        if callback_map[name].get("monitor") != "joint/average_macro_f1" or callback_map[name].get("mode") != "max": raise SystemExit(f"{path}: invalid VAL monitor")
    if callback_map["carewsm_val_checkpoint_callback"].get("save_top_k") != 1: raise SystemExit(f"{path}: checkpoint selection is not best-only")
    if {item.get("name") for item in config.get("logging", [])} != {"console_file_logger", "mlflow_logger"}: raise SystemExit(f"{path}: logger set mismatch")
    label = "STATIC_INPUT / TEMPORAL_INTERPRETATION_LIMITED" if is_static else "SEQUENCE_INPUT"
    print(f"OK {path}: fold={fold_text} model={model} family={family} dim={dimension} {label}")
PY2
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
  if [[ -n "$model" ]]; then
    if [[ "$feature" == "egemaps" || "$feature" == *dynamic ]]; then
      [[ "$model" == "mptsnet" || "$model" == "mdformer" ]] || { printf 'Model %s is not allowed for temporal feature %s\n' "$model" "$feature" >&2; return 2; }
    else
      [[ "$model" == "mlp" || "$model" == "mdformer" ]] || { printf 'Model %s is not allowed for static feature %s\n' "$model" "$feature" >&2; return 2; }
    fi
  fi
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
check_only=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) list_only=true; shift ;;
    --check) check_only=true; shift ;;
    --all) all_features=true; shift ;;
    --feature) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; feature="$2"; shift 2 ;;
    --model) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; model="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if "$list_only"; then
  [[ -z "$feature" && -z "$model" && "$all_features" == false && "$check_only" == false ]] || { usage >&2; exit 2; }
  list_features
  exit 0
fi
if "$check_only"; then
  [[ "$all_features" == true || -n "$feature" ]] || { usage >&2; exit 2; }
  [[ -z "$feature" || -d "configs/audio/${feature}" ]] || { printf 'Unknown feature folder: %s\n' "$feature" >&2; exit 2; }
  if [[ -n "$model" ]]; then
    if [[ "$feature" == "egemaps" || "$feature" == *dynamic ]]; then
      [[ "$model" == "mptsnet" || "$model" == "mdformer" ]] || { printf 'Model %s is not allowed for temporal feature %s\n' "$model" "$feature" >&2; exit 2; }
    else
      [[ "$model" == "mlp" || "$model" == "mdformer" ]] || { printf 'Model %s is not allowed for static feature %s\n' "$model" "$feature" >&2; exit 2; }
    fi
  fi
  configs=()
  if "$all_features"; then
    [[ -z "$feature" && -z "$model" ]] || { usage >&2; exit 2; }
    for directory in configs/audio/*/; do [[ -d "$directory" ]] || continue; configs+=("${directory}"*_fold[0-4].yaml); done
  else
    directory="configs/audio/${feature}"
    if [[ -n "$model" ]]; then for fold in 0 1 2 3 4; do configs+=("${directory}/${model}_fold${fold}.yaml"); done
    else configs+=("${directory}"/*_fold[0-4].yaml); fi
  fi
  check_configs "${configs[@]}"
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
