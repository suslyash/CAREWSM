#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
CONFIG_DIR="$ROOT/configs/fusion"
LOG_ROOT="$ROOT/logs/fusion"
LAUNCHER_DIR="$LOG_ROOT/launcher"
LEDGER="$LAUNCHER_DIR/status.tsv"
PYTHON="$ROOT/.venv/bin/python"
CHIMERA="$ROOT/.venv/bin/chimera-ml"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
usage() { printf '%s\n' 'Usage: bash scripts/fusion/run_all.sh --list|--check|--run|--resume'; }
configs() { find "$CONFIG_DIR" -maxdepth 1 -type f -name '*_fold[0-4].yaml' -print | sort; }
list_configs() {
  mapfile -t files < <(configs)
  [[ ${#files[@]} -eq 15 ]] || { echo "expected 15 fusion configs, found ${#files[@]}" >&2; return 1; }
  printf '%s\n' "${files[@]#"$ROOT/"}"
}
check() {
  [[ -x "$PYTHON" && -x "$CHIMERA" ]] || { echo 'missing .venv Chimera installation' >&2; return 1; }
  mapfile -t files < <(configs)
  [[ ${#files[@]} -eq 15 ]] || { echo "expected 15 fusion configs, found ${#files[@]}" >&2; return 1; }
  "$PYTHON" - "${files[@]}" <<'PY'
import sys
from pathlib import Path
import torch
import yaml
import chimera_plugin
from chimera_ml.core.config import ExperimentConfig
chimera_plugin.register()
expected_models = {"carewsm_text_guided_dual_gated", "carewsm_trimodal_cross_attention", "carewsm_shared_private_trimodal"}
seen = set()
for raw in sys.argv[1:]:
    path = Path(raw)
    config = yaml.safe_load(path.read_text())
    errors = ExperimentConfig.from_yaml(str(path)).validate(
        require_experiment_name=True
    )
    assert not errors, (path, errors)
    params = config["data"]["params"]
    model = config["model"]["name"]
    fold = params["fold"]
    key = (model, fold)
    assert model in expected_models and 0 <= fold <= 4 and key not in seen, key
    seen.add(key)
    assert (params["audio_source"], params["video_source"], params["text_source"]) == ("audio_id1", "video_4", "text_id9")
    assert config["model"]["params"]["audio_dim"] == 64 and config["model"]["params"]["video_dim"] == 128 and config["model"]["params"]["text_dim"] == 64
    assert config["experiment_info"]["params"]["run_name"] == path.stem
    assert config["loss"]["name"] == "carewsm_masked_bce"
    assert config["callbacks"][0]["name"] == "carewsm_validation_monitor"
    assert all("logs/fusion" in str(item["params"]["log_path"]) for item in config["callbacks"] if "log_path" in item.get("params", {}))
    assert config["logging"][1]["params"]["tracking_uri"] == "sqlite:///logs/fusion/mlflow.db"
assert len(seen) == 15
from fusion.data.carewsm_trimodal_datamodule import CAREWSMTrimodalDataModule
for fold in range(5):
    config = yaml.safe_load(Path(f"configs/fusion/text_guided_dual_gated_fold{fold}.yaml").read_text())
    module = CAREWSMTrimodalDataModule(**config["data"]["params"])
    for corpus, view_name in (("CARE", "val_care"), ("WSM", "val_wsm")):
        view = module.views[view_name]
        assert view, f"empty {corpus} validation view"
        record = view[0]
        assert all(Path(value).is_file() for value in record["artifact_paths"].values())
        assert record["target_provenance"] in {"CARE feature-audit target manifest", "WSM videos.csv y_* columns"}
    assert all(set(row["targets"]) == {"depression", "parkinson"} for row in module.views["val_care"])
print(f"validated {len(seen)} configs, five folds, and representative CARE/WSM artifacts with observed target masks")
PY
  "$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print(torch.cuda.get_device_name(0))
PY
  if pgrep -af 'chimera-ml train' | grep -v "$$"; then echo 'training process already active' >&2; return 1; fi
  for config in "${files[@]}"; do
    run=$(basename "$config" .yaml)
    if verified "$run"; then
      echo "completed run identity already exists: $run" >&2
      return 1
    fi
  done
  mkdir -p "$LAUNCHER_DIR"
  echo 'fusion preflight passed; no training was started'
}
verified() {
  local run="$1" dir="$LOG_ROOT/fusion/$run"
  [[ -f "$dir/train.log" && -f "$dir/summary.txt" ]] || return 1
  grep -q 'Training complete' "$dir/train.log" || return 1
  find "$dir/checkpoints" -maxdepth 1 -type f -name '*.pt' ! -name last.pt | grep -q .
}
run_all() {
  local mode="$1"
  mkdir -p "$LAUNCHER_DIR"
  printf 'run\tstatus\texit_code\n' > "$LEDGER"
  mapfile -t files < <(configs)
  for config in "${files[@]}"; do
    local run=$(basename "$config" .yaml)
    if [[ "$mode" == resume ]] && verified "$run"; then printf '%s\tCOMPLETED\t0\n' "$run" >> "$LEDGER"; continue; fi
    set +e
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$CHIMERA" train -c "$config" > "$LAUNCHER_DIR/${run}.log" 2>&1
    local code=$?
    set -e
    if (( code == 0 )) && verified "$run"; then printf '%s\tCOMPLETED\t0\n' "$run" >> "$LEDGER"; else printf '%s\tFAILED\t%s\n' "$run" "$code" >> "$LEDGER"; echo "fusion run failed: $run (exit $code)" >&2; return "$code"; fi
  done
}
[[ $# -eq 1 ]] || { usage >&2; exit 2; }
case "$1" in
  --list) list_configs ;;
  --check) check ;;
  --run) check && run_all run ;;
  --resume) run_all resume ;;
  *) usage >&2; exit 2 ;;
esac
