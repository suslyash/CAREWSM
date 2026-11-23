#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"
CHIMERA="$ROOT/.venv/bin/chimera-ml"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

usage() { printf '%s\n' 'Usage: bash scripts/fusion/run_all.sh --list|--check|--run|--resume [--matrix sweep]'; }
setup_matrix() {
  MATRIX="${1:-legacy}"
  case "$MATRIX" in
    legacy) CONFIG_DIR="$ROOT/configs/fusion"; LOG_ROOT="$ROOT/logs/fusion"; NAMESPACE=fusion; EXPECTED=15 ;;
    sweep) CONFIG_DIR="$ROOT/configs/fusion/sweep"; LOG_ROOT="$ROOT/logs/fusion"; NAMESPACE=sweep; EXPECTED=285 ;;
    *) echo "unknown matrix: $MATRIX" >&2; return 2 ;;
  esac
  LAUNCHER_DIR="$LOG_ROOT/launcher${MATRIX:+/$MATRIX}"
  LEDGER="$LAUNCHER_DIR/status.tsv"
}
configs() { find "$CONFIG_DIR" -maxdepth 1 -type f -name '*.yaml' -print | sort; }
list_configs() {
  mapfile -t files < <(configs)
  [[ ${#files[@]} -eq "$EXPECTED" ]] || { echo "expected $EXPECTED configs, found ${#files[@]}" >&2; return 1; }
  printf '%s\n' "${files[@]#"$ROOT/"}"
}
check_configs() {
  mapfile -t files < <(configs)
  [[ ${#files[@]} -eq "$EXPECTED" ]] || { echo "expected $EXPECTED configs, found ${#files[@]}" >&2; return 1; }
  "$PYTHON" - "${files[@]}" "$MATRIX" <<'PY'
import sys
from collections import Counter
from pathlib import Path
import yaml
import chimera_plugin
from chimera_ml.core.config import ExperimentConfig
chimera_plugin.register()
files = [Path(value) for value in sys.argv[1:-1]]
matrix = sys.argv[-1]
identities = set()
model_counts = Counter()
for path in files:
    config = yaml.safe_load(path.read_text())
    errors = ExperimentConfig.from_yaml(str(path)).validate(require_experiment_name=True)
    assert not errors, (path, errors)
    data = config["data"]["params"]
    model = config["model"]["name"]
    fold = data["fold"]
    run = config["experiment_info"]["params"]["run_name"]
    assert path.stem == run and 0 <= fold <= 4
    assert config["loss"]["name"] == "carewsm_masked_bce"
    assert (config["callbacks"][1]["params"]["log_path"] == "logs/fusion/sweep") if matrix == "sweep" else True
    assert config["logging"][0]["params"]["log_path"] == ("logs/fusion/sweep" if matrix == "sweep" else "logs/fusion")
    assert config["logging"][1]["params"]["tracking_uri"] == ("sqlite:///logs/fusion/sweep/mlflow.db" if matrix == "sweep" else "sqlite:///logs/fusion/mlflow.db")
    modalities = tuple(data["modalities"]) if "modalities" in data else ("audio", "video", "text")
    assert len(modalities) in (2, 3) and len(set(modalities)) == len(modalities)
    assert (tuple(config["model"]["params"]["modalities"]) == modalities) if matrix == "sweep" else True
    assert run not in identities, run
    identities.add(run)
    model_counts[model] += 1
assert len(identities) == (285 if matrix == "sweep" else 15)
assert all(count == (55 if matrix == "sweep" and model.startswith("carewsm_bimodal") else 30 if matrix == "sweep" else 5) for model, count in model_counts.items())
print(f"validated {len(identities)} unique {matrix} configurations")
PY
}

runtime_candidates() {
  local run="$1"
  local base="$LOG_ROOT/$NAMESPACE/$NAMESPACE"
  find "$base" -mindepth 1 -maxdepth 1 -type d \( -name "$run" -o -name "${run}_*" \) -print 2>/dev/null | sort
}
has_runtime() { [[ -n "$(runtime_candidates "$1")" ]]; }
verified() {
  local run="$1" candidates=() dir
  mapfile -t candidates < <(runtime_candidates "$run")
  [[ ${#candidates[@]} -eq 1 ]] || return 1
  dir="${candidates[0]}"
  [[ -f "$dir/train.log" && -f "$dir/summary.txt" ]] || return 1
  grep -q 'Training complete' "$dir/train.log" || return 1
  find "$dir/checkpoints" -maxdepth 1 -type f -name '*.pt' ! -name last.pt | grep -q .
}
check_existing() {
  mapfile -t files < <(configs)
  for config in "${files[@]}"; do
    local run
    run=$(basename "$config" .yaml)
    if has_runtime "$run"; then
      echo "runtime directory already exists: $run" >&2
      return 1
    fi
  done
}
run_all() {
  local mode="$1"
  if pgrep -af "[c]himera-ml train" >/dev/null; then
    echo "another Chimera training process is active" >&2
    return 1
  fi
  mkdir -p "$LAUNCHER_DIR"
  printf 'run\tstatus\texit_code\n' > "$LEDGER"
  mapfile -t files < <(configs)
  for config in "${files[@]}"; do
    local run archived code
    run=$(basename "$config" .yaml)
    if [[ "$mode" == resume ]] && verified "$run"; then
      printf '%s\tCOMPLETED\t0\n' "$run" >> "$LEDGER"
      continue
    fi
    if has_runtime "$run"; then
      mapfile -t candidates < <(runtime_candidates "$run")
      if (( ${#candidates[@]} > 1 )); then
        echo "ambiguous runtime directories for $run" >&2; return 1
      fi
      archived="$LAUNCHER_DIR/incomplete/${run}_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
      mkdir -p "$(dirname "$archived")"
      mv "${candidates[0]}" "$archived"
    fi
    set +e
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$CHIMERA" train -c "$config" > "$LAUNCHER_DIR/${run}.log" 2>&1
    code=$?
    set -e
    if (( code == 0 )) && verified "$run"; then
      printf '%s\tCOMPLETED\t0\n' "$run" >> "$LEDGER"
    else
      printf '%s\tFAILED\t%s\n' "$run" "$code" >> "$LEDGER"
      return "${code:-1}"
    fi
  done
}

[[ $# == 1 || $# == 3 ]] || { usage >&2; exit 2; }
ACTION="$1"
if [[ $# == 3 ]]; then [[ "$2" == "--matrix" ]] || { usage >&2; exit 2; }; setup_matrix "$3"; else setup_matrix legacy; fi
case "$ACTION" in
  --list) list_configs ;;
  --check) check_configs; [[ "$MATRIX" == legacy ]] && check_existing || true ;;
  --run) check_configs; check_existing; run_all run ;;
  --resume) check_configs; run_all resume ;;
  *) usage >&2; exit 2 ;;
esac
