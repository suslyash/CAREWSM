#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_ROOT="$REPO_ROOT/configs/audio_representation_pilot/deep008/followup_af"
LOG_ROOT="$REPO_ROOT/logs/audio_representation_pilot"
LAUNCH_ROOT="$LOG_ROOT/deep008_af_followup_launcher"
LAUNCH_LOG="$LAUNCH_ROOT/launcher.log"
PROGRESS_LOG="$LAUNCH_ROOT/progress.tsv"
PYTHON="$REPO_ROOT/.venv/bin/python"
CHIMERA="$REPO_ROOT/.venv/bin/chimera-ml"
SESSION_NAME="carewsm-deep008-af-followup"

configs=(
    "$CONFIG_ROOT/A_native_mlp64_fold1.yaml"
    "$CONFIG_ROOT/F_gated_emotion_trillsson_native64_fold1.yaml"
    "$CONFIG_ROOT/A_native_mlp64_fold2.yaml"
    "$CONFIG_ROOT/F_gated_emotion_trillsson_native64_fold2.yaml"
    "$CONFIG_ROOT/A_native_mlp64_fold3.yaml"
    "$CONFIG_ROOT/F_gated_emotion_trillsson_native64_fold3.yaml"
    "$CONFIG_ROOT/A_native_mlp64_fold4.yaml"
    "$CONFIG_ROOT/F_gated_emotion_trillsson_native64_fold4.yaml"
)

if [[ $# -ne 1 || ( "$1" != "--check" && "$1" != "--all" ) ]]; then
    printf 'Usage: scripts/run_audio_deep008_af_followup.sh --check|--all\n' >&2
    exit 2
fi
mode=$1

run_identity() {
    "$PYTHON" - "$1" <<'PYCODE'
import sys
from pathlib import Path
import yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
params = config["experiment_info"]["params"]
print(params["experiment_name"])
print(params["run_name"])
PYCODE
}

runtime_directory() {
    mapfile -t identity < <(run_identity "$1")
    experiment_dir="$LOG_ROOT/${identity[0]}"
    expected="$experiment_dir/${identity[1]}"
    mapfile -t candidates < <(find "$experiment_dir" -mindepth 1 -maxdepth 1 -type d -name "${identity[1]}_*" -print 2>/dev/null | sort)
    if [[ -e "$expected" && ${#candidates[@]} -eq 0 ]]; then
        printf '%s\n' "$expected"
    elif [[ -e "$expected" || ${#candidates[@]} -gt 1 ]]; then
        printf 'AMBIGUOUS_COLLISION\n'
        return 1
    elif [[ ${#candidates[@]} -eq 1 ]]; then
        printf '%s\n' "${candidates[0]}"
    else
        printf '%s\n' "$expected"
    fi
}

mlflow_status() {
    "$PYTHON" - "$1" <<'PYCODE'
import sys
from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri="sqlite:///logs/audio_representation_pilot/mlflow.db")
experiment = client.get_experiment_by_name("audio_representation_pilot_deep008")
if experiment is None:
    print("NONE")
    raise SystemExit
runs = [run for run in client.search_runs([experiment.experiment_id]) if (run.data.tags.get("mlflow.runName") or "").startswith(sys.argv[1] + "_")]
if len(runs) != 1:
    print(f"COUNT:{len(runs)}")
else:
    print(runs[0].info.status)
    print(runs[0].info.run_id)
PYCODE
}

is_completed() {
    local config=$1
    local run_dir=$2
    [[ -f "$run_dir/train.log" && -f "$run_dir/summary.txt" ]] || return 1
    [[ -n $(find "$run_dir/checkpoints" -maxdepth 1 -type f -name '*.pt' -print -quit 2>/dev/null) ]] || return 1
    rg -q 'Training complete' "$run_dir/train.log" || return 1
    mapfile -t identity < <(run_identity "$config")
    mapfile -t status < <(mlflow_status "${identity[1]}")
    [[ "${status[0]}" == "FINISHED" && -n "${status[1]:-}" ]]
}

check_population() {
    "$PYTHON" - "$REPO_ROOT" <<'PYCODE'
import csv
import sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = {}
with (root / "reports/feature_audit/sample_feature_availability.csv").open(newline="", encoding="utf-8-sig") as handle:
    for row in csv.DictReader(handle):
        manifest[(row["corpus"], row["sample_id"])] = row
expected = {1: (7027, 345, 1407), 2: (7045, 335, 1399), 3: (7020, 331, 1428), 4: (7026, 350, 1403)}
for fold, target in expected.items():
    counts = []
    for split in ("train", "val"):
        for corpus, data_root in (("CARE", "/media/maxim/Databases/CARE"), ("WSM", "/media/maxim/Databases/WSM_NEW")):
            path = Path(data_root) / "splits/folds" / f"fold_{fold}_{split}.csv"
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8-sig")))
            valid = [row for row in rows if all(manifest[(corpus, row["sample_id"])].get(family + "_status") == "VALID" for family in ("emotionmsp_embeds", "trillsson4_1"))]
            counts.append(len(valid))
    actual = (counts[0] + counts[1], counts[2], counts[3])
    if actual != target:
        raise SystemExit(f"fold {fold} population mismatch: {actual} != {target}")
    print(f"fold {fold}: joint_train={actual[0]} care_val={actual[1]} wsm_val={actual[2]}")
PYCODE
}

[[ -x "$PYTHON" && -x "$CHIMERA" ]] || { printf 'Repository .venv Chimera-ML installation is required.\n' >&2; exit 1; }
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "d2b3ed80ff578bfab6f21be5cce4f6e81a70a4cb"* || -n "${DEEP008_FOLLOWUP_ALLOW_SOURCE_SHA:-}" ]] || true
for config in "${configs[@]}"; do
    [[ -f "$config" ]] || { printf 'Missing config: %s\n' "$config" >&2; exit 1; }
    "$CHIMERA" validate-config -c "$config" >/dev/null
    mapfile -t identity < <(run_identity "$config")
    [[ "${identity[0]}" == "audio_representation_pilot_deep008" ]] || exit 1
    [[ "${identity[1]}" == A_*_fold[1-4] || "${identity[1]}" == F_*_fold[1-4] ]] || exit 1
    rg -q 'monitor: val_mean_corpora_tasks/macro_recall' "$config" || exit 1
    runtime_directory "$config" >/dev/null
done
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
if pgrep -af 'chimera-ml train' | rg -v 'rg ' >/dev/null; then
    printf 'An existing Chimera training process is active.\n' >&2
    exit 1
fi
check_population

if [[ "$mode" == "--check" ]]; then
    printf 'Preflight passed; no training was launched.\n'
    exit 0
fi

mkdir -p "$LAUNCH_ROOT"
if [[ ! -e "$PROGRESS_LOG" ]]; then
    printf 'configuration\tlaunch_utc\tstate\texit_code\trun_directory\n' > "$PROGRESS_LOG"
fi
printf '[%s] launcher mode=%s session=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$mode" "$SESSION_NAME" | tee -a "$LAUNCH_LOG"

for config in "${configs[@]}"; do
    mapfile -t identity < <(run_identity "$config")
    run_dir=$(runtime_directory "$config")
    launch_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    if [[ -e "$run_dir" ]]; then
        if is_completed "$config" "$run_dir"; then
            printf '%s\t%s\tSKIPPED_COMPLETED\t0\t%s\n' "$config" "$launch_utc" "$run_dir" >> "$PROGRESS_LOG"
            continue
        fi
        printf '%s\t%s\tAMBIGUOUS_COLLISION\t1\t%s\n' "$config" "$launch_utc" "$run_dir" >> "$PROGRESS_LOG"
        exit 1
    fi
    if [[ "$(mlflow_status "${identity[1]}" | head -1)" != "COUNT:0" ]]; then
        printf 'Existing MLflow identity for %s; stopping.\n' "${identity[1]}" >&2
        exit 1
    fi
    printf '%s\t%s\tSTARTED\t-\t%s\n' "$config" "$launch_utc" "$run_dir" >> "$PROGRESS_LOG"
    set +e
    CUDA_VISIBLE_DEVICES=0 "$CHIMERA" train -c "$config" 2>&1 | tee -a "$LAUNCH_LOG"
    exit_code=${PIPESTATUS[0]}
    set -e
    if [[ $exit_code -ne 0 ]]; then
        printf '%s\t%s\tFAILED\t%d\t%s\n' "$config" "$launch_utc" "$exit_code" "$run_dir" >> "$PROGRESS_LOG"
        exit "$exit_code"
    fi
    run_dir=$(runtime_directory "$config")
    if ! is_completed "$config" "$run_dir"; then
        printf '%s\t%s\tAMBIGUOUS_AFTER_EXIT\t0\t%s\n' "$config" "$launch_utc" "$run_dir" >> "$PROGRESS_LOG"
        exit 1
    fi
    printf '%s\t%s\tCOMPLETED\t0\t%s\n' "$config" "$launch_utc" "$run_dir" >> "$PROGRESS_LOG"
done
printf '[%s] launcher finished successfully; selected=%d\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${#configs[@]}" | tee -a "$LAUNCH_LOG"
