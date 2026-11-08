#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
CONFIG_ROOT="$REPO_ROOT/configs/audio_representation_pilot/deep008/task_attention"
LOG_ROOT="$REPO_ROOT/logs/audio_representation_pilot/deep008_task_attention"
EXPERIMENT=audio_representation_pilot_deep008_task_attention
MLFLOW_URI=sqlite:///logs/audio_representation_pilot/deep008_task_attention/mlflow.db
LAUNCH_ROOT="$LOG_ROOT/launcher"
LAUNCH_LOG="$LAUNCH_ROOT/launcher.log"
PROGRESS_LOG="$LAUNCH_ROOT/progress.tsv"
PYTHON="$REPO_ROOT/.venv/bin/python"
CHIMERA="$REPO_ROOT/.venv/bin/chimera-ml"
SESSION_NAME=carewsm-deep008-task-attention

configs=(
    "$CONFIG_ROOT/F_task_attention_fold0.yaml"
    "$CONFIG_ROOT/F_task_attention_fold1.yaml"
    "$CONFIG_ROOT/F_task_attention_fold2.yaml"
    "$CONFIG_ROOT/F_task_attention_fold3.yaml"
    "$CONFIG_ROOT/F_task_attention_fold4.yaml"
)

if [[ $# -ne 1 || ( "$1" != "--check" && "$1" != "--all" ) ]]; then
    printf 'Usage: scripts/run_audio_deep008_task_attention.sh --check|--all\n' >&2
    exit 2
fi
mode=$1

identity() {
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
    mapfile -t ids < <(identity "$1")
    base="$LOG_ROOT/${ids[0]}"
    expected="$base/${ids[1]}"
    mapfile -t candidates < <(
        find "$base" -mindepth 1 -maxdepth 1 -type d \
            -name "${ids[1]}_*" -print 2>/dev/null | sort
    )
    if [[ -e "$expected" || ${#candidates[@]} -gt 0 ]]; then
        printf 'COLLISION\n'
        return 1
    fi
    printf '%s\n' "$expected"
}

completed_runtime_directory() {
    mapfile -t ids < <(identity "$1")
    base="$LOG_ROOT/${ids[0]}"
    expected="$base/${ids[1]}"
    mapfile -t candidates < <(
        find "$base" -mindepth 1 -maxdepth 1 -type d \
            -name "${ids[1]}_*" -print 2>/dev/null | sort
    )
    if [[ -d "$expected" && ${#candidates[@]} -eq 0 ]]; then
        printf '%s\n' "$expected"
    elif [[ ! -e "$expected" && ${#candidates[@]} -eq 1 ]]; then
        printf '%s\n' "${candidates[0]}"
    else
        return 1
    fi
}

mlflow_status() {
    "$PYTHON" - "$1" "$MLFLOW_URI" <<'PYCODE'
import sys
from pathlib import Path

run_name, tracking_uri = sys.argv[1:]
database = Path(tracking_uri.removeprefix("sqlite:///"))
if not database.exists():
    print("COUNT:0")
    raise SystemExit

from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri=tracking_uri)
experiment = client.get_experiment_by_name("audio_representation_pilot_deep008_task_attention")
if experiment is None:
    print("COUNT:0")
    raise SystemExit
runs = [
    run
    for run in client.search_runs([experiment.experiment_id])
    if (run.data.tags.get("mlflow.runName") or "") == run_name
]
if len(runs) != 1:
    print(f"COUNT:{len(runs)}")
else:
    print(runs[0].info.status)
    print(runs[0].info.run_id)
PYCODE
}

assert_no_collision() {
    local config=$1
    runtime_directory "$config" >/dev/null
    mapfile -t ids < <(identity "$config")
    mapfile -t status < <(mlflow_status "${ids[1]}")
    [[ "${status[0]}" == "COUNT:0" ]] || {
        printf 'MLflow collision for %s.\n' "${ids[1]}" >&2
        return 1
    }
}

assert_completed() {
    local config=$1
    local directory
    directory=$(completed_runtime_directory "$config")
    [[ -f "$directory/train.log" && -f "$directory/summary.txt" ]]
    [[ -n $(find "$directory/checkpoints" -maxdepth 1 -type f -name '*.pt' -print -quit 2>/dev/null) ]]
    rg -q 'Training complete' "$directory/train.log"
}

[[ -x "$PYTHON" && -x "$CHIMERA" ]] || {
    printf 'Repository .venv Chimera-ML installation is required.\n' >&2
    exit 1
}

for config in "${configs[@]}"; do
    [[ -f "$config" ]] || {
        printf 'Missing config: %s\n' "$config" >&2
        exit 1
    }
    "$CHIMERA" validate-config -c "$config" >/dev/null
    mapfile -t ids < <(identity "$config")
    [[ "${ids[0]}" == "$EXPERIMENT" ]] || exit 1
    [[ "${ids[1]}" == F_task_attention_fold[0-4] ]] || exit 1
    rg -q 'name: carewsm_gated_task_attention' "$config" || exit 1
    rg -q 'monitor: val_mean_corpora_tasks/macro_recall' "$config" || exit 1
    rg -q "$MLFLOW_URI" "$config" || exit 1
    assert_no_collision "$config"
done

if [[ "$mode" == "--check" ]]; then
    printf 'Preflight passed; no training was launched.\n'
    exit 0
fi

"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
if pgrep -af 'chimera-ml train' | rg -v 'rg ' >/dev/null; then
    printf 'An existing Chimera training process is active.\n' >&2
    exit 1
fi

mkdir -p "$LAUNCH_ROOT"
if [[ ! -e "$PROGRESS_LOG" ]]; then
    printf 'configuration\tlaunch_utc\tstate\texit_code\trun_directory\n' > "$PROGRESS_LOG"
fi
printf '[%s] launcher mode=%s session=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$mode" "$SESSION_NAME" | tee -a "$LAUNCH_LOG"

for config in "${configs[@]}"; do
    assert_no_collision "$config"
    mapfile -t ids < <(identity "$config")
    launch_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    expected_directory="$LOG_ROOT/${ids[0]}/${ids[1]}"
    printf '%s\t%s\tSTARTED\t-\t%s\n' \
        "$config" "$launch_utc" "$expected_directory" >> "$PROGRESS_LOG"
    set +e
    CUDA_VISIBLE_DEVICES=0 "$CHIMERA" train -c "$config" 2>&1 | tee -a "$LAUNCH_LOG"
    exit_code=${PIPESTATUS[0]}
    set -e
    if [[ $exit_code -ne 0 ]]; then
        printf '%s\t%s\tFAILED\t%d\t%s\n' \
            "$config" "$launch_utc" "$exit_code" "$expected_directory" >> "$PROGRESS_LOG"
        exit "$exit_code"
    fi
    if ! assert_completed "$config"; then
        printf '%s\t%s\tAMBIGUOUS_AFTER_EXIT\t0\t%s\n' \
            "$config" "$launch_utc" "$expected_directory" >> "$PROGRESS_LOG"
        exit 1
    fi
    completed_directory=$(completed_runtime_directory "$config")
    printf '%s\t%s\tCOMPLETED\t0\t%s\n' \
        "$config" "$launch_utc" "$completed_directory" >> "$PROGRESS_LOG"
done

printf '[%s] launcher finished successfully; selected=%d\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${#configs[@]}" | tee -a "$LAUNCH_LOG"
