#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
CONFIG_ROOT="$REPO_ROOT/configs/audio_representation_pilot/deep008/batch01"
LOG_ROOT="$REPO_ROOT/logs/audio_representation_pilot/deep008_batch01"
EXPERIMENT=audio_representation_pilot_deep008_batch01
MLFLOW_URI=sqlite:///logs/audio_representation_pilot/deep008_batch01/mlflow.db
LAUNCH_ROOT="$LOG_ROOT/launcher"
LAUNCH_LOG="$LAUNCH_ROOT/launcher.log"
PROGRESS_LOG="$LAUNCH_ROOT/progress.tsv"
PYTHON="$REPO_ROOT/.venv/bin/python"
CHIMERA="$REPO_ROOT/.venv/bin/chimera-ml"
SESSION_NAME=carewsm-deep008-batch01

configs=(
    "$CONFIG_ROOT/H1_corpus_balanced_fold0.yaml"
    "$CONFIG_ROOT/H1_corpus_balanced_fold1.yaml"
    "$CONFIG_ROOT/H1_corpus_balanced_fold2.yaml"
    "$CONFIG_ROOT/H1_corpus_balanced_fold3.yaml"
    "$CONFIG_ROOT/H1_corpus_balanced_fold4.yaml"
    "$CONFIG_ROOT/H2_depression_weighted_fold0.yaml"
    "$CONFIG_ROOT/H2_depression_weighted_fold1.yaml"
    "$CONFIG_ROOT/H2_depression_weighted_fold2.yaml"
    "$CONFIG_ROOT/H2_depression_weighted_fold3.yaml"
    "$CONFIG_ROOT/H2_depression_weighted_fold4.yaml"
    "$CONFIG_ROOT/H3_dropout03_fold0.yaml"
    "$CONFIG_ROOT/H3_dropout03_fold1.yaml"
    "$CONFIG_ROOT/H3_dropout03_fold2.yaml"
    "$CONFIG_ROOT/H3_dropout03_fold3.yaml"
    "$CONFIG_ROOT/H3_dropout03_fold4.yaml"
    "$CONFIG_ROOT/H4_hidden32_fold0.yaml"
    "$CONFIG_ROOT/H4_hidden32_fold1.yaml"
    "$CONFIG_ROOT/H4_hidden32_fold2.yaml"
    "$CONFIG_ROOT/H4_hidden32_fold3.yaml"
    "$CONFIG_ROOT/H4_hidden32_fold4.yaml"
)

if [[ $# -ne 1 || ( "$1" != "--check" && "$1" != "--all" ) ]]; then
    printf 'Usage: scripts/run_audio_deep008_batch01.sh --check|--all\n' >&2
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
experiment = client.get_experiment_by_name("audio_representation_pilot_deep008_batch01")
if experiment is None:
    print("COUNT:0")
    raise SystemExit
run_name_prefix = run_name + "_"
runs = [
    run
    for run in client.search_runs([experiment.experiment_id])
    if (run.data.tags.get("mlflow.runName") or "").startswith(run_name_prefix)
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

validate_selected_checkpoint() {
    local summary=$1
    local checkpoint=$2
    "$PYTHON" - "$summary" "$checkpoint" <<'PYCODE'
import csv
import math
import re
import sys
from pathlib import Path

summary_path, checkpoint_path = sys.argv[1:]
with Path(summary_path).open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter=";")
    fields = reader.fieldnames or []
    if "val_mean_corpora_tasks/macro_recall" in fields:
        monitor = "val_mean_corpora_tasks/macro_recall"
    elif "val_mean_corpora_tasks_macro_recall" in fields:
        monitor = "val_mean_corpora_tasks_macro_recall"
    elif "dev_mean_macro_recall" in fields:
        monitor = "dev_mean_macro_recall"
    else:
        raise SystemExit("summary has no canonical validation monitor")

    rows = []
    for row in reader:
        if not row.get("epoch") and not row.get(monitor):
            continue
        try:
            epoch = int(row["epoch"])
            value = float(row[monitor])
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit("summary contains an invalid validation row") from error
        if not math.isfinite(value):
            raise SystemExit("summary contains a non-finite validation value")
        rows.append((epoch, value))

if not rows:
    raise SystemExit("summary contains no validation rows")
selected_epoch, selected_value = max(rows, key=lambda item: (item[1], item[0]))
match = re.fullmatch(
    r"epoch=(\d+)_val_mean_corpora_tasks_macro_recall=(\d+(?:\.\d+)?)\.pt",
    Path(checkpoint_path).name,
)
if match is None:
    raise SystemExit("checkpoint filename does not use the canonical validation monitor")
checkpoint_epoch = int(match.group(1))
checkpoint_value = match.group(2)
precision = len(checkpoint_value.partition(".")[2])
expected_value = f"{selected_value:.{precision}f}"
if checkpoint_epoch != selected_epoch or checkpoint_value != expected_value:
    raise SystemExit("checkpoint does not match the selected validation row")
PYCODE
}

assert_completed() {
    local config=$1
    local directory
    if ! directory=$(completed_runtime_directory "$config"); then
        printf 'Completion failure: runtime directory is missing or ambiguous.\n' >&2
        return 1
    fi
    if [[ ! -f "$directory/train.log" ]]; then
        printf 'Completion failure: train.log is missing.\n' >&2
        return 1
    fi
    if [[ ! -f "$directory/summary.txt" ]]; then
        printf 'Completion failure: summary.txt is missing.\n' >&2
        return 1
    fi
    if ! rg -q 'Training complete' "$directory/train.log"; then
        printf 'Completion failure: train.log has no Training complete marker.\n' >&2
        return 1
    fi

    mapfile -t checkpoints < <(
        find "$directory/checkpoints" -maxdepth 1 -type f -name '*.pt' \
            ! -name 'last.pt' -print 2>/dev/null | sort
    )
    if [[ ${#checkpoints[@]} -ne 1 ]]; then
        printf 'Completion failure: expected one selected checkpoint, found %d.\n' \
            "${#checkpoints[@]}" >&2
        return 1
    fi
    if ! validate_selected_checkpoint "$directory/summary.txt" "${checkpoints[0]}"; then
        printf 'Completion failure: selected checkpoint does not match VAL selection.\n' >&2
        return 1
    fi

    mapfile -t ids < <(identity "$config")
    mapfile -t status < <(mlflow_status "${ids[1]}")
    if [[ "${status[0]:-}" != "FINISHED" || -z "${status[1]:-}" ]]; then
        printf 'Completion failure: exactly one FINISHED MLflow run with an ID is required.\n' >&2
        return 1
    fi
}

if [[ "${CAREWSM_BATCH01_TEST_LIBRARY:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi

check_batch01_config() {
    "$PYTHON" - "$1" "$REPO_ROOT" <<'PYCODE'
import math
import sys
from pathlib import Path
import yaml

path, root = sys.argv[1:]
config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
name = config["experiment_info"]["params"]["run_name"]
loss = config["loss"]
fold = config["data"]["params"]["fold"]
parkinson = [
    2.358878504672897,
    2.375234521575985,
    2.3653308480894686,
    2.4134980988593155,
    2.3594771241830066,
][fold]
depression = [
    1.759119861030689,
    1.747422680412371,
    1.748143917761279,
    1.739749430523918,
    1.740846681922197,
][fold]
if name.startswith("H1_"):
    expected = "carewsm_corpus_balanced_weighted_bce"
    if loss["name"] != expected or not math.isclose(loss["params"]["parkinson_pos_weight"], parkinson, abs_tol=1e-15):
        raise SystemExit("H1 loss contract failed")
elif name.startswith("H2_"):
    expected = "carewsm_weighted_depression_parkinson_bce"
    params = loss["params"]
    if loss["name"] != expected or not math.isclose(params["depression_pos_weight"], depression, abs_tol=1e-15) or not math.isclose(params["parkinson_pos_weight"], parkinson, abs_tol=1e-15):
        raise SystemExit("H2 loss contract failed")
elif name.startswith("H3_"):
    if loss["name"] != "carewsm_weighted_parkinson_bce" or config["model"]["params"]["dropout"] != 0.3:
        raise SystemExit("H3 contract failed")
elif name.startswith("H4_"):
    if loss["name"] != "carewsm_weighted_parkinson_bce" or config["model"]["params"]["hidden_dim"] != 32:
        raise SystemExit("H4 contract failed")
else:
    raise SystemExit("unknown Batch 01 run identity")
PYCODE
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
    [[ "${ids[1]}" == H[1-4]_*_fold[0-4] ]] || exit 1
    rg -q 'name: carewsm_gated_fusion' "$config" || exit 1
    rg -q 'monitor: val_mean_corpora_tasks/macro_recall' "$config" || exit 1
    rg -q "$MLFLOW_URI" "$config" || exit 1
    check_batch01_config "$config"
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
        printf '%s\t%s\tCOMPLETION_FAILED\t0\t%s\n' \
            "$config" "$launch_utc" "$expected_directory" >> "$PROGRESS_LOG"
        exit 1
    fi
    completed_directory=$(completed_runtime_directory "$config")
    printf '%s\t%s\tCOMPLETED\t0\t%s\n' \
        "$config" "$launch_utc" "$completed_directory" >> "$PROGRESS_LOG"
done

printf '[%s] launcher finished successfully; selected=%d\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${#configs[@]}" | tee -a "$LAUNCH_LOG"
