#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_ROOT="$REPO_ROOT/configs/audio_representation_pilot/deep008/weighted_f"
LOG_ROOT="$REPO_ROOT/logs/audio_representation_pilot/deep008_weighted_f"
EXPERIMENT=audio_representation_pilot_deep008_weighted_f
MLFLOW_URI=sqlite:///logs/audio_representation_pilot/deep008_weighted_f/mlflow.db
LAUNCH_ROOT="$LOG_ROOT/launcher"
PROGRESS_LOG="$LAUNCH_ROOT/progress.tsv"
LAUNCH_LOG="$LAUNCH_ROOT/launcher.log"
PYTHON="$REPO_ROOT/.venv/bin/python"
CHIMERA="$REPO_ROOT/.venv/bin/chimera-ml"
SESSION_NAME=carewsm-deep008-weighted-f
configs=(
    "$CONFIG_ROOT/F_weighted_parkinson_fold0.yaml"
    "$CONFIG_ROOT/F_weighted_parkinson_fold1.yaml"
    "$CONFIG_ROOT/F_weighted_parkinson_fold2.yaml"
    "$CONFIG_ROOT/F_weighted_parkinson_fold3.yaml"
    "$CONFIG_ROOT/F_weighted_parkinson_fold4.yaml"
)

[[ $# -eq 1 && ( "$1" == --check || "$1" == --all ) ]] || {
    printf 'Usage: scripts/run_audio_deep008_weighted_f.sh --check|--all\n' >&2
    exit 2
}
mode=$1

identity() {
    "$PYTHON" - "$1" <<'PYCODE'
import sys
import yaml
from pathlib import Path
p = yaml.safe_load(Path(sys.argv[1]).read_text())['experiment_info']['params']
print(p['experiment_name'])
print(p['run_name'])
PYCODE
}

runtime_directory() {
    mapfile -t ids < <(identity "$1")
    base="$LOG_ROOT/${ids[0]}"
    expected="$base/${ids[1]}"
    mapfile -t candidates < <(find "$base" -mindepth 1 -maxdepth 1 -type d -name "${ids[1]}_*" -print 2>/dev/null | sort)
    if [[ -e "$expected" && ${#candidates[@]} -eq 0 ]]; then
        printf '%s\n' "$expected"
    elif [[ -e "$expected" || ${#candidates[@]} -gt 1 ]]; then
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
c = MlflowClient(tracking_uri='sqlite:///logs/audio_representation_pilot/deep008_weighted_f/mlflow.db')
e = c.get_experiment_by_name('audio_representation_pilot_deep008_weighted_f')
if e is None:
    print('COUNT:0')
    raise SystemExit
runs = [r for r in c.search_runs([e.experiment_id]) if (r.data.tags.get('mlflow.runName') or '').startswith(sys.argv[1] + '_')]
if len(runs) != 1:
    print(f'COUNT:{len(runs)}')
else:
    print(runs[0].info.status)
    print(runs[0].info.run_id)
PYCODE
}

completed() {
    local config=$1
    local directory=$2
    [[ -f "$directory/train.log" && -f "$directory/summary.txt" ]] || return 1
    [[ -n $(find "$directory/checkpoints" -maxdepth 1 -name '*.pt' -print -quit 2>/dev/null) ]] || return 1
    rg -q 'Training complete' "$directory/train.log" || return 1
    mapfile -t ids < <(identity "$config")
    mapfile -t status < <(mlflow_status "${ids[1]}")
    [[ "${status[0]}" == FINISHED && -n "${status[1]:-}" ]]
}

check_weights() {
    "$PYTHON" - "$REPO_ROOT" "${configs[@]}" <<'PYCODE'
import csv
import math
import sys
from pathlib import Path
import yaml
root = Path(sys.argv[1])
manifest = {}
with (root / 'reports/feature_audit/sample_feature_availability.csv').open(newline='', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        manifest[(row['corpus'], row['sample_id'])] = row
expected = {0: (1070, 2524, 2.358878504672897), 1: (1066, 2532, 2.375234521575985), 2: (1073, 2538, 2.3653308480894686), 3: (1052, 2539, 2.4134980988593155), 4: (1071, 2527, 2.3594771241830066)}
for path in sys.argv[2:]:
    config = yaml.safe_load(Path(path).read_text())
    fold = config['data']['params']['fold']
    positive = negative = 0
    for corpus, data_root in (('CARE', '/media/maxim/Databases/CARE'), ('WSM', '/media/maxim/Databases/WSM_NEW')):
        rows = list(csv.DictReader((Path(data_root) / 'splits/folds' / f'fold_{fold}_train.csv').open(newline='', encoding='utf-8-sig')))
        for row in rows:
            record = manifest[(corpus, row['sample_id'])]
            if not all(record.get(f'{family}_status') == 'VALID' for family in ('emotionmsp_embeds', 'trillsson4_1')):
                continue
            if record.get('parkinson') == '1':
                positive += 1
            elif record.get('parkinson') == '0':
                negative += 1
    expected_positive, expected_negative, expected_weight = expected[fold]
    configured = config['loss']['params']['parkinson_pos_weight']
    if (positive, negative) != (expected_positive, expected_negative) or not math.isclose(configured, expected_weight, rel_tol=0.0, abs_tol=1e-15):
        raise SystemExit(f'weight gate failed for fold {fold}')
    print(f'fold {fold}: positive={positive} negative={negative} weight={configured:.15f}')
PYCODE
}

[[ -x "$PYTHON" && -x "$CHIMERA" ]] || exit 1
for config in "${configs[@]}"; do
    [[ -f "$config" ]] || exit 1
    "$CHIMERA" validate-config -c "$config" >/dev/null
    mapfile -t ids < <(identity "$config")
    [[ "${ids[0]}" == "$EXPERIMENT" && "${ids[1]}" == F_weighted_parkinson_fold[0-4] ]] || exit 1
    rg -q 'monitor: val_mean_corpora_tasks/macro_recall' "$config" || exit 1
    rg -q "$MLFLOW_URI" "$config" || exit 1
    runtime_directory "$config" >/dev/null
done
check_weights
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
if pgrep -af 'chimera-ml train' | rg -v 'rg ' >/dev/null; then
    printf 'An existing Chimera training process is active.\n' >&2
    exit 1
fi
if [[ "$mode" == --check ]]; then
    printf 'Preflight passed; no training was launched.\n'
    exit 0
fi

mkdir -p "$LAUNCH_ROOT"
[[ -e "$PROGRESS_LOG" ]] || printf 'configuration\tlaunch_utc\tstate\texit_code\trun_directory\n' > "$PROGRESS_LOG"
for config in "${configs[@]}"; do
    mapfile -t ids < <(identity "$config")
    directory=$(runtime_directory "$config")
    now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    if [[ -e "$directory" ]]; then
        if completed "$config" "$directory"; then
            printf '%s\t%s\tSKIPPED_COMPLETED\t0\t%s\n' "$config" "$now" "$directory" >> "$PROGRESS_LOG"
            continue
        fi
        printf '%s\t%s\tAMBIGUOUS_COLLISION\t1\t%s\n' "$config" "$now" "$directory" >> "$PROGRESS_LOG"
        exit 1
    fi
    [[ "$(mlflow_status "${ids[1]}" | head -1)" == COUNT:0 ]] || exit 1
    printf '%s\t%s\tSTARTED\t-\t%s\n' "$config" "$now" "$directory" >> "$PROGRESS_LOG"
    set +e
    CUDA_VISIBLE_DEVICES=0 "$CHIMERA" train -c "$config" 2>&1 | tee -a "$LAUNCH_LOG"
    code=${PIPESTATUS[0]}
    set -e
    [[ $code -eq 0 ]] || { printf '%s\t%s\tFAILED\t%d\t%s\n' "$config" "$now" "$code" "$directory" >> "$PROGRESS_LOG"; exit "$code"; }
    directory=$(runtime_directory "$config")
    completed "$config" "$directory" || { printf '%s\t%s\tAMBIGUOUS_AFTER_EXIT\t0\t%s\n' "$config" "$now" "$directory" >> "$PROGRESS_LOG"; exit 1; }
    printf '%s\t%s\tCOMPLETED\t0\t%s\n' "$config" "$now" "$directory" >> "$PROGRESS_LOG"
done
printf '[%s] launcher finished successfully; selected=%d\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${#configs[@]}" | tee -a "$LAUNCH_LOG"
