#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_ROOT="$REPO_ROOT/configs/audio_representation_pilot/deep008"
LOG_ROOT="$REPO_ROOT/logs/audio_representation_pilot"
LAUNCH_ROOT="$LOG_ROOT/deep008_launcher"
LAUNCH_LOG="$LAUNCH_ROOT/launcher.log"
PROGRESS_LOG="$LAUNCH_ROOT/progress.tsv"
PYTHON="$REPO_ROOT/.venv/bin/python"
CHIMERA="$REPO_ROOT/.venv/bin/chimera-ml"
SESSION_NAME="carewsm-deep008-fold0"

configs=(
    "$CONFIG_ROOT/A_native_mlp64.yaml"
    "$CONFIG_ROOT/B_mean_mlp64.yaml"
    "$CONFIG_ROOT/C_mean_std_mlp64.yaml"
    "$CONFIG_ROOT/D_attentive_mlp64.yaml"
    "$CONFIG_ROOT/E_native_mlp128.yaml"
    "$CONFIG_ROOT/F_gated_emotion_trillsson_native64.yaml"
)

usage() {
    cat <<'EOF'
Usage: scripts/run_audio_deep008_pilot.sh --check|--all

--check   Validate all six configs and inspect fold populations/collisions only.
--all     Train A-F sequentially, skipping only verified completed runs.
EOF
}

if [[ $# -ne 1 || ( "$1" != "--check" && "$1" != "--all" ) ]]; then
    usage >&2
    exit 2
fi
mode=$1

if [[ ! -x "$PYTHON" || ! -x "$CHIMERA" ]]; then
    printf 'The repository .venv Chimera-ML installation is required.\n' >&2
    exit 1
fi

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

expected_run_directory() {
    local config=$1
    mapfile -t identity < <(run_identity "$config")
    printf '%s\n' "$LOG_ROOT/${identity[0]}/${identity[1]}"
}

runtime_directory() {
    local config=$1
    local expected
    expected=$(expected_run_directory "$config")
    mapfile -t identity < <(run_identity "$config")
    local experiment_dir="$LOG_ROOT/${identity[0]}"
    local prefix="${identity[1]}_"
    local candidates=()
    if [[ -d "$experiment_dir" ]]; then
        mapfile -t candidates < <(
            find "$experiment_dir" -mindepth 1 -maxdepth 1 -type d -name "${prefix}*" -print | sort
        )
    fi
    if [[ -e "$expected" && ${#candidates[@]} -eq 0 ]]; then
        printf '%s\n' "$expected"
        return 0
    fi
    if [[ -e "$expected" || ${#candidates[@]} -gt 1 ]]; then
        printf 'Multiple runtime directories found for %s.\n' "$config" >&2
        return 1
    fi
    if [[ ${#candidates[@]} -eq 1 ]]; then
        printf '%s\n' "${candidates[0]}"
    else
        printf '%s\n' "$expected"
    fi
}

is_completed() {
    local run_dir=$1
    [[ -f "$run_dir/train.log" ]] || return 1
    [[ -f "$run_dir/summary.txt" ]] || return 1
    [[ -d "$run_dir/checkpoints" ]] || return 1
    [[ -n $(find "$run_dir/checkpoints" -maxdepth 1 -type f -name '*.pt' -print -quit) ]] || return 1
    rg -q 'Training complete' "$run_dir/train.log"
}

check_population() {
    "$PYTHON" - "$REPO_ROOT" <<'PYCODE'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = root / "reports/feature_audit/sample_feature_availability.csv"
manifest = {}
with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
    for row in csv.DictReader(handle):
        manifest[(row["corpus"], row["sample_id"])] = row

counts = {}
for corpus, expected in (("CARE", 1361), ("WSM", 5640)):
    path = (Path("/media/maxim/Databases/CARE") if corpus == "CARE"
            else Path("/media/maxim/Databases/WSM_NEW")) / "splits/folds/fold_0_train.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8-sig")))
    valid = [manifest[(corpus, row["sample_id"])] for row in rows
             if all(manifest[(corpus, row["sample_id"])].get(f"{family}_status") == "VALID"
                    for family in ("emotionmsp_embeds", "trillsson4_1"))]
    counts[corpus] = len(valid)
    if len(rows) != expected:
        raise SystemExit(f"unexpected {corpus} fold-0 train source count: {len(rows)}")

for corpus, expected in (("CARE", 352), ("WSM", 1429)):
    path = (Path("/media/maxim/Databases/CARE") if corpus == "CARE"
            else Path("/media/maxim/Databases/WSM_NEW")) / "splits/folds/fold_0_val.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8-sig")))
    valid = [manifest[(corpus, row["sample_id"])] for row in rows
             if all(manifest[(corpus, row["sample_id"])].get(f"{family}_status") == "VALID"
                    for family in ("emotionmsp_embeds", "trillsson4_1"))]
    counts[f"{corpus}_val"] = len(valid)
    if len(valid) != expected:
        raise SystemExit(f"unexpected {corpus} fold-0 val retained count: {len(valid)}")

if counts["CARE"] + counts["WSM"] != 6998:
    raise SystemExit(f"unexpected joint fold-0 train retained count: {counts['CARE'] + counts['WSM']}")
print(f"population PASS joint_train={counts['CARE'] + counts['WSM']} care_val={counts['CARE_val']} wsm_val={counts['WSM_val']}")
PYCODE
}

inspect_config() {
    local config=$1
    local run_dir
    run_dir=$(runtime_directory "$config")
    if [[ -e "$run_dir" ]]; then
        if is_completed "$run_dir"; then
            printf 'COMPLETED\t%s\t%s\n' "$config" "$run_dir"
        else
            printf 'AMBIGUOUS\t%s\t%s\n' "$config" "$run_dir"
            return 1
        fi
    else
        printf 'READY\t%s\t%s\n' "$config" "$run_dir"
    fi
}

for config in "${configs[@]}"; do
    [[ -f "$config" ]] || { printf 'Missing config: %s\n' "$config" >&2; exit 1; }
    "$CHIMERA" validate-config -c "$config" >/dev/null
    mapfile -t identity < <(run_identity "$config")
    [[ "${identity[0]}" == "audio_representation_pilot_deep008" ]] || exit 1
    [[ "${identity[1]}" == *_fold0 ]] || exit 1
    rg -q 'sqlite:///logs/audio_representation_pilot/mlflow.db' "$config" || exit 1
    rg -q 'monitor: val_mean_corpora_tasks/macro_recall' "$config" || exit 1
    inspect_config "$config"
done
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
printf '[%s] repository=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$REPO_ROOT" | tee -a "$LAUNCH_LOG"

for config in "${configs[@]}"; do
    if ! run_dir=$(runtime_directory "$config"); then
        printf '%s\t%s\tAMBIGUOUS_COLLISION\t1\t%s\n' "$config" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${run_dir:-unknown}" >> "$PROGRESS_LOG"
        exit 1
    fi
    launch_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    if [[ -e "$run_dir" ]]; then
        if is_completed "$run_dir"; then
            printf '%s\t%s\tSKIPPED_COMPLETED\t0\t%s\n' "$config" "$launch_utc" "$run_dir" >> "$PROGRESS_LOG"
            printf '[%s] skipped verified completed run: %s\n' "$launch_utc" "$config" | tee -a "$LAUNCH_LOG"
            continue
        fi
        printf '%s\t%s\tAMBIGUOUS_COLLISION\t1\t%s\n' "$config" "$launch_utc" "$run_dir" >> "$PROGRESS_LOG"
        printf '[%s] ambiguous existing run; stopping: %s\n' "$launch_utc" "$run_dir" | tee -a "$LAUNCH_LOG" >&2
        exit 1
    fi

    printf '%s\t%s\tSTARTED\t-\t%s\n' "$config" "$launch_utc" "$run_dir" >> "$PROGRESS_LOG"
    printf '[%s] starting %s\n' "$launch_utc" "$config" | tee -a "$LAUNCH_LOG"
    set +e
    CUDA_VISIBLE_DEVICES=0 "$CHIMERA" train -c "$config" 2>&1 | tee -a "$LAUNCH_LOG"
    exit_code=${PIPESTATUS[0]}
    set -e

    if [[ $exit_code -ne 0 ]]; then
        printf '%s\t%s\tFAILED\t%d\t%s\n' "$config" "$launch_utc" "$exit_code" "$run_dir" >> "$PROGRESS_LOG"
        printf '[%s] failed exit_code=%d: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$exit_code" "$config" | tee -a "$LAUNCH_LOG" >&2
        exit "$exit_code"
    fi
    if ! run_dir=$(runtime_directory "$config") || ! is_completed "$run_dir"; then
        printf '%s\t%s\tAMBIGUOUS_AFTER_EXIT\t0\t%s\n' "$config" "$launch_utc" "${run_dir:-unknown}" >> "$PROGRESS_LOG"
        printf '[%s] successful exit but runtime evidence is incomplete or ambiguous: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${run_dir:-unknown}" | tee -a "$LAUNCH_LOG" >&2
        exit 1
    fi
    printf '%s\t%s\tCOMPLETED\t0\t%s\n' "$config" "$launch_utc" "$run_dir" >> "$PROGRESS_LOG"
    printf '[%s] completed %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run_dir" | tee -a "$LAUNCH_LOG"
done

printf '[%s] launcher finished successfully; selected=%d\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${#configs[@]}" | tee -a "$LAUNCH_LOG"
