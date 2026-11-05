#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_ROOT="$REPO_ROOT/configs/audio_deep_features/native_turn_fusion"
LOG_ROOT="$REPO_ROOT/logs/audio_deep_features"
LAUNCH_ROOT="$LOG_ROOT/fusion_launcher"
LAUNCH_LOG="$LAUNCH_ROOT/launcher.log"
PROGRESS_LOG="$LAUNCH_ROOT/progress.tsv"
PYTHON="$REPO_ROOT/.venv/bin/python"
CHIMERA="$REPO_ROOT/.venv/bin/chimera-ml"
CANARY_CONFIG="$CONFIG_ROOT/emotionmsp_embeds__trillsson4_1/mlp_fold0.yaml"
SESSION_NAME="carewsm-deep-fusion"

usage() {
    cat <<'EOF'
Usage: scripts/run_audio_deep_fusion.sh --check|--canary|--all

--check   Validate the 110-config manifest and inspect run collisions only.
--canary  Run the approved emotionmsp_embeds + trillsson4_1 MLP fold-0 config.
--all     Run the sorted 110-config manifest sequentially, skipping verified
          completed runs and stopping on ambiguous or failed runs.
EOF
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

mode=$1
case "$mode" in
    --check|--canary|--all) ;;
    *) usage >&2; exit 2 ;;
esac

mapfile -t configs < <(find "$CONFIG_ROOT" -type f -name '*.yaml' -print | sort)
if [[ ${#configs[@]} -ne 110 ]]; then
    printf 'Expected exactly 110 native-turn fusion YAMLs, found %d.\n' "${#configs[@]}" >&2
    exit 1
fi

if [[ ! -x "$PYTHON" || ! -x "$CHIMERA" ]]; then
    printf 'The repository .venv Chimera-ML installation is required.\n' >&2
    exit 1
fi

run_identity() {
    "$PYTHON" - "$1" <<'PY'
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
info = config["experiment_info"]["params"]
print(info["experiment_name"])
print(info["run_name"])
PY
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

if [[ "$mode" == "--check" ]]; then
    printf 'Checking exactly 110 configs in deterministic lexical order.\n'
    for config in "${configs[@]}"; do
        "$CHIMERA" validate-config -c "$config" >/dev/null
        inspect_config "$config"
    done
    printf 'Preflight passed; no training was launched.\n'
    exit 0
fi

mkdir -p "$LAUNCH_ROOT"
if [[ ! -e "$PROGRESS_LOG" ]]; then
    printf 'configuration\tlaunch_utc\tstate\texit_code\trun_directory\n' > "$PROGRESS_LOG"
fi

if [[ "$mode" == "--canary" ]]; then
    selected=("$CANARY_CONFIG")
else
    selected=("${configs[@]}")
fi

printf '[%s] launcher mode=%s session=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$mode" "$SESSION_NAME" | tee -a "$LAUNCH_LOG"
printf '[%s] repository=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$REPO_ROOT" | tee -a "$LAUNCH_LOG"

for config in "${selected[@]}"; do
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
        printf '[%s] process exited successfully but runtime evidence is incomplete or ambiguous: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${run_dir:-unknown}" | tee -a "$LAUNCH_LOG" >&2
        exit 1
    fi

    printf '%s\t%s\tCOMPLETED\t0\t%s\n' "$config" "$launch_utc" "$run_dir" >> "$PROGRESS_LOG"
    printf '[%s] completed %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run_dir" | tee -a "$LAUNCH_LOG"
done

printf '[%s] launcher finished successfully; selected=%d\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${#selected[@]}" | tee -a "$LAUNCH_LOG"
