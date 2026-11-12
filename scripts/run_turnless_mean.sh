#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_ROOT="$REPO_ROOT/configs/audio_representation_pilot/turnless_mean"
LOG_ROOT="$REPO_ROOT/logs/audio_representation_pilot/turnless_mean"
LAUNCH_ROOT="$LOG_ROOT/launcher"
PROGRESS_LOG="$LAUNCH_ROOT/progress.tsv"
LAUNCH_LOG="$LAUNCH_ROOT/launcher.log"
PYTHON="$REPO_ROOT/.venv/bin/python"
CHIMERA="$REPO_ROOT/.venv/bin/chimera-ml"

[[ $# -eq 1 && ( "$1" == --check || "$1" == --all ) ]] || {
    printf 'Usage: scripts/run_turnless_mean.sh --check|--all\n' >&2
    exit 2
}
[[ -x "$PYTHON" && -x "$CHIMERA" ]] || exit 1
[[ -f "$REPO_ROOT/reports/feature_audit/sample_feature_availability.csv" ]] || exit 1

"$PYTHON" "$REPO_ROOT/scripts/create_turnless_mean_configs.py" >/dev/null
"$PYTHON" "$REPO_ROOT/scripts/verify_turnless_mean.py" \
    --output "$LOG_ROOT/data_verification.json" >/dev/null

mapfile -t configs < <(find "$CONFIG_ROOT" -maxdepth 1 -name '*.yaml' -type f | sort)
[[ ${#configs[@]} -eq 20 ]] || exit 1
for config in "${configs[@]}"; do
    "$CHIMERA" validate-config -c "$config" >/dev/null
done
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
if pgrep -af 'chimera-ml train' | rg -v 'rg ' >/dev/null; then
    printf 'An existing Chimera training process is active.\n' >&2
    exit 1
fi
if [[ "$1" == --check ]]; then
    printf 'Preflight passed; no training was launched.\n'
    exit 0
fi

mkdir -p "$LAUNCH_ROOT"
[[ -e "$PROGRESS_LOG" ]] || printf 'configuration\tstate\texit_code\trun_name\n' > "$PROGRESS_LOG"
for config in "${configs[@]}"; do
    run_name=$(
        "$PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["experiment_info"]["params"]["run_name"])' "$config"
    )
    if rg -q "^${run_name}\\tCOMPLETED\\t" "$PROGRESS_LOG"; then
        continue
    fi
    printf '%s\tSTARTED\t-\t%s\n' "$config" "$run_name" >> "$PROGRESS_LOG"
    set +e
    CUDA_VISIBLE_DEVICES=0 "$CHIMERA" train -c "$config" 2>&1 | tee -a "$LAUNCH_LOG"
    code=${PIPESTATUS[0]}
    set -e
    if [[ $code -ne 0 ]]; then
        printf '%s\tFAILED\t%d\t%s\n' "$config" "$code" "$run_name" >> "$PROGRESS_LOG"
        "$PYTHON" "$REPO_ROOT/scripts/collect_turnless_mean_results.py" --allow-incomplete || true
        exit "$code"
    fi
    printf '%s\tCOMPLETED\t0\t%s\n' "$config" "$run_name" >> "$PROGRESS_LOG"
    "$PYTHON" "$REPO_ROOT/scripts/collect_turnless_mean_results.py" --allow-incomplete
done
"$PYTHON" "$REPO_ROOT/scripts/collect_turnless_mean_results.py"
printf '[%s] turnless mean launcher finished successfully.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LAUNCH_LOG"
