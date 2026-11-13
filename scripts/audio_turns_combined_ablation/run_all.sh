#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd "$ROOT"
SESSION=audio_turns_combined_ablation; PYTHON="$ROOT/.venv/bin/python"; WORKER="$ROOT/scripts/audio_turns_combined_ablation/run_worker.py"; STATUS="$ROOT/logs/audio_turns_combined_ablation/launcher/status.json"
usage(){ printf '%s\n' 'Usage: bash scripts/audio_turns_combined_ablation/run_all.sh --check|--start|--status|--resume|--attach'; }
[[ $# -eq 1 ]] || { usage >&2; exit 2; }
check(){
  [[ -x "$PYTHON" && -x "$ROOT/.venv/bin/chimera-ml" ]] || return 1
  "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
  command -v tmux >/dev/null || return 1
  mapfile -t configs < <(find configs/audio_turns_combined_ablation/ablation -name '*.yaml' -print | sort)
  [[ ${#configs[@]} -eq 5 ]] || { echo "expected exactly five ablation configs" >&2; return 1; }
  "$ROOT/.venv/bin/chimera-ml" validate-config -c "${configs[0]}" >/dev/null
  "$PYTHON" - "${configs[@]}" <<'PYCFG'
import sys
import yaml
for raw in sys.argv[1:]:
    config = yaml.safe_load(open(raw, encoding="utf-8"))
    params = config["data"]["params"]
    model = config["model"]
    if config["experiment_info"]["params"]["experiment_name"] != "audio_turns_combined_ablation":
        raise SystemExit(f"wrong namespace: {raw}")
    if params["feature_family"] != ["emotionmsp_embeds", "trillsson4_1"] or params["pooling"] != "mean":
        raise SystemExit(f"wrong feature or pooling: {raw}")
    if model["name"] != "carewsm_static_gated_fusion" or model["params"]["hidden_dim"] != 64:
        raise SystemExit(f"wrong model or hidden dimension: {raw}")
    if config["loss"]["name"] != "carewsm_weighted_parkinson_bce":
        raise SystemExit(f"wrong loss: {raw}")
print("validated exactly five AC static gated64 configurations")
PYCFG
  if pgrep -af 'chimera-ml train' | grep -v "$$"; then echo 'training process already active' >&2; return 1; fi
  echo 'audio_turns_combined_ablation preflight passed'
}
start_worker(){
  tmux has-session -t "$SESSION" 2>/dev/null && { echo 'worker already running' >&2; return 1; }
  mkdir -p logs/audio_turns_combined_ablation/launcher
  "$PYTHON" - <<'PYSTATE'
import json
from pathlib import Path
path = Path("logs/audio_turns_combined_ablation/launcher/status.json")
if path.exists():
    data = json.loads(path.read_text())
    if data.get("state") in {"STARTING", "RUNNING"}:
        data["state"] = "INTERRUPTED"
        data["error"] = "previous worker state was stale; no tmux session was present"
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PYSTATE
  tmux new-session -d -s "$SESSION" "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PYTHON' '$WORKER' >> logs/audio_turns_combined_ablation/launcher/launcher.log 2>&1"
}
case "$1" in
 --check) check;;
 --start) check && start_worker;;
 --resume) start_worker;;
 --attach) exec tmux attach-session -t "$SESSION";;
 --status)
   [[ -f "$STATUS" ]] && cat "$STATUS" || echo '{"state":"NOT_STARTED"}'
   if tmux has-session -t "$SESSION" 2>/dev/null; then
     echo 'tmux_worker: RUNNING'
   else
     echo 'tmux_worker: STOPPED'
     state=$(python3 -c "import json; d=json.load(open(\"$STATUS\")); print(d.get(\"state\", \"NOT_STARTED\"), d.get(\"pending\", 0))" 2>/dev/null || true)
     if [[ "$state" == STARTING* || "$state" == RUNNING* || "$state" == COMPLETED_WITH_FAILURES* || "$state" != *" 0" ]]; then
       echo 'effective_state: INTERRUPTED'
     fi
   fi
   exit 0
   ;;
 *) usage >&2; exit 2;;
esac
