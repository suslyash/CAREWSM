#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd "$ROOT"
SESSION=audio_turns_combined; PYTHON="$ROOT/.venv/bin/python"; WORKER="$ROOT/scripts/audio_turns_combined/run_worker.py"; STATUS="$ROOT/logs/audio_turns_combined/launcher/status.json"
usage(){ printf '%s\n' 'Usage: bash scripts/audio_turns_combined/run_all.sh --check|--start|--status|--resume|--attach'; }
[[ $# -eq 1 ]] || { usage >&2; exit 2; }
check(){
  [[ -x "$PYTHON" && -x "$ROOT/.venv/bin/chimera-ml" ]] || return 1
  "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
  command -v tmux >/dev/null || return 1
  for root in /media/maxim/Databases/CARE /media/maxim/Databases/WSM_NEW; do for file in train.csv dev.csv test.csv folds/fold_0_train.csv folds/fold_0_val.csv; do [[ -f "$root/splits/$file" ]] || { echo "missing $root/splits/$file" >&2; return 1; }; done; done
  "$PYTHON" scripts/audio_turns_combined/generate_configs.py stage1
  mapfile -t configs < <(find configs/audio_turns_combined/feature_screening -name '*.yaml' -print | sort); [[ ${#configs[@]} -eq 75 ]] || return 1
  "$ROOT/.venv/bin/chimera-ml" validate-config -c "${configs[0]}" >/dev/null
  "$PYTHON" - "${configs[@]}" <<'PYCFG'
import sys
import yaml
expected = {"A": 1024, "B": 3, "C": 1024, "D": 1024}
for raw in sys.argv[1:]:
    config = yaml.safe_load(open(raw, encoding="utf-8"))
    params = config["data"]["params"]
    configured_families = params["feature_family"]
    families = [configured_families] if isinstance(configured_families, str) else configured_families
    code = "".join({"emotionmsp_embeds": "A", "emotionmsp_trajectories": "B", "trillsson4_1": "C", "wav2vec_xlsr300m": "D"}[family] for family in families)
    if config["model"]["params"]["input_dim"] != sum(expected[item] for item in code):
        raise SystemExit(f"input dimension mismatch: {raw}")
    if params["pooling"] != "mean" or config["experiment_info"]["params"]["experiment_name"] != "audio_turns_combined":
        raise SystemExit(f"namespace or representation mismatch: {raw}")
print(f"validated {len(sys.argv)-1} Stage 1 configurations structurally")
PYCFG
  "$PYTHON" scripts/audio_turns_combined/validate_pooling.py
  if pgrep -af 'chimera-ml train' | grep -v "$$"; then echo 'training process already active' >&2; return 1; fi
  echo 'audio_turns_combined preflight passed'
}
start_worker(){
  tmux has-session -t "$SESSION" 2>/dev/null && { echo 'worker already running' >&2; return 1; }
  mkdir -p logs/audio_turns_combined/launcher
  "$PYTHON" - <<'PYSTATE'
import json
from pathlib import Path
path = Path("logs/audio_turns_combined/launcher/status.json")
if path.exists():
    data = json.loads(path.read_text())
    if data.get("state") in {"STARTING", "RUNNING"}:
        data["state"] = "INTERRUPTED"
        data["error"] = "previous worker state was stale; no tmux session was present"
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PYSTATE
  tmux new-session -d -s "$SESSION" "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 '$PYTHON' '$WORKER' >> logs/audio_turns_combined/launcher/launcher.log 2>&1"
}
case "$1" in
 --check) check;;
 --start) check && start_worker;;
 --resume) start_worker;;
 --attach) exec tmux attach-session -t "$SESSION";;
 --status) [[ -f "$STATUS" ]] && cat "$STATUS" || echo '{"state":"NOT_STARTED"}'; if tmux has-session -t "$SESSION" 2>/dev/null; then echo 'tmux_worker: RUNNING'; else echo 'tmux_worker: STOPPED'; state=$(python3 -c "import json; d=json.load(open(\"$STATUS\")); print(d.get(\"state\", \"NOT_STARTED\"), d.get(\"pending\", 0))" 2>/dev/null || true); [[ "$state" == STARTING* || "$state" == RUNNING* || "$state" == COMPLETED_WITH_FAILURES* || "$state" != *" 0" ]] && echo 'effective_state: INTERRUPTED'; fi;;
 *) usage >&2; exit 2;;
esac
