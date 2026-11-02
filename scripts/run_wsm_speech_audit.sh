#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="/media/maxim/Programs/Projects/CAREWSM"
VENV_DIR="$PROJECT_DIR/.venv-audio"
LOG_FILE="$PROJECT_DIR/logs/wsm_speech_content_audit.log"

mkdir -p "$PROJECT_DIR/logs"
exec > >(tee -a "$LOG_FILE") 2>&1

export HF_HOME="/media/maxim/Programs/ml_cache/hf"
export HF_HUB_CACHE="/media/maxim/Programs/ml_cache/hf/hub"
export HF_HUB_OFFLINE=1
export TFHUB_CACHE_DIR="/media/maxim/Programs/ml_cache/tensorflow_hub"
export TORCH_HOME="/media/maxim/Programs/ml_cache/torch"
export LD_LIBRARY_PATH="$VENV_DIR/lib/python3.10/site-packages/nvidia/cudnn/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cublas/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cusolver/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"

cd "$PROJECT_DIR"
echo "===== START WSM SPEECH CONTENT AUDIT $(date '+%F %T') ====="
"$VENV_DIR/bin/python" scripts/audit_wsm_speech_content.py \
  --n 200 \
  --seed 20261021 \
  --output logs/wsm_speech_content_audit.csv
echo "===== COMPLETED WSM SPEECH CONTENT AUDIT $(date '+%F %T') ====="
