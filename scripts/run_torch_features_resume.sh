#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="/media/maxim/Programs/Projects/CAREWSM"
VENV_DIR="$PROJECT_DIR/.venv-audio"
DB_DIR="/media/maxim/Databases/WSM_NEW"
INPUT_DIR="/tmp/carewsm_registered_wavs"
LOG_FILE="$PROJECT_DIR/logs/wsm_torch_features.log"

mkdir -p "$PROJECT_DIR/logs"
exec >>"$LOG_FILE" 2>&1

export LD_LIBRARY_PATH="$VENV_DIR/lib/python3.10/site-packages/nvidia/cublas/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cudnn/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cufft/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/curand/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cusolver/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cusparse/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/nccl/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME="/media/maxim/Programs/ml_cache/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run() {
    local script="$1"
    for dataset in depression parkinson; do
        echo "===== $script / $dataset $(date '+%F %T') ====="
        "$VENV_DIR/bin/python" "$PROJECT_DIR/care_scripts/audio_descriptors/$script" \
            --audio-dir "$INPUT_DIR/$dataset" \
            --output-dir "$DB_DIR/extracted" \
            --metadata-csv "$DB_DIR/splits/videos.csv"
    done
}

echo "===== START TORCH FEATURES $(date '+%F %T') ====="
run extract_wav2vec.py
run extract_audeering_emotionmsp.py
echo "===== COMPLETED TORCH FEATURES $(date '+%F %T') ====="
