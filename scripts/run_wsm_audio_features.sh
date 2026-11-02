#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="/media/maxim/Programs/Projects/CAREWSM"
VENV_DIR="$PROJECT_DIR/.venv-audio"
DB_DIR="/media/maxim/Databases/WSM_NEW"
INPUT_DIR="/tmp/carewsm_registered_wavs"
OUTPUT_DIR="$DB_DIR/extracted"
METADATA_CSV="$DB_DIR/splits/videos.csv"

export TFHUB_CACHE_DIR="/media/maxim/Programs/ml_cache/tensorflow_hub"
export LD_LIBRARY_PATH="$VENV_DIR/lib/python3.10/site-packages/nvidia/cublas/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cudnn/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cufft/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/curand/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cusolver/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cusparse/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/nccl/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

PYTHON="$VENV_DIR/bin/python"
SCRIPT_DIR="$PROJECT_DIR/care_scripts/audio_descriptors"

run() {
    echo
    echo "===== $(date '+%F %T') :: $* ====="
    for dataset in depression parkinson; do
        echo "----- dataset: $dataset -----"
    "$PYTHON" "$SCRIPT_DIR/$1" \
        --audio-dir "$INPUT_DIR/$dataset" \
        --output-dir "$OUTPUT_DIR" \
        --metadata-csv "$METADATA_CSV"
    done
}

cd "$PROJECT_DIR"

run extract_egemaps.py
run extract_disvoice.py
run extract_wav2vec.py
run extract_trillsson.py
run extract_audeering_emotionmsp.py

echo
echo "===== $(date '+%F %T') :: ALL AUDIO FEATURES COMPLETED ====="
