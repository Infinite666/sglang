#!/bin/sh
set -eu
# Run the same QA batch twice, flushing HBM between rounds:
# sh examples/runtime/engine/run_oceandisk_offline_batch_chat.sh oceandisk
# Edit BATCH_SIZE / MAX_NEW_TOKENS in oceandisk_offline_batch_chat.py.
MODEL_PATH="${MODEL_PATH:-/home/models/GLM-5.2-W4AFP8/}"
DATASET_PATH="${DATASET_PATH:-/home/z00591349/dataset/multifieldqa_zh.jsonl}"
OPEXTENSION="${OPEXTENSION:-/opt/oceandisk/libop_extension.so}"
USE_SINGLE_PORT="${USE_SINGLE_PORT:-0}"
GIDS_DIR="${GIDS_DIR:-/opt/oceandisk/config}"
GIDS_CONFIG="${GIDS_CONFIG:-${GIDS_DIR%/}/gids_config.txt}"
LAYERWISE_SAVE="${LAYERWISE_SAVE:-true}"
if [ "${1:-}" = oceandisk ]; then
    shift
fi
case "$USE_SINGLE_PORT" in
    0) GIDS_DIR="$(cd -- "$GIDS_DIR" && pwd)/" ;;
    1)
        if [ "$(basename -- "$GIDS_CONFIG")" != gids_config.txt ]; then
            printf 'GIDS_CONFIG must be named gids_config.txt\n' >&2
            exit 1
        fi
        GIDS_DIR="$(cd -- "$(dirname -- "$GIDS_CONFIG")" && pwd)/"
        ;;
    *) printf 'USE_SINGLE_PORT must be 0 or 1\n' >&2; exit 1 ;;
esac
export USE_SINGLE_PORT OPEXTENSION GIDS_DIR
export PYTHONPATH="$(dirname -- "$OPEXTENSION")${PYTHONPATH:+:${PYTHONPATH}}"
set -- --enable-unified-cache-external-linker \
    --unified-cache-external-linker-backend oceandisk \
    --hicache-storage-backend-extra-config "{\"layerwise_save\":${LAYERWISE_SAVE}}" "$@"

script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_DG_USE_NVRTC=1
export CUDA_LAUNCH_BLOCKING=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec "${PYTHON_BIN:-python3}" -u "${script_dir}/oceandisk_offline_batch_chat.py" \
    --model-path "$MODEL_PATH" \
    --dataset-path "$DATASET_PATH" \
    --context-length 32768 \
    --max-total-tokens 170000 \
    --tp-size 8 \
    --trust-remote-code \
    --page-size 64 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
    --moe-dense-tp-size 1 \
    --data-parallel-size 1 \
    --moe-a2a-backend none \
    --disable-prefill-cuda-graph \
    --enable-mixed-chunk \
    --chunked-prefill-size 2048 \
    --max-prefill-tokens 4096 \
    --mem-fraction-static 0.85 \
    "$@"
