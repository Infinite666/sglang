#!/bin/sh
set -eu

# GIDS_DIR 内放 gids_config_0.txt ... gids_config_7.txt。
# sh examples/runtime/engine/run_oceandisk_online_tp8.sh
# 关闭逐层写出：LAYERWISE_SAVE=false sh examples/runtime/engine/run_oceandisk_online_tp8.sh
LAYERWISE_SAVE="${LAYERWISE_SAVE:-true}"
MODEL_PATH="${MODEL_PATH:-/home/models/GLM-5.2-W4AFP8/}"
HOST="${HOST:-localhost}"
PORT="${PORT:-9101}"
OPEXTENSION="${OPEXTENSION:-/opt/oceandisk/libop_extension.so}"
# 0：每卡独立配置；1：所有卡使用 GIDS_CONFIG。
USE_SINGLE_PORT="${USE_SINGLE_PORT:-0}"
GIDS_DIR="${GIDS_DIR:-/opt/oceandisk/config}"
GIDS_CONFIG="${GIDS_CONFIG:-${GIDS_DIR%/}/gids_config.txt}"

case "$USE_SINGLE_PORT" in
    0)
        GIDS_DIR="$(cd -- "$GIDS_DIR" && pwd)/"
        ;;
    1)
        if [ "$(basename -- "$GIDS_CONFIG")" != gids_config.txt ]; then
            printf 'GIDS_CONFIG must point to a file named gids_config.txt\n' >&2
            exit 1
        fi
        GIDS_DIR="$(cd -- "$(dirname -- "$GIDS_CONFIG")" && pwd)/"
        ;;
    *)
        printf 'USE_SINGLE_PORT must be 0 or 1\n' >&2
        exit 1
        ;;
esac

script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"

export USE_SINGLE_PORT OPEXTENSION GIDS_DIR
export PYTHONPATH="${repo_root}/python:$(dirname -- "$OPEXTENSION")${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_DG_USE_NVRTC=1
export CUDA_LAUNCH_BLOCKING=0

exec "${PYTHON_BIN:-python3}" -u -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tp-size 8 \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --page-size 64 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
    --enable-metrics \
    --enable-cache-report \
    --moe-dense-tp-size 1 \
    --data-parallel-size 1 \
    --moe-a2a-backend none \
    --mem-fraction-static 0.85 \
    --context-length 170000 \
    --max-total-tokens 170000 \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --disable-prefill-cuda-graph \
    --enable-unified-cache-external-linker \
    --unified-cache-external-linker-backend oceandisk \
    --hicache-storage-backend-extra-config "{\"layerwise_save\":${LAYERWISE_SAVE}}" \
    "$@"
