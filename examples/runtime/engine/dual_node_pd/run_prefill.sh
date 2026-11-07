#!/bin/sh
# 132 prefill: TP8 CP8 DP1 on GPU0-7. Usage: ../oceandisk_pd_tp4.md section 4.
set -x

# One directory per start; "latest" always points at it.
log_dir=/home/j00977581/pd_logs/$(date +%m%d%H%M)
mkdir -p ${log_dir}
ln -sfn ${log_dir} /home/j00977581/pd_logs/latest
git -C /home/j00977581/sglang-pd rev-parse HEAD > ${log_dir}/commit.txt
sha256sum /home/j00977581/gids/lib/*.so > ${log_dir}/gids_lib.sha256

# 141.111.65.* must not go through a proxy
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# Keep physical GPU numbering for the native per-GPU GIDS config files
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export PYTHONPATH=/home/j00977581/sglang-pd/python:/home/j00977581/gids/lib
export OPEXTENSION=/home/j00977581/gids/lib/libop_extension.so
export LIBGIDS=/home/j00977581/gids/lib/libgids_kv_host.so
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600
export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=6
export SGLANG_HEALTH_CHECK_TIMEOUT=600
export SGLANG_TIMEOUT_KEEP_ALIVE=600

# for GIDS kernel
export GIDS_DIR=/home/j00977581/gids/cfg_fbs-h20-01/

# use single od port or full od ports
export USE_SINGLE_PORT=0

# Kernel cache
mkdir -p /home/cache/deep_gemm /home/cache/triton /home/cache/inductor
export SGLANG_DG_CACHE_DIR=/home/cache/deep_gemm
export TRITON_CACHE_DIR=/home/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/home/cache/inductor
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGL_DG_USE_NVRTC=1

# stream number control
export SGLANG_DISAGGREGATION_QUEUE_SIZE=1
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=1

# xinlang config
export SGLANG_ENABLE_JIT_DEEPGEMM=1

# notify
export SGLANG_DISAGGREGATION_LINKER_NOTIFY_LAYERS=50
export SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS=8192
export SGLANG_OCEANDISK_TRACE=0
export SGLANG_OCEANDISK_CACHE_DEBUG=0

# UCL trace
export SGLANG_UCL_TRACE=1
export SGLANG_UCL_TRACE_DIR=${log_dir}
export SGLANG_UCL_TRACE_GPU=1

export SGLANG_HOST_IP=141.111.65.132

# Ctrl+C stops the server, not this script: the report below still runs.
trap : INT

python -m sglang.launch_server \
    --log-level info \
    --model-path /home/models/GLM-5.2-W4AFP8/ \
    --watchdog-timeout 3600 \
    --host 141.111.65.132 \
    --port 31002 \
    --base-gpu-id 0 \
    --tensor-parallel-size 8 \
    --data-parallel-size 1 \
    --moe-dense-tp-size 1 \
    --enable-prefill-cp \
    --cp-strategy interleave \
    --disable-prefill-cuda-graph \
    --mem-fraction-static 0.85 \
    --moe-a2a-backend none \
    --page-size 64 \
    --kv-cache-dtype bf16 \
    --context-length 163840 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
    --enable-metrics \
    --enable-metrics-for-all-schedulers \
    --enable-cache-report \
    --trust-remote-code \
    --attention-backend dsa \
    --chunked-prefill-size 16384 \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device mlx5_1,mlx5_3,mlx5_5,mlx5_7,mlx5_9,mlx5_11,mlx5_13,mlx5_15 \
    --dist-init-addr 141.111.65.132:31500 \
    --disaggregation-bootstrap-port 31145 \
    --disaggregation-mode prefill \
    --disaggregation-linker-notify \
    --enable-unified-cache-external-linker \
    --unified-cache-external-linker-backend oceandisk \
    --hicache-storage-backend-extra-config '{"layerwise_save":true}' \
    2>&1 | tee -i ${log_dir}/prefill.log

python3 /home/j00977581/sglang-pd/examples/runtime/engine/Round/ucl_trace_tool.py report ${log_dir}
