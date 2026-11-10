#!/bin/sh
set -x

local_ip=141.111.65.134
port=30001
bootstrap_port=31145
model_path=/home/models/GLM-5-FP8
device_name="mlx5_0,mlx5_2,mlx5_4,mlx5_6,mlx5_8,mlx5_10,mlx5_12,mlx5_14"
log_dir=/home/wq/logs/0520
gids_config_dir="/home/wq/code/0520/scripts/decode_multi"
gids_dir="/home/wq/code/0520/GIDS"

export SGLANG_HOST_IP=${local_ip}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=20
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600
export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=6
export SGLANG_DISAGGREGATION_QUEUE_SIZE=2
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=8
export SGLANG_HEALTH_CHECK_TIMEOUT=600
export SGLANG_TIMEOUT_KEEP_ALIVE=600

# Kernel cache
mkdir -p /home/cache/deep_gemm /home/cache/triton /home/cache/inductor
export SGLANG_DG_CACHE_DIR="/home/cache/deep_gemm"
export TRITON_CACHE_DIR="/home/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="/home/cache/inductor"
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGL_DG_USE_NVRTC=1

# xinlang config
export SGLANG_ENABLE_JIT_DEEPGEMM=true
export SGLANG_USE_DEEPGEMM_BMM=true

# for GIDS kernel
export GIDS_DIR=${gids_config_dir}
export PYTHONPATH=${gids_dir}:$PYTHONPATH
export OPEXTENSION=${gids_dir}/libop_extension.so
export USE_SINGLE_PORT=0

# debug
export SGLANG_OCEANDISK_TRACE=0
export SGLANG_OCEANDISK_CACHE_DEBUG=0

python -m sglang.launch_server \
  --model-path ${model_path} \
  --host ${local_ip} \
  --port ${port} \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --incremental-streaming-output \
  --tokenizer-worker-num 32 \
  --tensor-parallel-size 8 \
  --moe-dense-tp-size 1 \
  --data-parallel-size 4 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --mem-fraction-static 0.85 \
  --moe-a2a-backend none \
  --deepep-mode low_latency \
  --load-balance-method total_tokens \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device ${device_name} \
  --disaggregation-bootstrap-port ${bootstrap_port} \
  --dist-init-addr ${local_ip}:5000 \
  --page-size 64 \
  --max-running-requests 128 \
  --cuda-graph-max-bs-decode 128 \
  --num-reserved-decode-tokens 512 \
  --dsa-decode-backend flashmla_kv \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
  --enable-metrics-for-all-schedulers \
  --enable-metrics \
  --enable-cache-report \
  --trust-remote-code \
  --log-level INFO \
  --watchdog-timeout 3600 \
  --disaggregation-decode-enable-radix-cache \
  --disaggregation-linker-notify \
  --enable-unified-cache-external-linker \
  --unified-cache-external-linker-backend oceandisk \
  --hicache-storage-backend-extra-config '{"layerwise_save":true}' \
  2>&1 | tee -i ${log_dir}/decode_$(date +%Y%m%d_%H%M%S).log
