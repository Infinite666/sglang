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

# notify
export SGLANG_DISAGGREGATION_LINKER_NOTIFY_LAYERS=50
export SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS=8192
export SGLANG_DISAGGREGATION_LINKER_LOAD_PAGES=0
export SGLANG_OCEANDISK_TRACE=0
export SGLANG_OCEANDISK_CACHE_DEBUG=0

python -m sglang.launch_server \
  --watchdog-timeout 3600 \
  --log-level INFO \
  --model-path ${model_path} \
  --disaggregation-mode decode \
  --host ${local_ip} \
  --port 30001 \
  --tensor-parallel-size 8 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-dense-tp-size 1 \
  --mem-fraction-static 0.85 \
  --max-running-requests 128 \
  --deepep-mode low_latency \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device ${device_name} \
  --dist-init-addr ${local_ip}:5000 \
  --disaggregation-bootstrap-port 9145 \
  --page-size 64 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
  --enable-metrics-for-all-schedulers \
  --enable-metrics \
  --enable-cache-report \
  --trust-remote-code \
  --disaggregation-bootstrap-port ${bootstrap_port} \
  --disaggregation-linker-notify \
  --enable-unified-cache-external-linker \
  --unified-cache-external-linker-backend oceandisk \
  --hicache-storage-backend-extra-config '{"layerwise_save":true}' \
  --speculative-draft-load-format auto \
  --disable-radix-cache \
  --dsa-decode-backend flashmla_kv \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --kv-cache-dtype fp8_e4m3 \
  --data-parallel-size 8 \
  --tokenizer-worker-num 32 \
  --num-reserved-decode-tokens 512 \
  --cuda-graph-max-bs-decode 32 \
  --moe-a2a-backend deepep \
  --speculative-moe-a2a-backend deepep \
  --load-balance-method total_tokens \
  --enable-hisparse \
  --hisparse-config "{\"top_k\": 2048, \"device_buffer_size\": 6144, \"page_size\": 64, \"host_to_device_ratio\": 10, \"swap_in_block_size\": 960}" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --incremental-streaming-output \
  2>&1 | tee -i ${log_dir}/decode_$(date +%Y%m%d_%H%M%S).log
