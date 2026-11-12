#!/bin/sh
# Client: run after P, D and the router are ready.
# Writes client.log, rounds.json and conversations/ to LOG_DIR.
set -x

# ===== 现场配置 =====
# 日志输出目录
export LOG_DIR=/home/pd_logs/rounds
# 语料：直接放到 examples/runtime/engine/Round/doupocangqiong.txt
# ====================

# UCL trace 默认关闭
# export SGLANG_UCL_TRACE=1
# export SGLANG_UCL_TRACE_DIR=${LOG_DIR}

mkdir -p "${LOG_DIR}"

python3 run_rounds.py \
    --base-url http://127.0.0.1:31345 \
    --model /home/models/GLM-5-FP8/ \
    --rounds 10 \
    --batch-size 10 \
    --initial-tokens 16000 \
    --increment-tokens 16000 \
    --output-len 16 \
    --record-response \
    --request-usage \
    --conversation-dir "${LOG_DIR}/conversations" \
    --output-json "${LOG_DIR}/rounds.json" \
    2>&1 | tee -i "${LOG_DIR}/client.log"

# 开启 UCL trace 后取消注释：等待两次 server flush 后生成 report
# python3 ucl_trace_tool.py report "${LOG_DIR}" --wait 3 --rounds-json "${LOG_DIR}/rounds.json"
