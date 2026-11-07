#!/bin/sh
# 134 client: run after P, D and the router are ready. Writes rounds.json,
# conversations/ and report.md, summary.json, trace.json next to decode.log.
set -x

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

export SGLANG_UCL_TRACE=1
export SGLANG_UCL_TRACE_DIR=/home/j00977581/pd_logs/latest

cd /home/j00977581/sglang-pd/examples/runtime/engine/Round
[ -e doupocangqiong.txt ] || ln -s /home/j00977581/Round/doupocangqiong.txt doupocangqiong.txt

python3 run_rounds.py \
    --base-url http://127.0.0.1:31345 \
    --model /home/models/GLM-5.2-W4AFP8/ \
    --rounds 10 \
    --batch-size 10 \
    --initial-tokens 16000 \
    --increment-tokens 16000 \
    --output-len 16 \
    --record-response \
    --request-usage \
    --conversation-dir /home/j00977581/pd_logs/latest/conversations \
    --output-json /home/j00977581/pd_logs/latest/rounds.json \
    2>&1 | tee -i /home/j00977581/pd_logs/latest/client.log

# Wait for two server flushes; P's half of the report stays on 132.
python3 ucl_trace_tool.py report /home/j00977581/pd_logs/latest --wait 3
